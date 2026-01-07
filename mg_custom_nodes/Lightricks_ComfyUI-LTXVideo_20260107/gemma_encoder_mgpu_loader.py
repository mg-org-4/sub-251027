import json
import logging
import weakref
from pathlib import Path
from types import MethodType

import accelerate
import comfy.sd
import comfy.supported_models_base
import folder_paths
import safetensors
import torch
from comfy.ldm.lightricks.model import LTXFrequenciesPrecision, LTXRopeType
from comfy.utils import load_torch_file
from transformers import (
    AutoImageProcessor,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
)

from .embeddings_connector import Embeddings1DConnector
from .gemma_encoder import (
    GemmaFeaturesExtractorProjLinear,
    LTXVGemmaTextEncoderModel,
    find_matching_dir,
    ltxv_gemma_tokenizer,
)
from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)

PREFIX_BASE = "model.diffusion_model."


def _is_device_like(x):
    return isinstance(x, (str, torch.device, int)) or isinstance(
        x, torch.Tensor
    )  # to(tensor) → matches tensor's device/dtype


def _is_dtype_like(x):
    # torch.float16 / torch.bfloat16 / torch.float32, etc.
    return isinstance(x, torch.dtype)


def protect_module_transfers(module: torch.nn.Module, warn=False):
    """
    Make .to/.cuda/.cpu NO-OPs for device moves across an entire module tree,
    while still allowing dtype/memory_format conversions.
    """

    def wrap_one(m: torch.nn.Module):
        if hasattr(m, "__to_patched"):  # idempotent
            return

        orig_to = m.__class__.to
        proxy = weakref.proxy(m)

        def safe_to(self, *args, **kwargs):
            # Block explicit/implicit device moves
            device_in_kwargs = "device" in kwargs
            device_like_positional = len(args) > 0 and _is_device_like(args[0])

            # Allow pure dtype/memory_format conversions:
            #   .to(torch.float16)
            #   .to(dtype=torch.float16)
            #   .to(memory_format=...)

            if device_in_kwargs or device_like_positional:
                if warn:
                    print("[protect_module_transfers] Ignoring device move .to(...).")
                return self  # NO-OP for device moves

            # Otherwise (dtype / memory_format / non_blocking without device) → delegate
            return orig_to(self, *args, **kwargs)

        # Bind methods
        m.to = MethodType(safe_to, proxy)
        m.cuda = MethodType(lambda self, *a, **k: self, proxy)  # NO-OP
        m.cpu = MethodType(lambda self, *a, **k: self, proxy)  # NO-OP
        m.__to_patched = True

    # Recurse over the whole tree
    for sub in module.modules():  # includes module itself and children
        wrap_one(sub)

    return module


def load_proj_matrix_from_ltxv(ltxv_path: Path, prefix=""):
    with safetensors.safe_open(ltxv_path, framework="pt", device="cpu") as f:
        keys = filter(lambda key: key.startswith(prefix), f.keys())
        sd = {k.removeprefix(prefix): f.get_tensor(k) for k in keys if k in f.keys()}

    if not sd:
        return None

    with torch.device("meta"):
        model = GemmaFeaturesExtractorProjLinear()

    for name, tensor in sd.items():
        accelerate.utils.set_module_tensor_to_device(model, name, "cpu", value=tensor)

    device_map = accelerate.infer_auto_device_map(
        model,
        max_memory=None,
    )

    model = accelerate.dispatch_model(
        model,
        device_map=device_map,
    )

    return model


def load_proj_matrix_from_checkpoint(checkpoint_path: Path):
    """
    Load model weights from a checkpoint file.
    :param checkpoint_path: Path to the checkpoint file.
    """
    with torch.device("meta"):
        model = GemmaFeaturesExtractorProjLinear()
    model = accelerate.load_checkpoint_and_dispatch(
        model, str(checkpoint_path.resolve()), device_map="auto"
    )
    return model


def ltxv_gemma_clip(encoder_path, ltxv_path, processor=None, dtype=None):
    class _LTXVGemmaTextEncoderModel(LTXVGemmaTextEncoderModel):
        def __init__(self, device="cpu", dtype=dtype, model_options={}):
            dtype = torch.bfloat16  # TODO: make this configurable
            gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                encoder_path,
                local_files_only=True,
                torch_dtype=dtype,
                device_map="auto",
            )

            feature_extractor_linear = load_proj_matrix_from_ltxv(
                ltxv_path, "text_embedding_projection."
            )
            if feature_extractor_linear is None:
                feature_extractor_linear = load_proj_matrix_from_checkpoint(
                    encoder_path / "proj_linear.safetensors"
                )

            embeddings_connector = load_video_embeddings_connector(ltxv_path)
            audio_embeddings_connector = load_audio_embeddings_connector(ltxv_path)
            super().__init__(
                model=gemma_model,
                feature_extractor_linear=feature_extractor_linear,
                embeddings_connector=embeddings_connector,
                audio_embeddings_connector=audio_embeddings_connector,
                processor=processor,
                dtype=dtype,
                device=device,
            )

            # Due to memory management issue on mgpu we need to disable the offloading of sharded gemma model.
            protect_module_transfers(self)

    return _LTXVGemmaTextEncoderModel


def load_video_embeddings_connector(ltxv_path, dtype=torch.bfloat16):
    sd, metadata = load_torch_file(str(ltxv_path), return_metadata=True)
    config = json.loads(metadata.get("config", "{}"))
    transformer_config = config.get("transformer", {})
    rope_type = LTXRopeType.from_dict(transformer_config)
    frequencies_precision = LTXFrequenciesPrecision.from_dict(transformer_config)
    pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])
    sd_keys = list(sd.keys())

    video_only_connector_prefix = f"{PREFIX_BASE}embeddings_connector."
    av_connector_prefix = f"{PREFIX_BASE}video_embeddings_connector."
    prefix = (
        av_connector_prefix
        if f"{PREFIX_BASE}audio_adaln_single.linear.weight" in sd_keys
        else video_only_connector_prefix
    )
    return load_embeddings_connector(
        sd, prefix, dtype, rope_type, frequencies_precision, pe_max_pos
    )


def load_audio_embeddings_connector(ltxv_path, dtype=torch.bfloat16):
    sd, metadata = load_torch_file(str(ltxv_path), return_metadata=True)
    config = json.loads(metadata.get("config", "{}"))
    transformer_config = config.get("transformer", {})
    rope_type = LTXRopeType.from_dict(transformer_config)
    frequencies_precision = LTXFrequenciesPrecision.from_dict(transformer_config)
    pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])
    return load_embeddings_connector(
        sd,
        f"{PREFIX_BASE}audio_embeddings_connector.",
        dtype,
        rope_type,
        frequencies_precision,
        pe_max_pos,
    )


def load_embeddings_connector(
    sd,
    connector_prefix,
    dtype=torch.bfloat16,
    rope_type=LTXRopeType.INTERLEAVED,
    frequencies_precision=LTXFrequenciesPrecision.FLOAT32,
    pe_max_pos=None,
):
    sd_connector = {
        k[len(connector_prefix) :]: v
        for k, v in sd.items()
        if k.startswith(connector_prefix)
    }

    if len(sd_connector) == 0:
        return None

    operations = comfy.ops.pick_operations(dtype, dtype, disable_fast_fp8=True)
    connector = Embeddings1DConnector(
        dtype=dtype,
        operations=operations,
        positional_embedding_max_pos=pe_max_pos if pe_max_pos is not None else [1],
        split_rope=rope_type == LTXRopeType.SPLIT,
        double_precision_rope=frequencies_precision == LTXFrequenciesPrecision.FLOAT64,
    )
    connector.load_state_dict(sd_connector)
    device_map = accelerate.infer_auto_device_map(connector)
    connector = accelerate.dispatch_model(connector, device_map=device_map)
    return connector


@comfy_node(
    name="LTXVGemmaCLIPModelLoaderMGPU", description="Gemma 3 Model Loader on MGPU"
)
class LTXVGemmaCLIPModelLoaderMGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gemma_path": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"tooltip": "The name of the text encoder model to load."},
                ),
                "ltxv_path": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the ltxv model to load."},
                ),
                "max_length": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 131072, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_model"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Gemma CLIP Loader"
    OUTPUT_NODE = False

    def load_model(self, gemma_path: str, ltxv_path: str, max_length: int):
        path = Path(folder_paths.get_full_path("text_encoders", gemma_path))
        model_root = path.parents[1]
        tokenizer_path = Path(find_matching_dir(model_root, "tokenizer.model"))
        gemma_model_path = Path(find_matching_dir(model_root, "model*.safetensors"))

        tokenizer_class = ltxv_gemma_tokenizer(tokenizer_path, max_length=max_length)

        processor = None
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                str(model_root),
                local_files_only=True,
            )
            processor = Gemma3Processor(
                image_processor=image_processor,
                tokenizer=tokenizer_class().tokenizer,
            )
            logger.info(f"Loaded processor from {model_root} - enhancement enabled")
        except Exception as e:
            logger.warning(f"Could not load processor from {model_root}: {e}")

        clip_dtype = torch.bfloat16
        ltxv_full_path = folder_paths.get_full_path("checkpoints", ltxv_path)
        clip_target = comfy.supported_models_base.ClipTarget(
            tokenizer=tokenizer_class,
            clip=ltxv_gemma_clip(
                gemma_model_path, ltxv_full_path, processor=processor, dtype=clip_dtype
            ),
        )

        return (comfy.sd.CLIP(clip_target),)
