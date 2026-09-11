# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path


REQUIRED_PROOF_NAMES = ("processor", "video", "nvfp4", "telemetry", "gpu")

COMFY_ROOT = Path(__file__).resolve().parents[2]
if str(COMFY_ROOT) not in sys.path:
    sys.path.append(str(COMFY_ROOT))


def _read_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _model_path_info(model_path: str) -> dict:
    path = Path(model_path or "")
    info = {
        "path_kind": "missing",
        "is_gguf_file": False,
        "is_hf_repo": False,
        "model_type": "",
        "architecture": "",
        "quant_algo": "",
        "quant_method": "",
        "quant_producer": "",
        "quant_producer_version": "",
        "transformers_version_required": "",
        "has_generation_config": False,
        "has_modelopt_state": False,
        "runtime_hint": "",
    }
    if not path.exists():
        return info
    if path.is_file():
        info["path_kind"] = "gguf_file" if path.suffix.lower() == ".gguf" else "file"
        info["is_gguf_file"] = path.suffix.lower() == ".gguf"
        if info["is_gguf_file"]:
            info["runtime_hint"] = "llama-diffusion-cli GGUF"
        return info
    if not path.is_dir():
        info["path_kind"] = "unknown"
        return info

    config = _read_json_file(path / "config.json")
    hf_quant_config = _read_json_file(path / "hf_quant_config.json")
    quant_config = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
    if not quant_config:
        quant_config = hf_quant_config.get("quantization") if isinstance(hf_quant_config.get("quantization"), dict) else {}
    producer = quant_config.get("producer") if isinstance(quant_config.get("producer"), dict) else {}
    if not producer:
        producer = hf_quant_config.get("producer") if isinstance(hf_quant_config.get("producer"), dict) else {}
    architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
    info.update(
        {
            "is_hf_repo": bool(config),
            "model_type": str(config.get("model_type", "")),
            "architecture": str(architectures[0]) if architectures else "",
            "quant_algo": str(quant_config.get("quant_algo", "")),
            "quant_method": str(quant_config.get("quant_method", "")),
            "quant_producer": str(producer.get("name", "")),
            "quant_producer_version": str(producer.get("version", "")),
            "transformers_version_required": str(config.get("transformers_version", "")),
            "has_generation_config": (path / "generation_config.json").exists(),
            "has_modelopt_state": (path / "modelopt_state.pth").exists(),
        }
    )
    if info["quant_algo"].upper() == "NVFP4":
        info["path_kind"] = "nvfp4_hf_repo"
        info["runtime_hint"] = "NVIDIA ModelOpt NVFP4 checkpoint"
    elif config:
        info["path_kind"] = "hf_repo"
        info["runtime_hint"] = "Transformers whole-repo model"
    else:
        info["path_kind"] = "directory"
    return info


def _short_video_processor_smoke(model_path: str) -> dict:
    result = {
        "passed": False,
        "frame_count": 11,
        "transport": "",
        "input_keys": [],
        "pixel_values_shape": [],
        "pixel_values_videos_shape": [],
        "transport_proof": {},
        "counterfactual_frame_count": 3,
        "counterfactual_transport_proof": {},
        "counterfactual_shape_changed": False,
        "error": "",
    }
    try:
        import torch
        from types import SimpleNamespace
        from transformers import AutoProcessor
        from nodes import (
            MediaContext,
            _processor_transport_proof,
            _transformers_messages_and_processor_kwargs,
            build_asset_registry,
        )

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model_stub = SimpleNamespace(
            config=SimpleNamespace(image_token_id=None, video_token_id=None)
        )

        def process(frame_count: int) -> tuple[dict, dict]:
            frames = torch.zeros((frame_count, 64, 64, 3), dtype=torch.float32)
            frames[:, :, :, 0] = torch.linspace(0.0, 1.0, frame_count).view(
                frame_count, 1, 1
            )
            metadata = {
                "source": "video",
                "duration_seconds": float(frame_count),
                "sample_fps": 1.0,
                "sampled_frame_count": frame_count,
                "video_sampled_frame_count": frame_count,
                "sampled_indices": list(range(frame_count)),
                "source_fps": 1.0,
                "width": 64,
                "height": 64,
            }
            context = MediaContext(images=frames, source="video", metadata=metadata)
            messages, processor_kwargs = _transformers_messages_and_processor_kwargs(
                "Describe this video.", context, visual_first=True
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                processor_kwargs=processor_kwargs,
                enable_thinking=False,
            )
            proof = _processor_transport_proof(
                inputs,
                model_stub,
                processor,
                messages,
                build_asset_registry(metadata, frame_count),
            )
            return inputs, proof

        inputs, transport_proof = process(11)
        counterfactual_inputs, counterfactual_proof = process(3)
        result["input_keys"] = sorted(str(key) for key in inputs.keys())
        image_values = inputs.get("pixel_values")
        video_values = inputs.get("pixel_values_videos")
        result["pixel_values_shape"] = list(image_values.shape) if image_values is not None else []
        result["pixel_values_videos_shape"] = list(video_values.shape) if video_values is not None else []
        result["transport_proof"] = transport_proof
        result["counterfactual_transport_proof"] = counterfactual_proof
        first_shapes = {
            key: tuple(value.get("shape", []))
            for key, value in transport_proof.get("pixel_tensors", {}).items()
        }
        second_shapes = {
            key: tuple(value.get("shape", []))
            for key, value in counterfactual_proof.get("pixel_tensors", {}).items()
        }
        result["counterfactual_shape_changed"] = first_shapes != second_shapes
        if image_values is not None:
            result["transport"] = "sampled_frame_images"
        elif video_values is not None:
            result["transport"] = "video_tokens"
        result["passed"] = bool(
            transport_proof.get("pixel_transport_confirmed")
            and counterfactual_proof.get("pixel_transport_confirmed")
            and transport_proof.get("processor_observed_sample_count") == 11
            and counterfactual_proof.get("processor_observed_sample_count") == 3
            and result["counterfactual_shape_changed"]
            and set(inputs) == set(counterfactual_inputs)
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _telemetry_capability_probe() -> dict:
    """Prove that passive telemetry can observe logits without replacing them."""
    result = {
        "passed": False,
        "generate_supports_logits_processor": False,
        "generate_supports_streamer": False,
        "pass_through_identity": False,
        "implementation_class_available": False,
        "implementation_pass_through_identity": False,
        "implementation": "transformers_capability",
        "error": "",
    }
    try:
        import torch
        from transformers import DiffusionGemmaForBlockDiffusion

        parameters = inspect.signature(DiffusionGemmaForBlockDiffusion.generate).parameters
        result["generate_supports_logits_processor"] = "logits_processor" in parameters
        result["generate_supports_streamer"] = "streamer" in parameters

        try:
            import grounding_telemetry

            implementation_class = getattr(
                grounding_telemetry,
                "PassiveTelemetryLogitsProcessor",
                None,
            ) or getattr(
                grounding_telemetry,
                "DiffusionGemmaTelemetryLogitsProcessor",
                None,
            )
            if implementation_class is not None:
                result["implementation_class_available"] = True
                result["implementation"] = (
                    f"grounding_telemetry.{implementation_class.__name__}"
                )
                builder = getattr(grounding_telemetry, "build_diffusiongemma_telemetry", None)
                if callable(builder):
                    _, implementation_processor, _ = builder(
                        initial_input_length=2,
                        max_denoising_steps=4,
                        canvas_size=2,
                    )
                    implementation_scores = torch.zeros((1, 2, 3), dtype=torch.float32)
                    implementation_returned = implementation_processor(
                        torch.zeros((1, 2), dtype=torch.long),
                        implementation_scores,
                        cur_step=1,
                    )
                    result["implementation_pass_through_identity"] = (
                        implementation_returned is implementation_scores
                    )
        except (ImportError, AttributeError):
            # Capability inspection remains useful before the optional host implementation
            # is present, and avoids coupling this proof script to node import side effects.
            pass

        class _IdentityProcessor:
            def __call__(self, input_ids, scores):
                del input_ids
                return scores

        scores = torch.zeros((1, 2, 3), dtype=torch.float32)
        returned = _IdentityProcessor()(torch.zeros((1, 2), dtype=torch.long), scores)
        result["pass_through_identity"] = returned is scores
        result["passed"] = bool(
            result["generate_supports_logits_processor"]
            and result["generate_supports_streamer"]
            and result["pass_through_identity"]
            and result["implementation_class_available"]
            and result["implementation_pass_through_identity"]
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _proof_status(result: dict) -> dict[str, bool]:
    bridge_smoke = result.get("nvfp4_bridge_smoke")
    video_smoke = result.get("short_video_processor_smoke")
    telemetry_probe = result.get("telemetry_probe")
    return {
        "processor": bool(result.get("processor_loads")),
        "video": bool(isinstance(video_smoke, dict) and video_smoke.get("passed")),
        "nvfp4": bool(
            result.get("path_kind") == "nvfp4_hf_repo"
            and result.get("comfy_nvfp4_bridge_supported")
            and isinstance(bridge_smoke, dict)
            and bridge_smoke.get("passed")
        ),
        "telemetry": bool(isinstance(telemetry_probe, dict) and telemetry_probe.get("passed")),
        "gpu": bool(result.get("torch_cuda_available")),
    }


def _normalize_requirements(groups: list[list[str]] | None) -> list[str]:
    requested: list[str] = []
    for group in groups or []:
        for name in group:
            expanded = REQUIRED_PROOF_NAMES if name == "all" else (name,)
            for item in expanded:
                if item not in requested:
                    requested.append(item)
    return requested


def _evaluate_required_proofs(result: dict, required: list[str]) -> dict:
    status = _proof_status(result)
    failed = [name for name in required if not status.get(name, False)]
    return {
        "available": status,
        "requested": list(required),
        "failed": failed,
        "passed": not failed,
    }


def probe(model_path: str) -> dict:
    result = {
        "python": sys.executable,
        "model_path": model_path,
        "model_path_exists": Path(model_path).exists(),
        **_model_path_info(model_path),
        "transformers_importable": importlib.util.find_spec("transformers") is not None,
        "diffusion_gemma_module": importlib.util.find_spec("transformers.models.diffusion_gemma") is not None,
        "vllm_importable": importlib.util.find_spec("vllm") is not None,
        "modelopt_importable": bool(
            importlib.util.find_spec("modelopt") or importlib.util.find_spec("nvidia_modelopt")
        ),
        "comfy_kitchen_importable": importlib.util.find_spec("comfy_kitchen") is not None,
        "fp_quant_importable": importlib.util.find_spec("fp_quant") is not None,
        "qutlass_importable": importlib.util.find_spec("qutlass") is not None,
        "tensorrt_llm_importable": importlib.util.find_spec("tensorrt_llm") is not None,
        "torch_cuda_available": False,
        "torch_cuda_capability": "",
        "transformers_version": "",
        "processor_class": "",
        "auto_processor": False,
        "diffusion_gemma_class": False,
        "processor_loads": False,
        "modelopt_nvfp4_supported": False,
        "comfy_nvfp4_bridge_supported": False,
        "nvfp4_bridge_smoke": {},
        "short_video_processor_smoke": {},
        "telemetry_probe": {},
        "notes": [],
        "errors": [],
    }
    try:
        import torch

        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["torch_cuda_capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability())
    except Exception as exc:
        result["errors"].append(f"Torch CUDA probe failed: {exc}")
    try:
        import transformers
        from transformers import AutoProcessor

        result["transformers_version"] = str(getattr(transformers, "__version__", ""))
        result["auto_processor"] = True
        if result["model_path_exists"]:
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            result["processor_class"] = type(processor).__name__
            result["processor_loads"] = True
            result["short_video_processor_smoke"] = _short_video_processor_smoke(model_path)
            if not result["short_video_processor_smoke"].get("passed"):
                result["errors"].append(
                    "Short sampled-video processor smoke failed; video pixels may fall back before model generation."
                )
    except Exception as exc:
        result["errors"].append(f"AutoProcessor probe failed: {exc}")

    try:
        from transformers import DiffusionGemmaForBlockDiffusion  # noqa: F401

        result["diffusion_gemma_class"] = True
    except Exception as exc:
        result["errors"].append(f"DiffusionGemmaForBlockDiffusion probe failed: {exc}")

    result["telemetry_probe"] = _telemetry_capability_probe()
    if not result["telemetry_probe"].get("passed"):
        error = result["telemetry_probe"].get("error") or "required generation hooks are unavailable"
        result["errors"].append(f"Passive telemetry capability probe failed: {error}")

    if result["path_kind"] == "nvfp4_hf_repo":
        modelopt_nvfp4 = result["quant_method"] == "modelopt"
        try:
            from nodes import RuntimeConfig, _runtime_status

            runtime_status = _runtime_status(
                RuntimeConfig(
                    model_path=model_path,
                    backend="transformers_inprocess",
                    dtype="auto",
                    quantization="modelopt_nvfp4",
                    local_files_only=True,
                    unload_policy="keep_loaded",
                    max_memory_gb=30.0,
                )
            )
            result["comfy_nvfp4_bridge_supported"] = bool(runtime_status.get("comfy_nvfp4_bridge_supported"))
            result["nvfp4_bridge_smoke"] = runtime_status.get("nvfp4_bridge_smoke", {})
            result["modelopt_nvfp4_supported"] = bool(runtime_status.get("modelopt_nvfp4_supported"))
        except Exception as exc:
            result["errors"].append(f"Comfy NVFP4 bridge probe failed: {exc}")

        if result["comfy_nvfp4_bridge_supported"]:
            result["notes"].append(
                "NVFP4 Hugging Face repo detected. Transformers can parse DiffusionGemma, and the local "
                "Comfy NVFP4 bridge smoke passed for ModelOpt packed expert weights."
            )
        else:
            result["modelopt_nvfp4_supported"] = bool(
                modelopt_nvfp4
                and result["modelopt_importable"]
                and result["tensorrt_llm_importable"]
                and result["has_modelopt_state"]
            )
            result["notes"].append(
                "NVFP4 Hugging Face repo detected. Transformers can parse the DiffusionGemma config and processor, "
                "but this ModelOpt NVFP4 checkpoint still requires a supported in-process ModelOpt NVFP4 loader."
            )
        if modelopt_nvfp4 and not result["modelopt_nvfp4_supported"]:
            result["errors"].append(
                "ModelOpt NVFP4 load is blocked: Transformers does not register quant_method=modelopt, "
                "modelopt_state.pth is absent, TensorRT-LLM is unavailable, or the Comfy NVFP4 bridge did not pass."
            )

    result["gate_1_passed"] = bool(
        result["model_path_exists"]
        and result["auto_processor"]
        and result["diffusion_gemma_class"]
        and result["processor_loads"]
        and not (
            result["path_kind"] == "nvfp4_hf_repo"
            and result["quant_method"] == "modelopt"
            and not result["modelopt_nvfp4_supported"]
        )
    )
    result["class_gate_passed"] = bool(
        result["model_path_exists"]
        and result["auto_processor"]
        and result["diffusion_gemma_class"]
        and result["processor_loads"]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DiffusionGemma runtime proof-gate probe.")
    parser.add_argument(
        "--model-path",
        default=str(Path.cwd() / "models" / "LLM" / "diffusiongemma-26B-A4B-it"),
        help="Local whole-repo DiffusionGemma path.",
    )
    parser.add_argument(
        "--require",
        action="append",
        nargs="+",
        choices=("all", *REQUIRED_PROOF_NAMES),
        default=[],
        metavar="PROOF",
        help=(
            "Exit nonzero when a requested proof fails. May be repeated; choices are "
            "all, processor, video, nvfp4, telemetry, and gpu. Without this option the "
            "probe remains report-only and exits zero."
        ),
    )
    args = parser.parse_args(argv)
    result = probe(args.model_path)
    required = _normalize_requirements(args.require)
    result["required_proofs"] = _evaluate_required_proofs(result, required)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if required and not result["required_proofs"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
