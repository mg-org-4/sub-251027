"""Wraps the ID-LoRA pipelines (one-stage and two-stage) for reusable ComfyUI execution."""

from __future__ import annotations

import gc
from dataclasses import replace as dc_replace
from pathlib import Path

import torch
import torchaudio

from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, SpatioTemporalScaleFactors
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.guidance import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.audio_vae import (
    AudioProcessor,
    decode_audio as vae_decode_audio,
)
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.quantization import QuantizationPolicy

from ltx_pipelines.utils import ModelLedger, cleanup_memory, euler_denoising_loop, encode_prompts, res2s_audio_video_denoising_loop
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import modality_from_latent_state, noise_video_state, noise_audio_state
from ltx_pipelines.utils.types import PipelineComponents


# ---------------------------------------------------------------------------
# Custom ComfyUI types
# ---------------------------------------------------------------------------

class IDLoraPipelineType:
    """Marker for the ID_LORA_PIPELINE custom type."""


class IDLoraConditioningType:
    """Marker for the ID_LORA_CONDITIONING custom type."""


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

RESOLUTION_DIVISOR = 32
MAX_LONG_SIDE = 512
MAX_PIXELS = 576 * 1024


def snap_to_divisor(value: int, divisor: int = RESOLUTION_DIVISOR) -> int:
    return max(int(round(value / divisor)) * divisor, divisor)


def compute_resolution_match_aspect(
    src_h: int, src_w: int,
    max_long: int = MAX_LONG_SIDE,
    max_pixels: int = MAX_PIXELS,
    divisor: int = RESOLUTION_DIVISOR,
) -> tuple[int, int]:
    scale = max_long / max(src_h, src_w)
    pixel_scale = (max_pixels / (src_h * src_w)) ** 0.5
    scale = min(scale, pixel_scale)
    return snap_to_divisor(int(round(src_h * scale)), divisor), snap_to_divisor(int(round(src_w * scale)), divisor)


# ---------------------------------------------------------------------------
# Shared base class for ID-LoRA pipelines
# ---------------------------------------------------------------------------

class _IDLoraBase:
    """Shared helpers used by both one-stage and two-stage pipelines."""

    dtype: torch.dtype
    device: torch.device
    _stg_scale: float
    _stg_blocks: list[int]
    _stg_mode: str
    _av_bimodal_cfg: bool
    _av_bimodal_scale: float
    _video_patchifier: VideoLatentPatchifier
    _audio_patchifier: AudioPatchifier
    _video_encoder: object
    _audio_encoder: object
    _audio_processor: object

    def _stg_config(self) -> BatchedPerturbationConfig:
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=self._stg_blocks)
        ]
        if self._stg_mode == "stg_av":
            perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=self._stg_blocks))
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)])

    def _av_bimodal_config(self) -> BatchedPerturbationConfig:
        return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=[
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ])])

    def _create_video_state(
        self,
        output_shape: VideoPixelShape,
        condition_image: torch.Tensor | None,
        noiser: GaussianNoiser,
        frame_rate: float,
    ) -> tuple[LatentState, VideoLatentTools, int]:
        video_tools = VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(output_shape),
            fps=frame_rate,
        )
        target_state = video_tools.create_initial_state(device=self.device, dtype=self.dtype)

        if condition_image is not None:
            target_state = self._apply_image_conditioning(target_state, condition_image, output_shape)

        video_state = noiser(latent_state=target_state, noise_scale=1.0)
        return video_state, video_tools, 0

    def _create_audio_state(
        self,
        output_shape: VideoPixelShape,
        reference_audio: torch.Tensor | None,
        reference_audio_sample_rate: int,
        noiser: GaussianNoiser,
    ) -> tuple[LatentState, AudioLatentTools, int]:
        duration = output_shape.frames / output_shape.fps
        audio_tools = AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=duration),
        )
        target_state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)
        ref_seq_len = 0

        if reference_audio is not None:
            ref_latent, ref_pos = self._encode_audio(reference_audio, reference_audio_sample_rate)
            ref_seq_len = ref_latent.shape[1]

            hop_length = 160
            downsample = 4
            sr = 16000
            time_per_latent = hop_length * downsample / sr
            aud_dur = ref_pos[:, :, -1, 1].max().item()
            ref_pos = ref_pos - aud_dur - time_per_latent

            ref_mask = torch.zeros(1, ref_seq_len, 1, device=self.device, dtype=torch.float32)
            combined = LatentState(
                latent=torch.cat([ref_latent, target_state.latent], dim=1),
                denoise_mask=torch.cat([ref_mask, target_state.denoise_mask], dim=1),
                positions=torch.cat([ref_pos, target_state.positions], dim=2),
                clean_latent=torch.cat([ref_latent, target_state.clean_latent], dim=1),
            )
            audio_state = noiser(latent_state=combined, noise_scale=1.0)
        else:
            audio_state = noiser(latent_state=target_state, noise_scale=1.0)

        return audio_state, audio_tools, ref_seq_len

    @staticmethod
    def _center_crop_resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
        import torch.nn.functional as F
        src_h, src_w = image.shape[1], image.shape[2]
        img = image.unsqueeze(0)
        if src_h != height or src_w != width:
            ar, tar = src_w / src_h, width / height
            rh, rw = (height, int(height * ar)) if ar > tar else (int(width / ar), width)
            img = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
            h0, w0 = (rh - height) // 2, (rw - width) // 2
            img = img[:, :, h0:h0 + height, w0:w0 + width]
        return img

    def _apply_image_conditioning(
        self, video_state: LatentState, image: torch.Tensor, output_shape: VideoPixelShape
    ) -> LatentState:
        image = self._center_crop_resize(image, output_shape.height, output_shape.width)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(2).to(device=self.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = self._video_encoder(image)
        patchified = self._video_patchifier.patchify(encoded)
        n = patchified.shape[1]

        new_latent = video_state.latent.clone()
        new_latent[:, :n] = patchified.to(new_latent.dtype)
        new_clean = video_state.clean_latent.clone()
        new_clean[:, :n] = patchified.to(new_clean.dtype)
        new_mask = video_state.denoise_mask.clone()
        new_mask[:, :n] = 0.0
        return LatentState(
            latent=new_latent, denoise_mask=new_mask,
            positions=video_state.positions, clean_latent=new_clean,
        )

    def _encode_audio(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        mel = self._audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sample_rate))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_raw = self._audio_encoder(mel.to(torch.float32))
        latent_raw = latent_raw.to(self.dtype)
        B, C, T, Fq = latent_raw.shape
        latent = self._audio_patchifier.patchify(latent_raw)
        positions = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(batch=B, channels=C, frames=T, mel_bins=Fq),
            device=self.device,
        ).to(self.dtype)
        return latent, positions


# ---------------------------------------------------------------------------
# One-stage ID-LoRA pipeline (adapted for reusable ComfyUI execution)
# ---------------------------------------------------------------------------

class IDLoraOneStagePipeline(_IDLoraBase):
    """
    One-stage pipeline for audio identity transfer with ID-LoRA.

    Key difference from the script version: the video encoder is stashed to CPU
    and reloaded instead of being deleted, so the pipeline is reusable across
    multiple generations.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device,
        quantize: bool = False,
        fp8: bool = False,
        stg_scale: float = 1.0,
        stg_blocks: list[int] | None = None,
        stg_mode: str = "stg_av",
        identity_guidance: bool = True,
        identity_guidance_scale: float = 3.0,
        av_bimodal_cfg: bool = True,
        av_bimodal_scale: float = 3.0,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self._checkpoint_path = checkpoint_path
        self._loras = loras
        self._quantize = quantize
        self._fp8 = fp8

        self._stg_scale = stg_scale
        self._stg_blocks = stg_blocks if stg_blocks is not None else [29]
        self._stg_mode = stg_mode
        self._identity_guidance = identity_guidance
        self._identity_guidance_scale = identity_guidance_scale
        self._av_bimodal_cfg = av_bimodal_cfg
        self._av_bimodal_scale = av_bimodal_scale

        quantization = QuantizationPolicy.fp8_cast() if fp8 else None
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=tuple(loras),
            quantization=quantization,
        )

        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)
        self._transformer = None
        self._video_encoder = None
        self._video_encoder_on_cpu = None

    def load_models(self):
        print("[ID-LoRA] Loading video encoder...")
        self._video_encoder = self.model_ledger.video_encoder()

        print("[ID-LoRA] Loading transformer...")
        if self._quantize:
            from ltx_trainer.quantization import quantize_model
            x0_model = self.model_ledger.transformer()
            quantize_model(x0_model.velocity_model, "int8-quanto")
            cleanup_memory()
            self._transformer = x0_model
        else:
            self._transformer = self.model_ledger.transformer()

        print("[ID-LoRA] Loading audio encoder...")
        self._audio_encoder = self.model_ledger.audio_encoder()

        print("[ID-LoRA] Creating audio processor...")
        self._audio_processor = AudioProcessor(
            target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
        ).to(self.device)

    def _ensure_video_encoder(self):
        """Restore video encoder from CPU stash if needed."""
        if self._video_encoder is None and self._video_encoder_on_cpu is not None:
            self._video_encoder = self._video_encoder_on_cpu.to(self.device)
            self._video_encoder_on_cpu = None
        elif self._video_encoder is None:
            print("[ID-LoRA] Re-loading video encoder...")
            self._video_encoder = self.model_ledger.video_encoder()

    def _stash_video_encoder(self):
        """Move video encoder to CPU instead of deleting it."""
        if self._video_encoder is not None:
            self._video_encoder_on_cpu = self._video_encoder.cpu()
            self._video_encoder = None
            cleanup_memory()

    @torch.inference_mode()
    def __call__(
        self,
        v_context_p: torch.Tensor,
        a_context_p: torch.Tensor,
        v_context_n: torch.Tensor,
        a_context_n: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        reference_audio: torch.Tensor | None = None,
        reference_audio_sample_rate: int = 16000,
        condition_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_video_encoder()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        video_cfg = CFGGuider(video_guidance_scale)
        audio_cfg = CFGGuider(audio_guidance_scale)
        stg_guider = STGGuider(self._stg_scale)
        av_bimodal_guider = CFGGuider(self._av_bimodal_scale if self._av_bimodal_cfg else 0.0)

        stg_pcfg = self._stg_config() if stg_guider.enabled() else None
        av_pcfg = self._av_bimodal_config() if av_bimodal_guider.enabled() else None

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )

        video_state, video_tools, ref_vid_len = self._create_video_state(
            output_shape=output_shape,
            condition_image=condition_image,
            noiser=noiser,
            frame_rate=frame_rate,
        )
        audio_state, audio_tools, ref_aud_len = self._create_audio_state(
            output_shape=output_shape,
            reference_audio=reference_audio,
            reference_audio_sample_rate=reference_audio_sample_rate,
            noiser=noiser,
        )

        # Stash video encoder to CPU (saves VRAM during denoising)
        self._stash_video_encoder()

        total_steps = len(sigmas) - 1

        def denoising_func(video_state, audio_state, sigmas, step_idx):
            sigma = sigmas[step_idx]
            print(f"  Step {step_idx + 1}/{total_steps} (sigma={sigma.item():.4f})", flush=True)

            pv = modality_from_latent_state(video_state, v_context_p, sigma)
            pa = modality_from_latent_state(audio_state, a_context_p, sigma)
            dv_pos, da_pos = self._transformer(video=pv, audio=pa, perturbations=None)

            delta_v = torch.zeros_like(dv_pos)
            delta_a = torch.zeros_like(da_pos) if da_pos is not None else None

            if video_cfg.enabled() or audio_cfg.enabled():
                nv = modality_from_latent_state(video_state, v_context_n, sigma)
                na = modality_from_latent_state(audio_state, a_context_n, sigma)
                dv_neg, da_neg = self._transformer(video=nv, audio=na, perturbations=None)
                delta_v = delta_v + video_cfg.delta(dv_pos, dv_neg)
                if delta_a is not None:
                    delta_a = delta_a + audio_cfg.delta(da_pos, da_neg)

            if self._identity_guidance and self._identity_guidance_scale > 0 and ref_aud_len > 0:
                tgt_aud = LatentState(
                    latent=audio_state.latent[:, ref_aud_len:],
                    denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                    positions=audio_state.positions[:, :, ref_aud_len:],
                    clean_latent=audio_state.clean_latent[:, ref_aud_len:],
                )
                nrv = modality_from_latent_state(video_state, v_context_p, sigma)
                nra = modality_from_latent_state(tgt_aud, a_context_p, sigma)
                _, da_noref = self._transformer(video=nrv, audio=nra, perturbations=None)
                if delta_a is not None and da_noref is not None:
                    id_delta = self._identity_guidance_scale * (da_pos[:, ref_aud_len:] - da_noref)
                    full_id = torch.zeros_like(delta_a)
                    full_id[:, ref_aud_len:] = id_delta
                    delta_a = delta_a + full_id

            if stg_guider.enabled() and stg_pcfg is not None:
                pv_s, pa_s = self._transformer(video=pv, audio=pa, perturbations=stg_pcfg)
                delta_v = delta_v + stg_guider.delta(dv_pos, pv_s)
                if delta_a is not None and pa_s is not None:
                    delta_a = delta_a + stg_guider.delta(da_pos, pa_s)

            if av_bimodal_guider.enabled() and av_pcfg is not None:
                pv_b, pa_b = self._transformer(video=pv, audio=pa, perturbations=av_pcfg)
                delta_v = delta_v + av_bimodal_guider.delta(dv_pos, pv_b)
                if delta_a is not None and pa_b is not None:
                    delta_a = delta_a + av_bimodal_guider.delta(da_pos, pa_b)

            out_v = dv_pos + delta_v
            out_a = (da_pos + delta_a) if (da_pos is not None and delta_a is not None) else da_pos
            return out_v, out_a

        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=denoising_func,
        )

        if ref_vid_len > 0:
            video_state = LatentState(
                latent=video_state.latent[:, ref_vid_len:],
                denoise_mask=video_state.denoise_mask[:, ref_vid_len:],
                positions=video_state.positions[:, :, ref_vid_len:],
                clean_latent=video_state.clean_latent[:, ref_vid_len:],
            )
        if ref_aud_len > 0:
            audio_state = LatentState(
                latent=audio_state.latent[:, ref_aud_len:],
                denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                positions=audio_state.positions[:, :, ref_aud_len:],
                clean_latent=audio_state.clean_latent[:, ref_aud_len:],
            )

        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)

        torch.cuda.synchronize()
        self._transformer.to("cpu")
        torch.cuda.synchronize()
        cleanup_memory()

        print("[ID-LoRA] Loading video decoder...")
        video_decoder = self.model_ledger.video_decoder()
        video_latent = video_state.latent.to(dtype=torch.bfloat16)
        decoded_video = video_decoder(video_latent)
        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        video_tensor = decoded_video[0].float().cpu()
        del video_latent, decoded_video, video_decoder

        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        print("[ID-LoRA] Loading audio decoder...")
        audio_decoder = self.model_ledger.audio_decoder()
        vocoder = self.model_ledger.vocoder()
        decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
        audio_output = decoded_audio.waveform.cpu()
        self._vocoder_sr = decoded_audio.sampling_rate

        del audio_decoder, vocoder
        gc.collect()
        torch.cuda.empty_cache()

        self._transformer.to(self.device)
        torch.cuda.synchronize()
        cleanup_memory()

        return video_tensor, audio_output


# ---------------------------------------------------------------------------
# Two-stage ID-LoRA pipeline
# ---------------------------------------------------------------------------

class IDLoraTwoStagePipeline(_IDLoraBase):
    """
    Two-stage pipeline for audio identity transfer with ID-LoRA.

    Stage 1: Generate at height x width with ID-LoRA + full guidance.
    Stage 2: 2x spatial upsample, refine with distilled LoRA only.
    Audio from stage 1 is frozen in stage 2.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        upsampler_path: str,
        distilled_lora_path: str | None,
        ic_loras: list[LoraPathStrengthAndSDOps],
        device: torch.device,
        quantize: bool = False,
        fp8: bool = False,
        stg_scale: float = 1.0,
        stg_blocks: list[int] | None = None,
        stg_mode: str = "stg_av",
        identity_guidance: bool = True,
        identity_guidance_scale: float = 3.0,
        av_bimodal_cfg: bool = True,
        av_bimodal_scale: float = 3.0,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self._checkpoint_path = checkpoint_path
        self._ic_loras = ic_loras
        self._quantize = quantize
        self._fp8 = fp8

        self._stg_scale = stg_scale
        self._stg_blocks = stg_blocks if stg_blocks is not None else [29]
        self._stg_mode = stg_mode
        self._identity_guidance = identity_guidance
        self._identity_guidance_scale = identity_guidance_scale
        self._av_bimodal_cfg = av_bimodal_cfg
        self._av_bimodal_scale = av_bimodal_scale

        quantization = QuantizationPolicy.fp8_cast() if fp8 else None
        self._stage_1_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=upsampler_path,
            loras=tuple(ic_loras),
            quantization=quantization,
        )

        distilled_loras: list[LoraPathStrengthAndSDOps] = []
        if distilled_lora_path and Path(distilled_lora_path).exists():
            distilled_loras = [LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )]
            print(f"[ID-LoRA] Stage-2 distilled LoRA: {distilled_lora_path}")
        else:
            print(f"[ID-LoRA] WARNING: distilled LoRA not found at {distilled_lora_path!r}, stage 2 runs without it")

        self._stage_2_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upsampler_path,
            loras=tuple(distilled_loras),
            quantization=quantization,
        )

        # Expose stage-1 ledger as model_ledger for prompt encoder compatibility
        self.model_ledger = self._stage_1_ledger

        self._pipeline_components = PipelineComponents(dtype=self.dtype, device=device)
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

        self._video_encoder = None
        self._stage_1_transformer = None
        self._audio_encoder = None
        self._audio_processor = None

    def load_stage_1_models(self):
        print("[ID-LoRA] Loading video encoder...")
        self._video_encoder = self._stage_1_ledger.video_encoder()

        print("[ID-LoRA] Loading stage-1 transformer (ID-LoRA)...")
        if self._quantize:
            from ltx_trainer.quantization import quantize_model
            x0_model = self._stage_1_ledger.transformer()
            quantize_model(x0_model.velocity_model, "int8-quanto")
            cleanup_memory()
            self._stage_1_transformer = x0_model
        else:
            self._stage_1_transformer = self._stage_1_ledger.transformer()

        print("[ID-LoRA] Loading audio encoder...")
        self._audio_encoder = self._stage_1_ledger.audio_encoder()

        print("[ID-LoRA] Creating audio processor...")
        self._audio_processor = AudioProcessor(
            target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
        ).to(self.device)

    def _free_stage_1_models(self):
        for attr in ("_stage_1_transformer", "_audio_encoder", "_audio_processor"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.cpu()
                except Exception:
                    pass
                del obj
                setattr(self, attr, None)
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_memory()
        print(f"[ID-LoRA] Stage-1 models freed.  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def load_stage_2_models(self):
        print("[ID-LoRA] Loading stage-2 transformer (distilled LoRA)...")
        if self._quantize:
            from ltx_trainer.quantization import quantize_model
            x0_model = self._stage_2_ledger.transformer()
            quantize_model(x0_model.velocity_model, "int8-quanto")
            cleanup_memory()
            self._stage_2_transformer = x0_model
        else:
            self._stage_2_transformer = self._stage_2_ledger.transformer()

        print("[ID-LoRA] Loading video/audio decoders + vocoder...")
        self._video_decoder = self._stage_2_ledger.video_decoder()
        self._audio_decoder = self._stage_2_ledger.audio_decoder()
        self._vocoder = self._stage_2_ledger.vocoder()

    @torch.inference_mode()
    def __call__(
        self,
        v_context_p: torch.Tensor,
        a_context_p: torch.Tensor,
        v_context_n: torch.Tensor,
        a_context_n: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        hq_mode: bool = False,
        reference_audio: torch.Tensor | None = None,
        reference_audio_sample_rate: int = 16000,
        condition_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = Res2sDiffusionStep() if hq_mode else EulerDiffusionStep()
        denoising_loop = res2s_audio_video_denoising_loop if hq_mode else euler_denoising_loop

        video_cfg = CFGGuider(video_guidance_scale)
        audio_cfg = CFGGuider(audio_guidance_scale)
        stg_guider = STGGuider(self._stg_scale)
        av_bimodal_guider = CFGGuider(self._av_bimodal_scale if self._av_bimodal_cfg else 0.0)

        stg_pcfg = self._stg_config() if stg_guider.enabled() else None
        av_pcfg = self._av_bimodal_config() if av_bimodal_guider.enabled() else None

        # ---- Stage 1: generate at target resolution with full guidance ----
        s1_h, s1_w = height, width
        s1_shape = VideoPixelShape(batch=1, frames=num_frames, width=s1_w, height=s1_h, fps=frame_rate)
        sigmas_s1 = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        video_state, _, ref_vid_len = self._create_video_state(
            output_shape=s1_shape,
            condition_image=condition_image,
            noiser=noiser,
            frame_rate=frame_rate,
        )
        audio_state, _, ref_aud_len = self._create_audio_state(
            output_shape=s1_shape,
            reference_audio=reference_audio,
            reference_audio_sample_rate=reference_audio_sample_rate,
            noiser=noiser,
        )

        total_s1 = len(sigmas_s1) - 1

        def stage_1_denoise(video_state, audio_state, sigmas, step_index):
            sigma = sigmas[step_index]
            print(f"  S1 {step_index+1}/{total_s1} sigma={sigma.item():.4f}", flush=True)

            pv = modality_from_latent_state(video_state, v_context_p, sigma)
            pa = modality_from_latent_state(audio_state, a_context_p, sigma)
            dv_pos, da_pos = self._stage_1_transformer(video=pv, audio=pa, perturbations=None)

            dv_delta = torch.zeros_like(dv_pos)
            da_delta = torch.zeros_like(da_pos) if da_pos is not None else None

            if video_cfg.enabled() or audio_cfg.enabled():
                nv = modality_from_latent_state(video_state, v_context_n, sigma)
                na = modality_from_latent_state(audio_state, a_context_n, sigma)
                dv_neg, da_neg = self._stage_1_transformer(video=nv, audio=na, perturbations=None)
                dv_delta = dv_delta + video_cfg.delta(dv_pos, dv_neg)
                if da_delta is not None:
                    da_delta = da_delta + audio_cfg.delta(da_pos, da_neg)

            if self._identity_guidance and self._identity_guidance_scale > 0 and ref_aud_len > 0:
                tgt_aud = LatentState(
                    latent=audio_state.latent[:, ref_aud_len:],
                    denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                    positions=audio_state.positions[:, :, ref_aud_len:],
                    clean_latent=audio_state.clean_latent[:, ref_aud_len:],
                )
                nrv = modality_from_latent_state(video_state, v_context_p, sigma)
                nra = modality_from_latent_state(tgt_aud, a_context_p, sigma)
                _, da_noref = self._stage_1_transformer(video=nrv, audio=nra, perturbations=None)
                if da_delta is not None and da_noref is not None:
                    id_delta = self._identity_guidance_scale * (da_pos[:, ref_aud_len:] - da_noref)
                    full_id_delta = torch.zeros_like(da_delta)
                    full_id_delta[:, ref_aud_len:] = id_delta
                    da_delta = da_delta + full_id_delta

            if stg_guider.enabled() and stg_pcfg is not None:
                pv_s, pa_s = self._stage_1_transformer(video=pv, audio=pa, perturbations=stg_pcfg)
                dv_delta = dv_delta + stg_guider.delta(dv_pos, pv_s)
                if da_delta is not None and pa_s is not None:
                    da_delta = da_delta + stg_guider.delta(da_pos, pa_s)

            if av_bimodal_guider.enabled() and av_pcfg is not None:
                pv_b, pa_b = self._stage_1_transformer(video=pv, audio=pa, perturbations=av_pcfg)
                dv_delta = dv_delta + av_bimodal_guider.delta(dv_pos, pv_b)
                if da_delta is not None and pa_b is not None:
                    da_delta = da_delta + av_bimodal_guider.delta(da_pos, pa_b)

            dv_out = dv_pos + dv_delta
            da_out = (da_pos + da_delta) if (da_pos is not None and da_delta is not None) else da_pos
            return dv_out, da_out

        sampler_name = "res2s" if hq_mode else "euler"
        print(f"[ID-LoRA] Stage 1: generating at {s1_h}x{s1_w} ({num_inference_steps} steps, {sampler_name})...")
        video_state, audio_state = denoising_loop(
            sigmas=sigmas_s1,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=stage_1_denoise,
        )

        # Strip reference tokens
        if ref_vid_len > 0:
            video_state = LatentState(
                latent=video_state.latent[:, ref_vid_len:],
                denoise_mask=video_state.denoise_mask[:, ref_vid_len:],
                positions=video_state.positions[:, :, ref_vid_len:],
                clean_latent=video_state.clean_latent[:, ref_vid_len:],
            )
        if ref_aud_len > 0:
            audio_state = LatentState(
                latent=audio_state.latent[:, ref_aud_len:],
                denoise_mask=audio_state.denoise_mask[:, ref_aud_len:],
                positions=audio_state.positions[:, :, ref_aud_len:],
                clean_latent=audio_state.clean_latent[:, ref_aud_len:],
            )

        s1_vid_tools = VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(s1_shape),
            fps=frame_rate,
        )
        s1_aud_tools = AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate),
        )
        video_state = s1_vid_tools.clear_conditioning(video_state)
        video_state = s1_vid_tools.unpatchify(video_state)
        audio_state = s1_aud_tools.clear_conditioning(audio_state)
        audio_state = s1_aud_tools.unpatchify(audio_state)

        s1_video_latent = video_state.latent
        s1_audio_latent = audio_state.latent

        # ---- Transition: free stage-1 models, upsample ----
        self._free_stage_1_models()

        s2_h, s2_w = height * 2, width * 2
        s2_shape = VideoPixelShape(batch=1, frames=num_frames, width=s2_w, height=s2_h, fps=frame_rate)

        print("[ID-LoRA] Upsampling video latent 2x...")
        _upsampler = self._stage_2_ledger.spatial_upsampler()
        upscaled_latent = upsample_video(
            latent=s1_video_latent[:1],
            video_encoder=self._video_encoder,
            upsampler=_upsampler,
        )
        print(f"  {s1_video_latent.shape} -> {upscaled_latent.shape}")

        # Re-encode first frame at 2x resolution for stage-2 conditioning
        s2_conditionings = []
        if condition_image is not None:
            s2_image = self._center_crop_resize(condition_image, s2_h, s2_w)
            s2_image = s2_image * 2.0 - 1.0
            s2_image = s2_image.unsqueeze(2).to(device=self.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                s2_encoded = self._video_encoder(s2_image)
            s2_conditionings.append(
                VideoConditionByLatentIndex(latent=s2_encoded, strength=1.0, latent_idx=0)
            )

        # Free video encoder and upsampler
        self._video_encoder.cpu()
        _upsampler.cpu()
        del self._video_encoder, _upsampler
        self._video_encoder = None
        cleanup_memory()

        # ---- Stage 2: refine at 2x resolution ----
        sigmas_s2 = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)

        self.load_stage_2_models()

        def stage_2_denoise(video_state, audio_state, sigmas, step_index):
            sigma = sigmas[step_index]
            v_mod = modality_from_latent_state(video_state, v_context_p, sigma)
            a_mod = modality_from_latent_state(audio_state, a_context_p, sigma)
            dv, da = self._stage_2_transformer(video=v_mod, audio=a_mod, perturbations=None)
            return dv, da

        total_s2 = len(sigmas_s2) - 1
        print(f"[ID-LoRA] Stage 2: refining at {s2_h}x{s2_w} ({total_s2} steps, {sampler_name})...")

        video_state, video_tools = noise_video_state(
            output_shape=s2_shape, noiser=noiser, conditionings=s2_conditionings,
            components=self._pipeline_components, dtype=self.dtype, device=self.device,
            noise_scale=sigmas_s2[0].item(), initial_latent=upscaled_latent,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=s2_shape, noiser=noiser, conditionings=[],
            components=self._pipeline_components, dtype=self.dtype, device=self.device,
            noise_scale=0.0, initial_latent=s1_audio_latent,
        )
        audio_state = dc_replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

        stepper_s2 = Res2sDiffusionStep() if hq_mode else EulerDiffusionStep()
        video_state, audio_state = denoising_loop(
            sigmas=sigmas_s2, video_state=video_state, audio_state=audio_state,
            stepper=stepper_s2, denoise_fn=stage_2_denoise,
        )
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        # ---- Decode ----
        video_latent = video_state.latent.to(torch.bfloat16)
        tiling_config = TilingConfig.default()
        chunks = list(self._video_decoder.tiled_decode(video_latent[:1], tiling_config))
        decoded_video = torch.cat(chunks, dim=2)
        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        video_tensor = decoded_video[0].float().cpu()

        decoded_audio = vae_decode_audio(audio_state.latent, self._audio_decoder, self._vocoder)
        audio_output = decoded_audio.waveform.cpu()
        self._vocoder_sr = decoded_audio.sampling_rate

        # Free stage-2 models
        for attr in ("_stage_2_transformer", "_video_decoder", "_audio_decoder", "_vocoder"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.cpu()
                except Exception:
                    pass
                del obj
                setattr(self, attr, None)
        cleanup_memory()

        return video_tensor, audio_output
