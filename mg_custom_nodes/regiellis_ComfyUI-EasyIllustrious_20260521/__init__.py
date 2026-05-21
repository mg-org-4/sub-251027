import os

# Avoid legacy naming in this file by binding via indirect attribute access
from . import src as _src
from .src.nodes import (
    pony_tokens as _pony_tokens,
    poses as _poses,
    ksampler as _ksampler,
    multi_pass_sampler as _mps,
    scheduler as _scheduler,
    clip_encoder as _clip,
    latent as _latent,
)
from .src.nodes.vae import IllustriousVAEDecode, IllustriousVAEEncode
from .src.nodes.illustrious_regional_conditioning import (
    IllustriousRegionalConditioning,
    IllustriousMakeRegion,
    IllustriousEmptyRegions,
    IllustriousAppendRegion,
)
from .src.nodes.tipo_optimizer import TIPOPromptOptimizer
from .src.nodes.attention_couple import IllustriousAttentionCouple
from .src.core import anime_scene_system as _anime
from .src.nodes.color_suite import IllustriousColorSuite
from .src.nodes.auto_outpaint import IllustriousAutoOutpaint
from .src.nodes.emotions import IllustriousEmotions
from .src.nodes.smart_prompt_generator import IllustriousSmartSceneGenerator
# Import smart cache server module to register routes (side effects only)
from .src.nodes import smart_cache_server as _ill_smart_cache_routes  # noqa: F401

# Bind Illustrious-facing classes (prefer Illustrious names; fallback to any legacy names that may linger)
IllustriousMasterModel = getattr(_src, "IllustriousMasterModel")
IllustriousPrompt = getattr(_src, "IllustriousPrompt")
IllustriousCharacters = getattr(_src, "IllustriousCharacters")
IllustriousArtists = getattr(_src, "IllustriousArtists")
IllustriousE621Characters = getattr(_src, "IllustriousE621Characters")
IllustriousE621Artists = getattr(_src, "IllustriousE621Artists")
IllustriousHairstyles = getattr(_src, "IllustriousHairstyles")
IllustriousClothing = getattr(_src, "IllustriousClothing")

IllustriousPonyTokens = getattr(_pony_tokens, "IllustriousPony")
IllustriousPoses = getattr(_poses, "IllustriousPoses")

from .src.nodes.ksampler import IllustriousKSamplerPro, IllustriousKSamplerPresets

IllustriousMultiPassSampler = getattr(_mps, "IllustriousMultiPassSampler")
IllustriousTriplePassSampler = getattr(_mps, "IllustriousTriplePassSampler")
IllustriousSchedulerNode = getattr(_scheduler, "IllustriousSchedulerNode")
from .src.nodes.clip_encoder import (
    IllustriousCLIPTextEncoder,
    IllustriousNegativeCLIPEncoder,
)

IllustriousEmptyLatentImage = getattr(_latent, "IllustriousEmptyLatentImage")
IllustriousLatentUpscale = getattr(_latent, "IllustriousLatentUpscale")

IllustriousScenesPlusEngine = getattr(_anime, "ScenesPlus")

# Import web routes to register API endpoints
from .src import web_routes

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

NODE_CLASS_MAPPINGS = {
    # Illustrious prompt/model suite
    "IllustriousMasterModel": IllustriousMasterModel,
    "IllustriousPrompt": IllustriousPrompt,
    "IllustriousCharacters": IllustriousCharacters,
    "IllustriousArtists": IllustriousArtists,
    "IllustriousE621Characters": IllustriousE621Characters,
    "IllustriousE621Artists": IllustriousE621Artists,
    "IllustriousHairstyles": IllustriousHairstyles,
    "IllustriousClothing": IllustriousClothing,
    "IllustriousPonyTokens": IllustriousPonyTokens,
    "IllustriousPoses": IllustriousPoses,
    "IllustriousEmotions": IllustriousEmotions,
    # Illustrious samplers/schedulers
    "IllustriousKSamplerPro": IllustriousKSamplerPro,
    "IllustriousKSamplerPresets": IllustriousKSamplerPresets,
    "IllustriousMultiPassSampler": IllustriousMultiPassSampler,
    "IllustriousTriplePassSampler": IllustriousTriplePassSampler,
    "IllustriousScheduler": IllustriousSchedulerNode,
    # Illustrious encoders/latents
    "IllustriousCLIPTextEncoder": IllustriousCLIPTextEncoder,
    "IllustriousNegativeCLIPEncoder": IllustriousNegativeCLIPEncoder,
    "IllustriousEmptyLatentImage": IllustriousEmptyLatentImage,
    "IllustriousLatentUpscale": IllustriousLatentUpscale,
    # Illustrious utilities/features
    "IllustriousColorSuite": IllustriousColorSuite,
    "IllustriousVAEDecode": IllustriousVAEDecode,
    "IllustriousVAEEncode": IllustriousVAEEncode,
    "IllustriousRegionalConditioning": IllustriousRegionalConditioning,
    "IllustriousMakeRegion": IllustriousMakeRegion,
    "IllustriousEmptyRegions": IllustriousEmptyRegions,
    "IllustriousAppendRegion": IllustriousAppendRegion,
    # (image comparer removed)
    "TIPOPromptOptimizer": TIPOPromptOptimizer,
    "IllustriousScenesPlus": IllustriousScenesPlusEngine,
    "IllustriousAttentionCouple": IllustriousAttentionCouple,
    "IllustriousAutoOutpaint": IllustriousAutoOutpaint,
    "IllustriousSmartSceneGenerator": IllustriousSmartSceneGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Prompt/model
    "IllustriousMasterModel": "◆ Base Model",
    "IllustriousPrompt": "∇ Prompt",
    "IllustriousCharacters": "◊ Characters",
    "IllustriousArtists": "◈ Artists",
    "IllustriousE621Characters": "◉ E621 Characters",
    "IllustriousE621Artists": "◎ E621 Artists",
    "IllustriousHairstyles": "◐ Hairstyles",
    "IllustriousClothing": "◑ Clothing",
    "IllustriousPonyTokens": "◒ Pony Tokens",
    "IllustriousPoses": "◓ Poses",
    "IllustriousEmotions": "◔ Emotions/Expressions",
    # Samplers/schedulers
    "IllustriousKSamplerPro": "⚡ KSampler (Illustrious)",
    "IllustriousKSamplerPresets": "⚙ KSampler Presets",
    "IllustriousMultiPassSampler": "⟲ Multi-Pass Sampler",
    "IllustriousTriplePassSampler": "⟳ Triple-Pass Sampler",
    "IllustriousScheduler": "⏱ Custom Scheduler",
    # Encoders/latents
    "IllustriousCLIPTextEncoder": "⌘ CLIP Text Encode (Illustrious)",
    "IllustriousNegativeCLIPEncoder": "⌫ CLIP Negative Encode (Illustrious)",
    "IllustriousEmptyLatentImage": "▢ Empty Latent Image",
    "IllustriousLatentUpscale": "▲ Latent Upscale",
    # Utilities/features
    "IllustriousColorSuite": "◈ Color Suite (Illustrious)",
    "IllustriousVAEDecode": "◀ VAE Decode (Illustrious)",
    "IllustriousVAEEncode": "▶ VAE Encode (Illustrious)",
    "IllustriousRegionalConditioning": "⌦ Regional Conditioning (Illustrious)",
    "IllustriousMakeRegion": "＋ Make Region (Illustrious)",
    "IllustriousEmptyRegions": "□ Empty Regions (Illustrious)",
    "IllustriousAppendRegion": "→ Append Region (Illustrious)",
    "IllustriousSmartSceneGenerator": "✦ Smart Scene Generator",
    # (image comparer removed)
    "TIPOPromptOptimizer": "◆ TIPO Prompt Optimizer",
    "IllustriousScenesPlus": "◊ Scenes",
    "IllustriousAttentionCouple": "◈ Attention Couple (Illustrious)",
    "IllustriousAutoOutpaint": "▭ Auto Outpaint (Illustrious)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
