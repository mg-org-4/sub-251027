import sys
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# COMFYUI IMPORTS
import comfy.sd
import folder_paths
import torch

from .artists import ARTISTS
from .characters import CHARACTERS
from .e621_artists import E621_ARTISTS
from .e621_characters import E621_CHARACTERS
from .core.anime_scene_system import ScenesPlus as IllustriousScenesPlus

# Import multi-pass samplers
from .nodes.multi_pass_sampler import (
    IllustriousMultiPassSampler,
    IllustriousTriplePassSampler,
)

# Search-only approach - no external imports needed

"""
Illustrious Prompt Suite and resources.
"""
# Global resolutions dictionary - primary resolution selector
RESOLUTIONS: Dict[str, str] = {
    "Portrait | SDXL Well Known (5:12) - 640x1536": "640x1536",
    "Portrait | SDXL Well Known (4:7) - 768x1344": "768x1344",
    "Portrait | SDXL Well Known (13:19) - 832x1216": "832x1216",
    "Portrait | SDXL Well Known (7:9) - 896x1152": "896x1152",
    "Portrait | Character Sheet (2:3) - 832x1248": "832x1248",
    "Portrait | VN CG (3:4) - 880x1176": "880x1176",
    "Portrait | Manga Cover (4:5) - 912x1144": "912x1144",
    "Mobile | Vertical Story (9:16) - 768x1360": "768x1360",
    "Square | Model Preview (1:1) - 1024x1024": "1024x1024",
    "Landscape | SDXL Well Known (12:5) - 1536x640": "1536x640",
    "Landscape | SDXL Well Known (7:4) - 1344x768": "1344x768",
    "Landscape | SDXL Well Known (19:13) - 1216x832": "1216x832",
    "Landscape | SDXL Well Known (9:7) - 1152x896": "1152x896",
    "Landscape | Character Sheet (2:3) - 832x1248": "832x1248",
    "Retro Anime TV (4:3) - 1176x888": "1176x888",
    "Key Visual Poster (1.43:1) - 1224x856": "1224x856",
    "Wide Panel (1.66:1) - 1312x792": "1312x792",
    "Storyboard Wide (16:9) - 1360x768": "1360x768",
    "Anime Movie Wide (1.85:1) - 1392x752": "1392x752",
    "Ultra Wide Banner (2.35:1) - 1568x664": "1568x664",
    "Ultra Wide Banner+ (2.39:1) - 1576x656": "1576x656",
    "Retro TV Special (4:3) - 1176x880": "1176x880",
    "Golden Ratio | Balanced (1.618:1) - 1296x800": "1296x800",
}

PSONA_UI_MODEL_SETTINGS: Dict = {
    "required": {
        "Model": (
            folder_paths.get_filename_list("checkpoints"),
            {"tooltip": "Use Illustrious-XL compatible models here."},
        ),
        "Stop at Clip Layer": (
            "INT",
            {"default": -2, "min": -2, "max": 10, "step": 1},
        ),
        "Resolution": (
            list(RESOLUTIONS.keys()),
            {
                "default": "Square | Model Preview (1:1) - 1024x1024",
                "tooltip": "Anime-focused aspect presets with usage hints.",
            },
        ),
        "Batch Size": (
            "INT",
            {
                "default": 1,
                "min": 1,
                "max": 100,
                "tooltip": "The number of latent images in the batch.",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 250000,
                "step": 1,
                "display": "slider",
            },
        ),
        "steps": (
            "INT",
            {
                "default": 30,
                "min": 1,
                "max": 100,
                "step": 1,
                "display": "slider",
            },
        ),
        "cfg": (
            "FLOAT",
            {
                "default": 7.0,
                "min": 1.0,
                "max": 30.0,
                "step": 0.1,
                "tooltip": "CFG Scale.",
            },
        ),
    },
}


class IllustriousMasterModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return PSONA_UI_MODEL_SETTINGS

    RETURN_TYPES = (
        "MODEL",
        "VAE",
        "CLIP",
        "LATENT",
        "INT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "MODEL",
        "VAE",
        "CLIP",
        "LATENT",
        "SEED",
        "STEPS",
        "CFG SCALE",
    )
    FUNCTION = "load_model_settings"
    CATEGORY = "Easy Illustrious "

    # FROM COMFYUI CORE
    def generate(self, width, height, batch_size=1) -> tuple:
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return ({"samples": latent},)

    # FROM COMFYUI CORE
    def modify_clip(self, clip, stop_at_clip_layer):
        clip.clip_layer(stop_at_clip_layer)
        return clip

    # FROM COMFYUI CORE
    def load_checkpoint(self, ckpt_name) -> tuple:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]

    def parse_resolution(self, resolution: str) -> tuple:
        dimensions = RESOLUTIONS[resolution].split("x")
        return int(dimensions[0]), int(dimensions[1])

    def load_model_settings(self, **kwargs):
        resolution = self.parse_resolution(kwargs["Resolution"])
        steps = int(kwargs.get("steps", 30))
        seed = int(kwargs.get("seed", random.randint(0, 250000)))
        batch_size = int(kwargs.get("Batch Size", 1))
        stop_at_clip_layer = kwargs.get("Stop at Clip Layer", -2)
        cfg = float(kwargs.get("cfg", 7.0))

        model, clip, vae = self.load_checkpoint(kwargs["Model"])
        clip = self.modify_clip(clip, stop_at_clip_layer)
        latent = self.generate(resolution[0], resolution[1], batch_size)[0]

        return (
            model,
            vae,
            clip,
            latent,
            seed,
            steps,
            cfg,
        )


class IllustriousPrompt:

    GIRLS: List[str] = [f"{n} girl{'s' if n > 1 else ' solo'}" for n in range(1, 11)]
    BOYS: List[str] = [f"{n} boy{'s' if n > 1 else ' solo'}" for n in range(1, 11)]
    current_year = datetime.now().year
    YEARS = [f"{n} year{'' if n > 1 else ''}" for n in range(2000, current_year + 1)]

    NEG = " ".join(
        [
            "(((watermark))), worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early, very displeasing, displeasing"
        ]
    )
    NEG_EXTRA = ", ".join(
        [
            "ai-generated, worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early",
            "very displeasing, displeasing, adversarial noise, what, off-topic, text, artist name, signature, username, logo",
            "watermark, copyright name, copyright symbol, low quality, lowres, jpeg artifacts, compression artifacts, blurry",
            "artistic error, bad anatomy, bad hands, bad feet, disfigured, deformed, extra digits, fewer digits, missing fingers",
            "censored, unfinished, bad proportions, bad perspective, monochrome, sketch, concept art, unclear, 2koma, 4koma,",
            "letterboxed, speech bubble, cropped",
        ]
    )
    NEG_BOOST = ", ".join(
        [
            "ai-generated, ai-assisted, stable diffusion, nai diffusion, worst quality, worst aesthetic, bad quality, normal quality, average quality, oldest, old, early, very displeasing",
            "displeasing, adversarial noise, unknown artist, banned artist, what, off-topic, artist request, text, artist name, signature, username, logo, watermark, copyright name, copyright symbol",
            "resized, downscaled, source larger, low quality, lowres, jpeg artifacts, compression artifacts, blurry, artistic error, bad anatomy, bad hands, bad feet, disfigured, deformed, extra digits",
            "fewer digits, missing fingers, censored, bar censor, mosaic censoring, missing, extra, fewer, bad, hyper, error, ugly, worst, tagme, unfinished, bad proportions, bad perspective, aliasing",
            "simple background, asymmetrical, monochrome, sketch, concept art, flat color, flat colors, simple shading, jaggy lines, traditional media \\(artwork\\), microsoft paint \\(artwork\\), ms paint \\(medium\\)",
            "unclear, photo, icon, multiple views, sequence, comic, 2koma, 4koma, multiple images, turnaround, collage, panel skew, letterboxed, framed, border, speech bubble, 3d, lossy-lossless, scan artifacts",
            "out of frame, cropped,",
        ]
    )

    NEG_ADDITIONAL = ", ".join([",(abstract:0.91), (doesnotexist:0.91)"])

    # Enhanced negative prompts based on Illustrious best practices
    NEG_ARTICLE = "(lowres:1.2), (worst quality:1.4), (low quality:1.4), (bad anatomy:1.4), multiple views, jpeg artifacts, artist name, censored, censored nipples, young, 2D"

    # New negative prompts from latest article examples
    NEG_LATEST = "worst_quality, bad_quality, poorly_detailed, jpeg_artifacts, extra_fingers, malformed_hands, blurry, compression_artifacts, pixelated"

    NEGATIVES: Dict[str, str] = {
        "Basic": NEG,
        "Extra": NEG_EXTRA,
        "Boost": NEG_BOOST,
        "Article (Recommended)": NEG_ARTICLE,
        "Latest (Underscores)": NEG_LATEST,
    }

    QUAILTY_BOOST = "high_quality, best_quality, highres, masterpiece, detailed_eyes, 8k, professional, detailed, subject_focus, depth_of_field"
    CINEMATIC = "(volumetric lighting:1.1, dof:1.1, depth of field:1.1)"

    # Quality variations based on article examples
    QUALITY_VARIATIONS: Dict[str, str] = {
        "Standard": "high_quality, best_quality, highres, masterpiece, detailed_eyes",
        "Ultra Detailed": "absurdres, very_detailed, extremely_detailed, ultra_detailed",
        "Sharp Focus": "sharp_focus, clean_lines, high_detail",
        "Perfect Anatomy": "perfect_anatomy, detailed, professional",
    }

    # Additional style presets for quality boosts
    QUALITY_STYLES: Dict[str, str] = {
        "Painterly": "painterly, brushwork, rich texture, color depth",
        "Photorealistic": "photorealistic, realistic_skin, fine details, film grain",
        "Anime Clean": "clean_lines, saturated_colors, cel_shading, crisp edges",
        "Cinematic": "cinematic lighting, dramatic shadows, color grading",
    }

    # Comprehensive hairstyle taxonomy used by hairstyle/clothing helper nodes
    HAIRSTYLE: Dict[str, dict | list] = {
        "Length and Volume": [
            "very long hair",
            "long hair",
            "medium hair",
            "short hair",
            "very short hair",
            "thick hair",
            "thin hair",
            "fluffy hair",
            "big hair",
        ],
        "Haircuts": {
            "Short": ["bob cut", "pixie cut", "crew cut", "bowl cut"],
            "Medium": ["hime cut", "layered hair", "shag cut"],
            "Long": ["layered long hair", "straight long hair", "wavy long hair"],
        },
        "Hairstyles": {
            "Tails": [
                "twintails",
                "low twintails",
                "short twintails",
                "twisted hair",
            ],
            "Braids": [
                "front braid",
                "side braid",
                "french braid",
                "single braid",
                "twin braids",
                "half up braid",
                "low-braided long hair",
                "cornrows",
                "dreadlocks",
            ],
            "Hair buns": [
                "braided bun",
                "single hair bun",
                "double bun",
                "cone hair bun",
                "doughnut hair bun",
            ],
            "Hair rings": ["hair rings", "single hair ring"],
            "Ponytails": [
                "ponytails",
                "folded ponytail",
                "front ponytail",
                "high ponytail",
                "short pontail",
                "side ponytail",
                "topknot",
            ],
            "Tall hair": ["afro", "beehive hairdo", "crested hair", "pompadour"],
            "Hair texture": [
                "wavy hair",
                "straight hair",
                "spiked hair",
                "ringlets",
                "pointy hair",
                "messy hair",
                "hair flaps",
                "twin drills",
                "drill hair",
                "curly hair",
            ],
        },
        "Hairstyle Front": {
            "Bangs": [
                "bangs",
                "arched bangs",
                "asymmetrical bangs",
                "bangs pinned back",
                "blunt bangs",
                "crossed bangs",
                "diagonal bangs",
                "hair over eyes",
                "hair over one eye",
                "long bangs",
                "parted bangs",
                "curtained hair",
                "wispy bangs",
                "short bangs",
                "hair between eyes",
            ],
            "Hair intakes": ["hair intakes", "single hair intake"],
            "Sidelocks": [
                "sidelocks",
                "asymmetrical sidelocks",
                "drill sidelocks",
                "low-tied sidelocks",
                "single sidelocks",
                "widow's peak",
            ],
        },
        "Hairstyle Top": {
            "Top of the head": [
                "ahoge",
                "heart ahoge",
                "huge ahoge",
                "antenna hair",
                "heart antenna hair",
                "hair pulled back",
                "hair slicked back",
                "mohawk",
            ],
        },
        "Hair Colors": {
            "Aqua Hair": ["aqua hair", "dark aqua hair", "light aqua hair"],
            "Black Hair": [
                "black hair",
                "multicolred black hair",
                "gradient black hair",
            ],
            "Blonde Hair": [
                "blonde hair",
                "multicolred blonde hair",
                "gradient blonde hair",
            ],
            "Blue Hair": ["blue hair", "dark blue hair", "light blue hair"],
            "Light Blue Hair": [
                "light blue hair",
                "multicolred light blue hair",
                "gradient light blue hair",
            ],
            "Dark Blue Hair": [
                "dark blue hair",
                "multicolred dark blue hair",
                "gradient dark blue hair",
            ],
            "Brown Hair": ["brown hair", "dark brown hair", "light brown hair"],
            "Light Brown Hair": [
                "light brown hair",
                "multicolred light brown hair",
                "gradient light brown hair",
            ],
            "Green Hair": ["green hair", "dark green hair", "light green hair"],
            "Grey Hair": ["grey hair", "dark grey hair", "light grey hair"],
            "Orange Hair": ["orange hair", "dark orange hair", "light orange hair"],
            "Pink Hair": ["pink hair", "dark pink hair", "light pink hair"],
            "Purple Hair": ["purple hair", "dark purple hair", "light purple hair"],
            "Red Hair": ["red hair", "dark red hair", "light red hair"],
            "White Hair": ["white hair", "dark white hair", "light white hair"],
            "Colored Inner Hair": [
                "colored inner hair",
                "dark colored inner hair",
                "light colored inner hair",
            ],
            "Colored Tips": ["colored tips", "dark colored tips", "light colored tips"],
            "Roots": ["roots", "dark roots", "light roots"],
            "Gradient Hair": [
                "gradient hair",
                "dark gradient hair",
                "light gradient hair",
            ],
            "Patterned Hair": [
                "patterned hair",
                "dark patterned hair",
                "light patterned hair",
            ],
            "Rainbow Hair": ["rainbow hair", "dark rainbow hair", "light rainbow hair"],
            "Split-Color Hair": [
                "split-color hair",
                "dark split-color hair",
                "light split-color hair",
            ],
            "Spotted Hair": ["spotted hair", "dark spotted hair", "light spotted hair"],
            "Streaked Hair": [
                "streaked hair",
                "dark streaked hair",
                "light streaked hair",
            ],
            "Striped Hair": ["striped hair", "dark striped hair", "light striped hair"],
            "Raccoon Tails": [
                "raccoon tails",
                "dark raccoon tails",
                "light raccoon tails",
            ],
            "Two-Tone Hair": [
                "two-tone hair",
                "dark two-tone hair",
                "light two-tone hair",
            ],
        },
    }

    CLOTHING = {
        "outfits": [
            "tuxedo",
            "evening_gown",
            "canonicals",
            "cocktail_dress",
            "gown",
            "wedding_dress",
            "maid",
            "miko",
            "school_uniform",
            "sailor",
            "serafuku",
            "sailor_senshi_uniform",
            "summer_uniform",
            "naval_uniform",
            "military_uniform",
            "business_suit",
            "nurse",
            "chef_uniform",
            "labcoat",
            "cheerleader",
            "band_uniform",
            "space_suit",
            "leotard",
            "domineering",
            "cheongsam",
            "china_dress",
            "chinese_style",
            "traditional_clothes",
            "uchikake",
            "off-shoulder_dress",
            "sleeveless_kimono",
            "print_kimono",
            "japanese_clothes",
            "hanten_(clothes)",
            "hanbok",
            "korean_clothes",
            "german_clothes",
            "gothic",
            "lolita",
            "gothic_lolita",
            "byzantine_fashion",
            "tropical cloth",
            "indian_style",
            "Ao_Dai",
            "ainu_clothes",
            "arabian_clothes",
            "egyptian_clothes",
            "hawaii costume",
            "furisode",
            "animal_costume",
            "bunny_costume",
            "cat_costume",
            "dog_costume",
            "bear_costume",
            "santa_costume",
            "halloween_costume",
            "kourindou_tengu_costume",
            "meme_attire",
            "casual",
            "loungewear",
            "robe",
            "cloak",
            "hooded_cloak",
            "winter_clothes",
            "down jacket",
            "santa",
            "harem_outfit",
            "shrug_clothing",
            "gym_uniform",
            "athletic_leotard",
            "volleyball_uniform",
            "tennis_uniform",
            "baseball_uniform",
            "letterman_jacket",
            "biker_clothes",
            "bikesuit",
            "wrestling_outfit",
            "front_zipper_swimsuit",
            "shell_bikini",
            "frilled_swimsuit",
            "strapless_dress",
            "backless_dress",
            "halter_dress",
            "sundress",
            "sleeveless_dress",
            "sailor_dress",
            "summer_dress",
            "pinafore_dress",
            "frilled_dress",
            "sweater_dress",
            "armored_dress",
            "fur-trimmed_dress",
            "lace-trimmed_dress",
            "collared_dress",
            "layered_dress",
            "pleated_dress",
            "taut_dress",
            "pencil_dress",
            "multicolored_dress",
            "striped_dress",
            "polka_dot_dress",
            "plaid_dress",
            "print_dress",
            "vertical-striped_dress",
            "ribbed_dress",
            "short_jumpsuit",
            "multicolored_clothes",
            "expressive_clothes",
            "multicolored_bodysuit",
            "jumpsuit",  # moved from "top"
        ],
        "top": [
            "blouse",  # moved from "bottoms"
            "collared_shirt",
            "dress_shirt",
            "sailor_shirt",
            "cropped_shirt",
            "t-shirt",
            "off-shoulder_shirt",
            "shrug_clothing",
            "gym_shirt",
            "cardigan",
            "criss-cross_halter",
            "frilled_shirt",
            "sweatshirt",
            "hawaiian_shirt",
            "hoodie",
            "kappougi",
            "plaid_shirt",
            "polo_shirt",
            "print_shirt",
            "sleeveless_hoodie",
            "sleeveless_shirt",
            "striped_shirt",
            "tank_top",
            "vest",
            "waistcoat",
            "tied_shirt",
            "undershirt",
            "crop_top",
            "camisole",
            "midriff",
            "oversized_shirt",
            "borrowed_garments",
            "blazer",
            "overcoat",
            "double-breasted",
            "long_coat",
            "winter_coat",
            "hooded_coat",
            "fur_coat",
            "fur-trimmed_coat",
            "duffel_coat",
            "parka",
            "cropped_jacket",
            "track_jacket",
            "hooded_track_jacket",
            "military_jacket",
            "camouflage_jacket",
            "leather_jacket",
            "letterman_jacket",
            "fur_trimmed_jacket",
            "two-tone_jacket",
            "trench_coat",
            "windbreaker",
            "raincoat",
            "hagoromo",
            "tunic",
            "cape",
            "capelet",
            "sweater",
            "pullover_sweaters",
            "ribbed_sweater",
            "sweater_vest",
            "backless_sweater",
            "aran_sweater",
            "beige_sweater",
            "brown_sweater",
            "hooded_sweater",
            "off-shoulder_sweater",
            "striped_sweater",
            "puffer_jacket",
            "short_over_long_sleeves",
            "impossible_clothes",
            "heart_cutout",
            "ofuda_on_clothes",
            "front-tie_top",
            "jacket_on_shoulders",
        ],
        "bottoms": [
            "skirt",
            "mini_skirt",
            "skirt_suit",
            "bikini_skirt",
            "pleated_skirt",
            "pencil_skirt",
            "bubble_skirt",
            "tutu",
            "ballgown",
            "beltskirt",
            "denim_skirt",
            "suspender_skirt",
            "long_skirt",
            "summer_long_skirt",
            "hakama_skirt",
            "high-waist_skirt",
            "suspender_long_skirt",
            "chiffon_skirt",
            "lace_skirt",
            "ribbon-trimmed_skirt",
            "layered_skirt",
            "print_skirt",
            "multicolored_skirt",
            "striped_skirt",
            "plaid_skirt",
            "flared_skirt",
            "floral_skirt",
            "hot_pants",
            "striped_shorts",
            "suspender_shorts",
            "denim_shorts",
            "puffy_shorts",
            "dolphin_shorts",
            "dolfin_shorts",
            "tight_pants",
            "track_pants",
            "yoga_pants",
            "bike_shorts",
            "gym_shorts",
            "pants",
            "puffy_pants",
            "pumpkin_pants",
            "hakama_pants",
            "harem_pants",
            "bloomers",
            "buruma",
            "jeans",
            "cargo_pants",
            "camouflage_pants",
            "capri_pants",
            "chaps",
            "lowleg_pants",
            "plaid_pants",
            "single_pantsleg",
            "striped_pants",
            "torn_jeans",
            "hakama",
            "harness",
            "rigging",
            "waist_apron",
            "maid_apron",
            "waist_cape",
            "clothes_around_waist",
            "jacket_around_waist",
            "sweater_around_waist",
            "loincloth",
            "bustier",
            "corset",
            "girdle",
            "armor",
            "bikini_armor",
            "full_armor",
            "plate_armor",
            "japanese_armor",
            "kusazuri",
            "power_armor",
            "mecha",
            "helmet",
            "kabuto",
            "off-shoulder_armor",
            "shoulder_armor",
            "muneate",
            "breastplate",
            "faulds",
            "wringing_clothes",
            "shiny_clothes",
            "kariginu",
        ],
    }

    BACKGROUND = []

    WEIGHTS = {"LOW": ":0.6", "NORMAL": ":1.0", "HIGH": ":1.15", "VERY HIGH": ":1.4"}

    DEFAULT = " ".join(
        [
            "leaning against a wall in a alley, glitch hair, iridescent hair, holding gun, profile, backlighting, ",
            "scenery of a ruin city, patrol team in the background",
        ]
    )

    CHAIN_INSERT_TOKEN = "[EN122112_CHAIN]"

    # Camera/shot/style dictionaries used by UI and prompt mapping
    SHOT_TYPES: Dict[str, str] = {
        "Extreme Close-Up": "extreme close-up",
        "Close-Up": "close-up",
        "Portrait": "portrait",
        "Bust": "bust shot",
        "Half Body": "upper body",
        "Three-Quarter": "three-quarter view",
        "Full Body": "full body",
        "Wide Shot": "wide shot",
    }

    FRAMING: Dict[str, str] = {
        "Portrait": "portrait",
        "Profile": "profile",
        "Upper Body": "upper body",
        "Lower Body": "lower body",
        "On Back": "on back, inverted",
        "Feet Out Of Frame": "feet out of frame",
        "Cowboy Shot": "cowboy shot",
        "Full Body": "full body",
        "Wide Shot": "wide shot",
        "Very Wide ": "very wide",
        "Cropped Arms": "cropped arms",
        "Cropped Legs": "cropped legs",
        "Cropped Shoulders": "cropped shoulders",
        "Cropped Head": "cropped head",
        "Cropped Torso": "cropped torso",
        "Close-up": "close-up",
        "Cut-in": "cut-in",
        "Split crop": "split crop",
        "Multiple Views": "multiple views",
    }

    PERSPECTIVE: Dict[str, str] = {
        "Atmospheric Perspective": "atmospheric perspective",
        "Fisheye": "fisheye",
        "Panorama": "panorama",
        "Perspective": "perspective",
        "Vanishing Point": "vanishing point",
        "Variations": "variations",
    }

    FOCUS: Dict[str, str] = {
        "Subject Focus": "subject focus",
        "Shallow DOF": "shallow depth of field",
        "Deep DOF": "deep focus",
        "Soft Focus": "soft focus",
        "Sharp Focus": "sharp focus",
    }

    # Legacy Noobai vocabularies (kept for compatibility with trained tag sets)
    NOOBAI_FRAMING: Dict[str, str] = {
        "Portrait": "portrait",
        "Profile": "profile",
        "Upper Body": "upper body",
        "Lower Body": "lower body",
        "On Back": "on back, inverted",
        "Feet Out Of Frame": "feet out of frame",
        "Cowboy Shot": "cowboy shot",
        "Full Body": "full body",
        "Wide Shot": "wide shot",
        "Very Wide ": "very wide",
        "Cropped Arms": "cropped arms",
        "Cropped Legs": "cropped legs",
        "Cropped Shoulders": "cropped shoulders",
        "Cropped Head": "cropped head",
        "Cropped Torso": "cropped torso",
        "Close-up": "close-up",
        "Cut-in": "cut-in",
        "Split crop": "split crop",
        "Multiple Views": "multiple views",
    }

    NOOBAI_FOCUS: Dict[str, str] = {
        "Animal Focus": "(animal focus:1.4)",
        "Armpit Focus": "(armpit focus:1.4)",
        "Ass Focus": "(ass focus:1.4)",
        "Back Focus": "(back focus:1.4)",
        "Book Focus": "(book focus:1.4)",
        "Breast Focus": "(breast focus:1.4)",
        "Cloud Focus": "(cloud focus:1.4)",
        "Eye Focus": "(eye focus:1.4)",
        "Food Focus": "(food focus:1.4)",
        "Foot Focus": "(foot focus:1.4)",
        "Hand Focus": "(hand focus:1.4)",
        "Hip Focus": "(hip focus:1.4)",
        "Male Focus": "(male focus:1.4)",
        "Monster Focus": "(monster focus:1.4)",
        "Navel Focus": "(navel focus:1.4)",
        "Object Focus": "(object focus:1.4)",
        "Other Focus": "(other focus:1.4)",
        "Plant Focus": "(plant focus:1.4)",
        "Pectoral Focus": "(pectoral focus:1.4)",
        "Solo Focus": "(solo focus:1.4)",
        "Vehicle Focus": "(vehicle focus:1.4)",
        "Text Focus": "(text focus:1.4)",
        "Thigh Focus": "(thigh focus:1.4)",
        "Weapon Focus": "(weapon focus:1.4)",
        "Wind Chime Focus": "(wind chime focus:1.4)",
    }

    # Illustrious-trained shot angle/orientation terms (weighted)
    SHOT_ANGLES: Dict[str, str] = {
        "Dutch Angle": "(dutch angle:1.15)",
        "From Above": "(from above:1.15)",
        "From Behind": "(from behind:1.15)",
        "From Below": "(from below:1.15)",
        "From Side": "(from side:1.15)",
        "Upside Down": "(upside down:1.15)",
        "High Up": "(high up:1.15)",
        "Multiple Views": "(multiple views:1.15)",
        "Sideways": "(sideways:1.15)",
        "Straight-On": "(straight-on:1.15)",
        "Three Quarter View": "(three quarter view:1.15)",
    }

    # Unified options used by UI
    # Avoid duplicating framing-style entries in Shot Types now that FRAMING holds those.
    _FRAMING_KEYS_NORM = set([k.lower().replace("-", " ").strip() for k in FRAMING.keys()])
    _COMBINED_ST: Dict[str, str] = {}
    for k, v in SHOT_TYPES.items():
        if k.lower().replace("-", " ").strip() not in _FRAMING_KEYS_NORM:
            _COMBINED_ST[k] = v
    COMBINED_SHOT_TYPES: Dict[str, str] = _COMBINED_ST
    COMBINED_FOCUS: Dict[str, str] = {**FOCUS, **NOOBAI_FOCUS}

    LIGHTING: Dict[str, str] = {
        "Soft": "soft lighting",
        "Hard": "hard lighting",
        "Backlighting": "backlighting",
        "Rim": "rim lighting",
        "Volumetric": "volumetric lighting",
        "Golden Hour": "golden hour lighting",
        "Noon": "bright daylight",
        "Night": "night lighting",
    }

    DEPTH_AND_FOCUS: Dict[str, str] = {
        "Bokeh": "bokeh",
        "Tilt Shift": "tilt-shift effect",
        "Depth Cues": "strong depth cues",
        "Fog/Mist": "atmospheric fog",
    }

    MOTION_EXPRESSION: Dict[str, str] = {
        "Dynamic": "dynamic pose",
        "Action": "action pose",
        "Calm": "calm expression",
        "Smiling": "smiling",
        "Serious": "serious expression",
    }

    # SFW negative helper
    CENSORSHIP: str = "nsfw, nude, nipples, explicit, porn, exposed, sex"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        UI_ILLUSTRIOUS_PROMPT: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
                "Add Chain Insert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": f"If True, places '{cls.CHAIN_INSERT_TOKEN}' for the next chained node to insert its content.",
                    },
                ),
                # Prompt weighting preferences (used to shape weight suffix magnitudes only; actual enforcement happens at encoding)
                "Weight Interpretation": (
                    ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                    {
                        "default": "comfy",
                        "tooltip": "How to interpret token weights; used here to gently adjust (:w) magnitudes.",
                    },
                ),
                "Token Normalization": (
                    ["none", "mean", "length", "length+mean"],
                    {
                        "default": "none",
                        "tooltip": "Normalization hint; reduces weight magnitude slightly for long tokens.",
                    },
                ),
            },
            "required": {
                "Character": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Type character name (e.g. hatsune_miku, saber, nezuko)",
                        "placeholder": "Search characters...",
                    },
                ),
                "Artist": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Type artist name (e.g. kantoku, wlop, artgerm)",
                        "placeholder": "Search artists...",
                    },
                ),
                "E621 Character": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Type E621 character name (e.g. lucario, umbreon)",
                        "placeholder": "Search E621 characters...",
                    },
                ),
                "E621 Artist": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Type E621 artist name (e.g. wolfy-nail, dimwitdog)",
                        "placeholder": "Search E621 artists...",
                    },
                ),
                "Character Weight": (
                    ["-"] + list(cls.WEIGHTS.keys()),
                    {"default": "-", "tooltip": "Weight of the Character"},
                ),
                "Artist Weight": (
                    ["-"] + list(cls.WEIGHTS.keys()),
                    {"default": "-", "tooltip": "Weight of the Artist"},
                ),
                "Girl Characters": (
                    ["-"] + cls.GIRLS,
                    {"default": "-", "tooltip": "Number of Girl Characters"},
                ),
                "Boy Characters": (
                    ["-"] + cls.BOYS,
                    {"default": "-", "tooltip": "Number of Boy Characters"},
                ),
                "Mature Characters": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Changes Girl to Woman, Boy to Man"},
                ),
                "Year": (
                    ["-"] + cls.YEARS,
                    {"default": "-", "tooltip": "Year of the character"},
                ),
                "Shot Type": (
                    ["-"] + list(cls.COMBINED_SHOT_TYPES.keys()),
                    {"default": "-", "tooltip": "Type of shot"},
                ),
                "Framing": (
                    ["-"] + list(cls.FRAMING.keys()),
                    {"default": "-", "tooltip": "Composition/framing rules"},
                ),
                "Perspective": (
                    ["-"] + list(cls.PERSPECTIVE.keys()),
                    {"default": "-", "tooltip": "Type of Chracter perspective"},
                ),
                "Focus": (
                    ["-"] + list(cls.COMBINED_FOCUS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Focus terms trained into Illustrious models",
                    },
                ),
                "Shot Angle": (
                    ["-"] + list(cls.SHOT_ANGLES.keys()),
                    {
                        "default": "-",
                        "tooltip": "Camera angle/orientation (Illustrious-trained)",
                    },
                ),
                "Lighting": (
                    ["-"] + list(cls.LIGHTING.keys()),
                    {"default": "-", "tooltip": "Lighting style for the scene"},
                ),
                "Depth & Focus": (
                    ["-"] + list(cls.DEPTH_AND_FOCUS.keys()),
                    {"default": "-", "tooltip": "Depth of field and focus effects"},
                ),
                "Motion & Expression": (
                    ["-"] + list(cls.MOTION_EXPRESSION.keys()),
                    {
                        "default": "-",
                        "tooltip": "Character motion and facial expressions",
                    },
                ),
                "Prompt": (
                    "STRING",
                    {"default": cls.DEFAULT, "multiline": True},
                ),
                "Negative Prompt": (
                    ["-"] + list(cls.NEGATIVES.keys()),
                    {
                        "default": "Latest (Underscores)",
                        "tooltip": "Select the type of negative prompt to use.",
                    },
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Will format the tag using \\( and \\)",
                    },
                ),
                "Break Format": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Break the prompt into multiple tokens.",
                    },
                ),
                "SFW": (
                    "BOOLEAN",
                    {"default": False, "forceInput": False, "tooltip": "Safe for Work"},
                ),
                "Quality Boost (Beta)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Boost the quality of the image using negative prompts.",
                    },
                ),
                "Quality Style": (
                    ["-"] + list(cls.QUALITY_STYLES.keys()),
                    {
                        "default": "-",
                        "tooltip": "Add style-specific quality terms based on Illustrious guide",
                    },
                ),
                "Quality Variation": (
                    ["-"] + list(cls.QUALITY_VARIATIONS.keys()),
                    {
                        "default": "-",
                        "tooltip": "Add quality variation tags (can combine with Quality Boost)",
                    },
                ),
                "Prefix QB (Beta)": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Move the quality boost to the end of the prompt.",
                    },
                ),
                ("Cinematic (Beta)"): (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add cinematic elements to the prompt.",
                    },
                ),
            },
        }
        return UI_ILLUSTRIOUS_PROMPT

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "PROMPT",
        "NEGATIVE",
    )
    OUTPUT_IS_LIST = (
        False,
        False,
    )
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def _join_prompt_parts(self, *parts: str) -> str:
        """Helper to join non-empty prompt parts with ', ' and clean up."""
        filtered_parts = [p.strip().removesuffix(",") for p in parts if p and p.strip()]
        joined = ", ".join(filtered_parts)
        # Final cleanup for stray commas or multiple commas
        joined = re.sub(r"\s*,\s*", ", ", joined).strip()
        joined = re.sub(r",{2,}", ",", joined)
        joined = joined.removeprefix(",").removesuffix(",").strip()
        return joined

    def _dedupe_prompt_items(self, prompt: str) -> str:
        """Remove duplicate items from a comma-separated prompt string.
        - Prefers weighted variants over unweighted when duplicates exist.
        - Treats surrounding parentheses (and escaped \( \)) as wrappers only when they wrap the entire item.
        - Normalizes by lowercasing and trimming trailing numeric weights like :1.15.
        """
        if not prompt:
            return prompt
        parts = [p.strip() for p in prompt.split(",")]
        out: List[str] = []
        seen: Dict[str, Tuple[int, float, bool]] = {}

        def normalize(item: str) -> Tuple[str, float, bool]:
            raw = item.strip()
            has_weight = False
            weight_val = -1.0  # -1 means no explicit weight detected
            s = raw
            # Strip full-item wrapping parentheses, including escaped
            if (s.startswith(r"\(") and s.endswith(r"\)")):
                s_inner = s[2:-2].strip()
                s = s_inner
            elif s.startswith("(") and s.endswith(")"):
                s_inner = s[1:-1].strip()
                s = s_inner
            # Detect and strip a trailing numeric weight (e.g., :1.15)
            m = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)\s*$", s)
            if m:
                has_weight = True
                try:
                    weight_val = float(m.group(1))
                except Exception:
                    weight_val = -1.0
                s = re.sub(r":\s*[0-9]+(?:\.[0-9]+)?\s*$", "", s)
            key = s.strip().lower()
            return key, weight_val, has_weight

        for part in parts:
            if not part:
                continue
            key, w, has_w = normalize(part)
            if key not in seen:
                seen[key] = (len(out), w, has_w)
                out.append(part)
            else:
                idx, w_old, has_w_old = seen[key]
                better = False
                # Prefer items with explicit weight over unweighted
                if has_w and not has_w_old:
                    better = True
                elif has_w and has_w_old and w > w_old:
                    better = True
                # Replace earlier entry if this is better
                if better:
                    out[idx] = part
                    seen[key] = (idx, w, has_w)

        # Rebuild string
        cleaned = ", ".join([p for p in out if p and p.strip()])
        cleaned = re.sub(r"\s*,\s*", ", ", cleaned).strip()
        cleaned = re.sub(r",{2,}", ",", cleaned)
        cleaned = cleaned.removeprefix(",").removesuffix(",").strip()
        return cleaned

    def construct(self, **kwargs) -> tuple:
        prefix_input_str = kwargs.get("prefix", "").strip()
        suffix_chain_input_str = kwargs.get(
            "suffix", ""
        ).strip()  # Suffix from this node's input
        add_insert_point_for_next_node = kwargs.get("Add Chain Insert", False)

        this_node_core_elements = []
        weight_interpretation = kwargs.get("Weight Interpretation", "comfy")
        token_normalization = kwargs.get("Token Normalization", "none")
        ca_weight = self._compute_weight_suffix(
            kwargs.get("Character Weight", "-"), weight_interpretation, token_normalization
        )
        aa_weight = self._compute_weight_suffix(
            kwargs.get("Artist Weight", "-"), weight_interpretation, token_normalization
        )

        if kwargs.get("Boy Characters", "-") != "-":
            this_node_core_elements.append(kwargs["Boy Characters"])
        if kwargs.get("Girl Characters", "-") != "-":
            this_node_core_elements.append(kwargs["Girl Characters"])

        format_tag_active = kwargs.get("Format Tag", True)
        char_val = kwargs.get("Character", "").strip()
        if char_val and char_val != "-":
            this_node_core_elements.append(
                f"\\({self.format_tag(char_val)}{ca_weight}\\)"
                if format_tag_active
                else f"{char_val}{ca_weight}"
            )

        echar_val = kwargs.get("E621 Character", "").strip()
        if echar_val and echar_val != "-":
            this_node_core_elements.append(
                f"\\({self.format_tag(echar_val)}{ca_weight}\\)"
                if format_tag_active
                else f"{echar_val}{ca_weight}"
            )

        art_val = kwargs.get("Artist", "").strip()
        if art_val and art_val != "-":
            tag_str = self.format_tag(art_val)
            this_node_core_elements.append(f"artist:{tag_str}{aa_weight}")

        eart_val = kwargs.get("E621 Artist", "").strip()
        if eart_val and eart_val != "-":
            tag_str = self.format_tag(eart_val)
            this_node_core_elements.append(f"artist:{tag_str}{aa_weight}")

        if kwargs.get("Year", "-") != "-":
            this_node_core_elements.append(kwargs["Year"])

        shots_map = {
            "Shot Type": IllustriousPrompt.COMBINED_SHOT_TYPES,
            "Shot Angle": IllustriousPrompt.SHOT_ANGLES,
            "Framing": IllustriousPrompt.FRAMING,
            "Perspective": IllustriousPrompt.PERSPECTIVE,
            "Focus": IllustriousPrompt.COMBINED_FOCUS,
            "Lighting": IllustriousPrompt.LIGHTING,
            "Depth & Focus": IllustriousPrompt.DEPTH_AND_FOCUS,
            "Motion & Expression": IllustriousPrompt.MOTION_EXPRESSION,
        }
        for key, D_MAP in shots_map.items():
            val = kwargs.get(key, "-")
            if val != "-" and D_MAP.get(val):
                this_node_core_elements.append(D_MAP.get(val))

        this_node_core_content_str = self._join_prompt_parts(*this_node_core_elements)

        working_prompt_str = ""
        deferred_suffix_from_parent = ""

        if IllustriousPrompt.CHAIN_INSERT_TOKEN in prefix_input_str:
            parts = prefix_input_str.split(IllustriousPrompt.CHAIN_INSERT_TOKEN, 1)
            prefix_head = parts[0].strip()
            if len(parts) > 1:
                deferred_suffix_from_parent = parts[1].strip()

            working_prompt_str = self._join_prompt_parts(
                prefix_head, this_node_core_content_str
            )
        else:
            working_prompt_str = self._join_prompt_parts(
                prefix_input_str, this_node_core_content_str
            )

        elements_after_core = []
        if add_insert_point_for_next_node:
            elements_after_core.append(IllustriousPrompt.CHAIN_INSERT_TOKEN)

        this_node_main_prompt_text = kwargs.get("Prompt", "").strip()
        if this_node_main_prompt_text:
            elements_after_core.append(this_node_main_prompt_text)

        if elements_after_core:
            additional_str = self._join_prompt_parts(*elements_after_core)
            working_prompt_str = self._join_prompt_parts(
                working_prompt_str, additional_str
            )

        prompt_after_own_modifiers = working_prompt_str

        sfw_positive_tag = ""
        if kwargs.get("SFW", False):
            sfw_positive_tag = "(sfw:1.2)"

        # Quality Boost, Style, and Cinematic
        qb_cinematic_string = ""
        if kwargs.get("Quality Boost (Beta)", False):
            qb_elements = [self.QUAILTY_BOOST.strip().removesuffix(",")]
            if kwargs.get("Cinematic (Beta)", False):
                qb_elements.append(self.CINEMATIC.strip().removesuffix(","))
            qb_cinematic_string = self._join_prompt_parts(*qb_elements)

        # Add Quality Style
        quality_style = kwargs.get("Quality Style", "-")
        if quality_style != "-" and quality_style in self.QUALITY_STYLES:
            quality_style_string = self.QUALITY_STYLES[quality_style]
            if qb_cinematic_string:
                qb_cinematic_string = self._join_prompt_parts(
                    qb_cinematic_string, quality_style_string
                )
            else:
                qb_cinematic_string = quality_style_string

        # Add Quality Variation
        quality_variation = kwargs.get("Quality Variation", "-")
        if quality_variation != "-" and quality_variation in self.QUALITY_VARIATIONS:
            quality_variation_string = self.QUALITY_VARIATIONS[quality_variation]
            if qb_cinematic_string:
                qb_cinematic_string = self._join_prompt_parts(
                    qb_cinematic_string, quality_variation_string
                )
            else:
                qb_cinematic_string = quality_variation_string

        if kwargs.get("Prefix QB (Beta)", False):
            prompt_after_own_modifiers = self._join_prompt_parts(
                qb_cinematic_string, sfw_positive_tag, prompt_after_own_modifiers
            )
        else:
            prompt_after_own_modifiers = self._join_prompt_parts(
                prompt_after_own_modifiers, qb_cinematic_string, sfw_positive_tag
            )

        final_prompt_assembly = self._join_prompt_parts(
            prompt_after_own_modifiers,
            deferred_suffix_from_parent,
            suffix_chain_input_str,
        )

        final_prompt_str = final_prompt_assembly
        if kwargs.get("Mature Characters", False):
            replacements = {
                "girls": "adult women",
                "girl": "adult woman",
                "boys": "adult men",
                "boy": "adult man",
            }
            for old, new_val in replacements.items():
                final_prompt_str = re.sub(
                    r"\b" + re.escape(old) + r"\b",
                    new_val,
                    final_prompt_str,
                    flags=re.IGNORECASE,
                )

        # Remove duplicate items before optional formatting
        final_prompt_str = self._dedupe_prompt_items(final_prompt_str)
        if kwargs.get("Break Format", False):
            final_prompt_str = self.format_prompt(final_prompt_str)

        negative_elements = []
        if kwargs.get("SFW", False):
            negative_elements.append(self.CENSORSHIP.strip())

        neg_prompt_choice = kwargs.get("Negative Prompt", "-")
        if neg_prompt_choice != "-":
            negative_elements.append(self.NEGATIVES[neg_prompt_choice])

        char_val_for_neg = kwargs.get("Character", "").strip()
        echar_val_for_neg = kwargs.get("E621 Character", "").strip()
        if (char_val_for_neg and char_val_for_neg != "-") or (
            echar_val_for_neg and echar_val_for_neg != "-"
        ):
            negative_elements.append(self.NEG_ADDITIONAL)

        final_negative_str = self._join_prompt_parts(*negative_elements)

        # De-dupe negatives as well
        final_negative_str = self._dedupe_prompt_items(final_negative_str)
        if kwargs.get("Break Format", False):
            final_negative_str = self.format_prompt(final_negative_str)

        return (
            final_prompt_str if final_prompt_str else " ",
            final_negative_str if final_negative_str else " ",
        )

    def _compute_weight_suffix(self, level: str, weight_interpretation: str, token_normalization: str) -> str:
        """Compute a (:w) suffix based on desired interpretation/normalization without changing outputs.
        Levels are one of '-', 'LOW', 'NORMAL', 'HIGH', 'VERY HIGH'.
        """
        if not level or level == "-":
            return ""
        base_map = {"LOW": 0.60, "NORMAL": 1.00, "HIGH": 1.15, "VERY HIGH": 1.40}
        base = float(base_map.get(level, 1.0))
        # Slightly reduce aggressive weights when normalization spreads long tokens
        if token_normalization in ("length", "length+mean") and base > 1.0:
            base = max(1.0, base - 0.05)
        # Down-weight mode caps at 1.0 (no up-weighting); give LOW a gentle 0.90
        if weight_interpretation == "down_weight":
            if base > 1.0:
                base = 1.0
            if level == "LOW":
                base = 0.90
        # Clamp sane bounds and format
        base = float(max(0.50, min(1.50, base)))
        return f":{base:.2f}"

    def format_prompt(self, prompt: str) -> str:
        prompt_no_commas = prompt.replace(",", "")
        words = prompt_no_commas.split()

        formatted_parts = []
        current_part = []
        current_length = 0

        for word in words:
            if word == IllustriousPrompt.CHAIN_INSERT_TOKEN:
                if current_part:
                    formatted_parts.append(
                        " ".join(current_part) + ("," if current_part else "")
                    )
                formatted_parts.append(word)
                current_part = []
                current_length = 0
                continue

            if current_length + len(word) + (1 if current_part else 0) > 70:
                if current_part:
                    last_elem_in_part = current_part[-1]
                    ends_with_comma = last_elem_in_part.endswith(",")
                    is_token = last_elem_in_part == IllustriousPrompt.CHAIN_INSERT_TOKEN

                    part_str = " ".join(current_part)
                    if not ends_with_comma and not is_token:
                        part_str += ","
                    formatted_parts.append(part_str)

                formatted_parts.append("BREAK")
                current_part = [word]
                current_length = len(word)
            else:
                current_part.append(word)
                current_length += len(word) + (1 if len(current_part) > 1 else 0)

        if current_part:
            part_str = " ".join(current_part)
            if (
                part_str != IllustriousPrompt.CHAIN_INSERT_TOKEN
                and not part_str.endswith(",")
            ):
                pass
            formatted_parts.append(part_str)

        final_formatted_prompt = " ".join(formatted_parts)
        final_formatted_prompt = re.sub(
            r"\s*BREAK\s*", " BREAK ", final_formatted_prompt
        ).strip()
        final_formatted_prompt = re.sub(r"\s*,\s*", ", ", final_formatted_prompt)
        final_formatted_prompt = re.sub(r",{2,}", ",", final_formatted_prompt)
        final_formatted_prompt = re.sub(r",\s*BREAK", ", BREAK", final_formatted_prompt)
        final_formatted_prompt = re.sub(r"BREAK\s*,", "BREAK ", final_formatted_prompt)

        return final_formatted_prompt

    def format_tag(self, tag: str) -> str:
        parts = tag.split("(")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_parts.append(part.replace("_", " "))
            else:
                subparts = part.split(")")
                if len(subparts) > 1:
                    formatted_parts.append(
                        "(" + subparts[0] + ")" + subparts[1].replace("_", " ")
                    )
                else:
                    formatted_parts.append("(" + part.replace("_", " "))
        return "".join(formatted_parts)


class IllustriousCharacters:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        UI_ILLUSTRIOUS_CHARACTERS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Character (Base)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Primary character",
                        "placeholder": "Search characters...",
                    },
                ),
                "Character (Secondary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Secondary character",
                        "placeholder": "Search characters...",
                    },
                ),
                "Character (Tertiary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Tertiary character",
                        "placeholder": "Search characters...",
                    },
                ),
                "Character (Quaternary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Quaternary character",
                        "placeholder": "Search characters...",
                    },
                ),
                "Character (Quinary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Quinary character",
                        "placeholder": "Search characters...",
                    },
                ),
                "Weighed Average": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Use averaged weights"},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Wrap with ( ) for emphasis"},
                ),
            },
        }
        return UI_ILLUSTRIOUS_CHARACTERS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CHARACTER PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        characters = [
            ("Character (Base)", 1.20),
            ("Character (Secondary)", 1.15),
            ("Character (Tertiary)", 1.12),
            ("Character (Quaternary)", 1.10),
            ("Character (Quinary)", 1.00),
        ]

        character_names = [name for name, _ in characters]
        if not any(kwargs.get(name, "").strip() for name in character_names):
            return (f"{prefix} {suffix}".strip(),)

        selected: List[Tuple[str, float]] = []
        for name, w in characters:
            val = kwargs.get(name, "").strip()
            if val and val != "-":
                selected.append((val, w))

        avg = self.calculate_avg_weight(selected)
        use_avg = kwargs.get("Weighed Average", False)
        if use_avg:
            weights = self.distribute_weights(avg, len(selected))
        else:
            weights = [w for _, w in selected]

        out_parts: List[str] = []
        for (val, _), w in zip(selected, weights):
            token = f"{val}:{w:.2f}"
            out_parts.append(f"\\({token}\\)" if kwargs.get("Format Tag") else token)

        content = ", ".join(out_parts)
        if suffix:
            content = f"{content}, {suffix}" if content else suffix

        if prefix:
            # Insert after second comma when available
            commas = list(re.finditer(r",", prefix))
            if len(commas) >= 2:
                idx = commas[1].end()
            else:
                idx = len(prefix)
            final = (
                f"{prefix[:idx].rstrip()} {content}, {prefix[idx:].lstrip()}"
                if content
                else prefix
            )
        else:
            final = content

        final = re.sub(r"\s+", " ", final).strip()
        return (final if final else " ",)

    @staticmethod
    def calculate_avg_weight(items: List[Tuple[str, float]]) -> float:
        if not items:
            return 0.0
        return sum(w for _, w in items) / len(items)

    @staticmethod
    def distribute_weights(avg_weight: float, count: int) -> List[float]:
        if count <= 1:
            return [min(round(avg_weight, 2), 1.15)] if count == 1 else []
        max_w = 1.15
        min_w = max(0, 2 * avg_weight - max_w)
        step = (max_w - min_w) / (count - 1)
        weights = [round(max_w - i * step, 2) for i in range(count)]
        total = sum(weights) or 1.0
        norm = [
            min(round(w * (avg_weight * count / total - 0.1), 2), 1.15) for w in weights
        ]
        return norm

    @staticmethod
    def format_tag(tag: str) -> str:
        parts = tag.split("(")
        formatted_parts: List[str] = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_parts.append(part.replace("_", " "))
            else:
                subparts = part.split(")")
                if len(subparts) > 1:
                    formatted_parts.append(
                        "(" + subparts[0] + ")" + subparts[1].replace("_", " ")
                    )
                else:
                    formatted_parts.append("(" + part.replace("_", " "))
        return "".join(formatted_parts)

    @staticmethod
    def add_prefix(prompt_elements: List[str], prefix: str) -> str:
        prompt = " ".join(prompt_elements).lower()
        gender_words = ["girl", "girls", "boy", "boys", "woman", "women", "man", "men"]
        pattern = r"\b(" + "|".join(gender_words) + r")\b"
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            idx = match.end()
            return f"{prompt[:idx]} {prefix} {prompt[idx:]}"
        return f"{prefix} {prompt}"


class IllustriousArtists:

    def __init__(self):
        # Placeholder for future instance-specific initialization
        # Currently no state needs to be maintained per instance
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        UI_ILLUSTRIOUS_ARTISTS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Artist (Base)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Artist Base, Heavily influences the prompt (e.g. kantoku, wlop)",
                        "placeholder": "Search artists...",
                    },
                ),
                "Artist (Secondary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Artist Secondary, influences the prompt",
                        "placeholder": "Search artists...",
                    },
                ),
                "Artist (Tertiary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Artist Tertiary, influences the prompt less",
                        "placeholder": "Search artists...",
                    },
                ),
                "Artist (Quaternary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Artist Quaternary, influences the prompt very little",
                        "placeholder": "Search artists...",
                    },
                ),
                "Artist (Quinary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Artist Quinary, influences the prompt very little",
                        "placeholder": "Search artists...",
                    },
                ),
                "Weighed Average": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Weighed Average of all artists"},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Will format the tag using \\( and \\)",
                    },
                ),
            },
        }
        return UI_ILLUSTRIOUS_ARTISTS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ARTIST PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        artists = [
            ("Artist (Base)", 1.2),
            ("Artist (Secondary)", 1.15),
            ("Artist (Tertiary)", 1.125),
            ("Artist (Quaternary)", 1.1),
            ("Artist (Quinary)", 1),
        ]

        artist_names = [artist[0] for artist in artists]
        if not any(kwargs.get(artist, "").strip() for artist in artist_names):
            return (f"{prefix} {suffix}".strip(),)

        artist_prompts: List[str] = []
        selected_artists = []
        for artist, weight in artists:
            artist_value = kwargs.get(artist, "").strip()
            if artist_value and artist_value != "-":
                selected_artists.append((artist_value, weight))

        avg_weight = self.calculate_avg_weight(selected_artists)
        use_avg_weight = kwargs.get("Weighed Average", False)

        if use_avg_weight:
            distributed_weights = self.distribute_weights(
                avg_weight, len(selected_artists)
            )
        else:
            distributed_weights = [weight for _, weight in selected_artists]

        for i, (artist_value, _) in enumerate(selected_artists):
            weight_to_use = distributed_weights[i]
            formatted_artist = f"{artist_value}:{weight_to_use:.2f}"

            final_artist = (
                f"(artist:{formatted_artist})"
                if kwargs.get("Format Tag")
                else f"{self.format_tag(formatted_artist)}"
            )
            artist_prompts.append(final_artist)

        # Join artist prompts with commas
        artist_prompts_str = ", ".join(artist_prompts)
        if suffix:
            artist_prompts_str += f", {suffix}"

        if prefix:
            character_names = list(CHARACTERS.keys())
            insert_index = self.find_insertion_point(prefix, character_names)
            final_prompt = f"{prefix[:insert_index].rstrip()}, {artist_prompts_str}, {prefix[insert_index:].lstrip()}"
        else:
            final_prompt = artist_prompts_str

        final_prompt = re.sub(r"\s+", " ", final_prompt).strip()

        return (final_prompt,)

    @staticmethod
    def find_insertion_point(text: str, character_names: List[str]) -> int:
        last_char_index: int = -1
        for name in character_names:
            pattern: str = rf"{re.escape(name)}(?:\s*\([^)]*\))?(?::[0-9.]+)?"
            matches: List[re.Match] = list(re.finditer(pattern, text))
            if matches:
                last_char_index = max(last_char_index, matches[-1].end())

        if last_char_index != -1:
            return last_char_index
        else:
            comma_match: Optional[re.Match] = re.search(r",\s*", text)
            if comma_match:
                return comma_match.end()
            else:
                return len(text)

    @staticmethod
    def calculate_avg_weight(characters: List[Tuple[str, float]]) -> float:
        if not characters:
            return 0
        total_weight = sum(weight for _, weight in characters)
        return total_weight / len(characters)

    @staticmethod
    def distribute_weights(avg_weight: float, count: int) -> List[float]:
        if count == 1:
            return [min(round(avg_weight, 2), 1.15)]

        max_weight = 1.15
        min_weight = max(
            0, 2 * avg_weight - max_weight
        )  # Ensure min_weight is non-negative

        step = (max_weight - min_weight) / (count - 1)

        weights = [round(max_weight - i * step, 2) for i in range(count)]
        total = sum(weights)
        normalized_weights = [
            min(round(w * (avg_weight * count / total - 0.1), 2), 1.15) for w in weights
        ]

        return normalized_weights

    @staticmethod
    def format_tag(tag: str) -> str:
        parts = tag.split("(")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                formatted_parts.append(part.replace("_", " "))
            else:
                subparts = part.split(")")
                if len(subparts) > 1:
                    formatted_parts.append(
                        "(" + subparts[0] + ")" + subparts[1].replace("_", " ")
                    )
                else:
                    formatted_parts.append("(" + part.replace("_", " "))

        return "".join(formatted_parts)


class IllustriousE621Characters:

    def __init__(self):
        # Placeholder for future instance-specific initialization
        # Currently no state needs to be maintained per instance
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        E621_UI_CHARACTERS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with existing tags.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with additional tags.",
                    },
                ),
            },
            "required": {
                "Character (Base)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Primary E621 character (e.g. lucario, umbreon)",
                        "placeholder": "Search E621 characters...",
                    },
                ),
                "Character (Secondary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Secondary E621 character",
                        "placeholder": "Search E621 characters...",
                    },
                ),
                "Character (Tertiary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Tertiary E621 character",
                        "placeholder": "Search E621 characters...",
                    },
                ),
                "Lore Tags": (
                    "STRING",
                    {"default": "", "tooltip": "Additional lore tags"},
                ),
                "General Tags": (
                    "STRING",
                    {"default": "", "tooltip": "General tags (min 10)"},
                ),
            },
        }

        return E621_UI_CHARACTERS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("E621 CHARACTER TAGS",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        characters = [
            "Character (Base)",
            "Character (Secondary)",
            "Character (Tertiary)",
        ]

        character_tags = []
        for char in characters:
            char_value = kwargs.get(char, "").strip()
            if char_value and char_value != "-":
                formatted_char = self.format_tag(char_value)
                character_tags.append(f"\\(character:{formatted_char}\\)")

        lore_tags = self.format_lore_tags(kwargs.get("Lore Tags", ""))
        general_tags = self.format_general_tags(kwargs.get("General Tags", ""))

        all_tags = character_tags + lore_tags + general_tags
        all_tags_str = ", ".join(all_tags)

        if prefix:
            final_prompt = f"{prefix} {all_tags_str}"
        else:
            final_prompt = all_tags_str

        if suffix:
            final_prompt += f" {suffix}"

        return (final_prompt.strip(),)

    @staticmethod
    def format_tag(tag: str) -> str:
        # Convert to lowercase, replace spaces with underscores, remove non-alphanumeric characters
        return re.sub(r"[^a-z0-9_]", "", tag.lower().replace(" ", "_"))

    @staticmethod
    def format_lore_tags(lore_tags: str) -> List[str]:
        return [
            f"lore:{IllustriousE621Characters.format_tag(tag)}"
            for tag in lore_tags.split(",")
            if tag.strip()
        ]

    @staticmethod
    def format_general_tags(general_tags: str) -> List[str]:
        tags = [
            IllustriousE621Characters.format_tag(tag)
            for tag in general_tags.split(",")
            if tag.strip()
        ]
        # # Ensure at least 10 general tags
        # while len(tags) < 10:
        #     tags.append("")
        return tags


class IllustriousE621Artists:

    def __init__(self):
        # Placeholder for future instance-specific initialization
        # Currently no state needs to be maintained per instance
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        E621_UI_ARTISTS: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with existing tags.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with additional tags.",
                    },
                ),
            },
            "required": {
                "Artist (Primary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Primary E621 artist (e.g. wolfy-nail, dimwitdog)",
                        "placeholder": "Search E621 artists...",
                    },
                ),
                "Artist (Secondary)": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Secondary E621 artist",
                        "placeholder": "Search E621 artists...",
                    },
                ),
                "Artist Aliases": (
                    "STRING",
                    {"default": "", "tooltip": "Additional artist aliases"},
                ),
                "Artist URLs": (
                    "STRING",
                    {"default": "", "tooltip": "Associated artist URLs"},
                ),
            },
        }

        return E621_UI_ARTISTS

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("E621 ARTIST TAGS",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        artists = ["Artist (Primary)", "Artist (Secondary)"]

        artist_tags = []
        for artist in artists:
            artist_value = kwargs.get(artist, "").strip()
            if artist_value and artist_value != "-":
                formatted_artist = self.format_tag(artist_value)
                artist_tags.append(f"artist:{formatted_artist}")

        alias_tags = self.format_aliases(kwargs.get("Artist Aliases", ""))
        url_tags = self.format_urls(kwargs.get("Artist URLs", ""))

        all_tags = artist_tags + alias_tags + url_tags
        all_tags_str = " ".join(all_tags)

        if prefix:
            final_prompt = f"{prefix} {all_tags_str}"
        else:
            final_prompt = all_tags_str

        if suffix:
            final_prompt += f" {suffix}"

        return (final_prompt.strip(),)

    @staticmethod
    def format_tag(tag: str) -> str:
        # Convert to lowercase, replace spaces with underscores, remove non-alphanumeric characters
        return re.sub(r"[^a-z0-9_]", "", tag.lower().replace(" ", "_"))

    @staticmethod
    def format_aliases(aliases: str) -> List[str]:
        return [
            f"alias:{IllustriousE621Artists.format_tag(alias)}"
            for alias in aliases.split(",")
            if alias.strip()
        ]

    @staticmethod
    def format_urls(urls: str) -> List[str]:
        return [f"url:{url.strip()}" for url in urls.split(",") if url.strip()]


class IllustriousHairstyles:
    def __init__(self):
        # Placeholder for future instance-specific initialization
        # Currently no state needs to be maintained per instance
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        HAIRSTYLE_UI: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Length and Volume": (
                    ["-"] + list(IllustriousPrompt.HAIRSTYLE["Length and Volume"]),
                    {"default": "-", "tooltip": "Select a Length and Volume"},
                ),
                "Haircuts": (
                    ["-"]
                    + [
                        f"{category.lower()} {option}"
                        for category, options in IllustriousPrompt.HAIRSTYLE[
                            "Haircuts"
                        ].items()
                        for option in options
                    ],
                    {"default": "-", "tooltip": "Select a haircut"},
                ),
                "Hairstyles": (
                    ["-"]
                    + [
                        f"{category.lower()} {option}"
                        for category, options in IllustriousPrompt.HAIRSTYLE[
                            "Hairstyles"
                        ].items()
                        for option in options
                    ],
                    {"default": "-", "tooltip": "Select a hairstyle"},
                ),
                "Hair Colors": (
                    ["-"]
                    + [
                        f"{option}"
                        for category, options in IllustriousPrompt.HAIRSTYLE[
                            "Hair Colors"
                        ].items()
                        for option in options
                    ],
                    {"default": "-", "tooltip": "Select a Hair Color"},
                ),
                "Inject Styles": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Inject into prefix content"},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Will format the tag using \\( and \\)",
                    },
                ),
                "Output Format": (
                    ["Tags (comma)", "Tags (space)", "Phrase"],
                    {
                        "default": "Tags (comma)",
                        "tooltip": "How to format the hairstyle: comma-separated tags, space-separated tags, or a short phrase.",
                    },
                ),
            },
        }

        return HAIRSTYLE_UI

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        # Get hairstyle components
        length_volume = kwargs.get("Length and Volume", "-")
        haircolors = kwargs.get("Hair Colors", "-")
        haircut = kwargs.get("Haircuts", "-")
        hairstyle = kwargs.get("Hairstyles", "-")
        inject_styles = kwargs.get("Inject Styles", True)
        format_tag = kwargs.get("Format Tag", False)
        output_format = kwargs.get("Output Format", "Tags (comma)")

        # Filter valid components and construct the sentence
        components = [
            haircolors if haircolors != "-" else "",
            length_volume if length_volume != "-" else "",
            haircut if haircut != "-" else "",
            hairstyle if hairstyle != "-" else "",
        ]

        # Build according to desired output format
        if output_format == "Tags (space)":
            tag_phrase = " ".join(filter(None, components)).strip()
        elif output_format == "Phrase":
            tag_phrase = self._build_hairstyle_phrase(haircolors, length_volume, haircut, hairstyle).strip()
            # Fallback if phrase couldn't be built meaningfully
            if not tag_phrase:
                tag_phrase = ", ".join([c for c in components if c]).strip()
        else:  # Tags (comma)
            tag_phrase = ", ".join([c for c in components if c]).strip()

        if not tag_phrase or not inject_styles:
            return (f"{prefix} {suffix}".strip(),)

        # Build the hairstyle sentence (leading space + trailing comma), with optional tag formatting
        if format_tag:
            hairstyle_sentence = f" \\({tag_phrase}\\),"
        else:
            hairstyle_sentence = f" {tag_phrase},"

        # Smart injection: Place hairstyle after hair color and before camera/framing terms
        if prefix:
            prompt = self.inject_hairstyle_smartly(prefix, hairstyle_sentence, suffix)
        else:
            # No prefix case
            prompt = f"{hairstyle_sentence} {suffix}".strip()

        # Cleanup stray spaces before commas
        prompt = prompt.replace(" ,", ",")
        return (prompt,)

    @staticmethod
    def _strip_suffix(text: str, suffix: str) -> str:
        return text[:-len(suffix)] if text.endswith(suffix) else text

    def _build_hairstyle_phrase(
        self,
        haircolors: str,
        length_volume: str,
        haircut: str,
        hairstyle: str,
    ) -> str:
        """Create a small, readable hairstyle phrase like 'long brown hair with twin braids'.
        Falls back to simple joining if insufficient parts are provided.
        """
        parts = []

        color_adj = ""
        length_adj = ""

        if haircolors and haircolors != "-":
            # convert 'brown hair' -> 'brown', but keep complex items as-is
            if haircolors.endswith(" hair"):
                color_adj = self._strip_suffix(haircolors, " hair")
            else:
                color_adj = haircolors

        if length_volume and length_volume != "-":
            if length_volume.endswith(" hair"):
                length_adj = self._strip_suffix(length_volume, " hair")
            else:
                length_adj = length_volume

        base = "".join([
            f"{length_adj} " if length_adj else "",
            f"{color_adj} " if color_adj else "",
        ]).strip()

        phrase = ""
        if base:
            phrase = f"{base} hair"
        # Append haircut and/or hairstyle in a simple readable way
        tail_bits = []
        if haircut and haircut != "-":
            tail_bits.append(f"styled as {haircut}")
        if hairstyle and hairstyle != "-":
            tail_bits.append(f"with {hairstyle}")

        if phrase and tail_bits:
            phrase = f"{phrase} {', '.join(tail_bits)}"
        elif not phrase and tail_bits:
            phrase = ", ".join(tail_bits)

        return phrase

    def inject_hairstyle_smartly(
        self, prefix: str, hairstyle_sentence: str, suffix: str
    ) -> str:
        """Insert hairstyle after hair color descriptions and before camera angles"""
        if not hairstyle_sentence:
            return f"{prefix} {suffix}".strip()

        # Common hair color terms that hairstyles should come after
        hair_color_terms = [
            "brown hair",
            "black hair",
            "blonde hair",
            "white hair",
            "silver hair",
            "gray hair",
            "grey hair",
            "red hair",
            "orange hair",
            "pink hair",
            "purple hair",
            "blue hair",
            "green hair",
            "aqua hair",
            "multicolored hair",
            "gradient hair",
            "two-tone hair",
            "streaked hair",
            "colored hair",
        ]

        # Camera/framing terms that hairstyles should come before
        camera_terms = [
            "straight-on",
            "from side",
            "from behind",
            "from above",
            "from below",
            "front view",
            "side view",
            "back view",
            "close up",
            "upper body",
            "portrait",
            "cropped torso",
            "dutch angle",
            "dynamic angle",
        ]

        prefix_lower = prefix.lower()

        # Strategy 1: Find hair color and insert after it
        best_hair_pos = -1
        for hair_term in hair_color_terms:
            pos = prefix_lower.find(hair_term)
            if pos != -1:
                # Find the end position after the hair term
                end_pos = pos + len(hair_term)
                # Look for comma after hair term
                comma_pos = prefix.find(",", end_pos)
                if comma_pos != -1:
                    best_hair_pos = max(best_hair_pos, comma_pos + 1)
                else:
                    best_hair_pos = max(best_hair_pos, end_pos)

        if best_hair_pos != -1:
            # Insert after hair color
            before = prefix[:best_hair_pos].strip()
            after = prefix[best_hair_pos:].strip()
            return f"{before},{hairstyle_sentence} {after} {suffix}".strip()

        # Strategy 2: Find camera terms and insert before them
        best_camera_pos = -1
        for camera_term in camera_terms:
            pos = prefix_lower.find(camera_term)
            if pos != -1:
                # Find the start of this term (look backwards for comma)
                comma_pos = prefix.rfind(",", 0, pos)
                if comma_pos != -1:
                    best_camera_pos = (
                        comma_pos
                        if best_camera_pos == -1
                        else min(best_camera_pos, comma_pos)
                    )
                else:
                    best_camera_pos = (
                        pos if best_camera_pos == -1 else min(best_camera_pos, pos)
                    )

        if best_camera_pos != -1:
            # Insert before camera terms
            before = prefix[:best_camera_pos].strip()
            after = prefix[best_camera_pos:].strip()
            return f"{before},{hairstyle_sentence} {after} {suffix}".strip()

        # Strategy 3: Fallback - append at the end of prefix
        return f"{prefix},{hairstyle_sentence} {suffix}".strip()


class IllustriousClothing:
    def __init__(self):
        # Placeholder for future instance-specific initialization
        # Currently no state needs to be maintained per instance
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        CLOTHING_UI: Dict = {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Prefix to the prompt with a custom string.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Suffix to the prompt with a custom string.",
                    },
                ),
            },
            "required": {
                "Outfits": (
                    ["-"] + IllustriousPrompt.CLOTHING["outfits"],
                    {"default": "-", "tooltip": "Select an outfit."},
                ),
                "Top": (
                    ["-"] + IllustriousPrompt.CLOTHING["top"],
                    {"default": "-", "tooltip": "Select a top clothing item."},
                ),
                "Bottoms": (
                    ["-"] + IllustriousPrompt.CLOTHING["bottoms"],
                    {"default": "-", "tooltip": "Select a bottom clothing item."},
                ),
                "General Style": (
                    ["-"] + list(getattr(IllustriousScenesPlus, "GENERAL_STYLES", [])) if 'IllustriousScenesPlus' in globals() else ["-"],
                    {"default": "-", "tooltip": "Overall clothing style/aesthetic."},
                ),
                "Headwear": (
                    ["-"] + list(getattr(IllustriousScenesPlus, "HEADWEAR", [])) if 'IllustriousScenesPlus' in globals() else ["-"],
                    {"default": "-", "tooltip": "Specific headwear item to include."},
                ),
                "Inject Clothing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Inject into prefix content."},
                ),
                "Format Tag": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Will format the tag using \\( and \\).",
                    },
                ),
            },
        }

        return CLOTHING_UI

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def construct(self, **kwargs) -> tuple:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()

        # Get clothing components
        outfit = kwargs.get("Outfits", "-")
        top = kwargs.get("Top", "-")
        bottoms = kwargs.get("Bottoms", "-")
        inject_styles = kwargs.get("Inject Clothing", True)
        format_tag = kwargs.get("Format Tag", False)

        # Filter valid components
        style = kwargs.get("General Style", "-")
        headwear = kwargs.get("Headwear", "-")
        components = [comp for comp in [outfit, top, bottoms, style, headwear] if comp != "-"]

        if not components or not inject_styles:
            return (f"{prefix} {suffix}".strip(),)

        # Create combined clothing string
        combined = ", ".join(components)
        formatted = f"\\({combined}\\)" if format_tag else combined

        # Handle prefix insertion logic
        if prefix:
            prompt = f"{prefix} {formatted} {suffix}".strip().replace(" ,", ",")
        else:
            # No prefix case
            prompt = f"{formatted} {suffix}".strip()

        return (prompt,)
