import random
import re
import time
from typing import Dict, List, Tuple

# Reuse your chain token if available from Illustrious prompt class
try:
    from .. import IllustriousPrompt as IP  # Illustrious prompt/model suite

    CHAIN_INSERT_TOKEN = IP.CHAIN_INSERT_TOKEN
except Exception:
    CHAIN_INSERT_TOKEN = "[EN122112_CHAIN]"


class ScenesPlus:
    """
    Anime Scene Vocabulary System - Comprehensive scene generation for Illustrious.

    Designed to be a 'fill-in-the-middle' but can run standalone.
    - NEVER emits quality/artist/lighting/DoF/camera/framing (your IllustriousPrompt owns those).
    - Emits: person description, pose/action, clothing, props, environment, time/weather, ambience, event.
    - Extended with comprehensive anime genre vocabulary and situations.
    - Toggle each section to avoid overlap with other nodes.
    """

    # --- Extended Anime Scene Vocabulary Pools ---
    PERSON_DESC = {
        "hair_color": [
            # Natural colors
            "blonde hair",
            "black hair",
            "brown hair",
            "white hair",
            "silver hair",
            "grey hair",
            # Vibrant anime colors
            "red hair",
            "orange hair",
            "pink hair",
            "purple hair",
            "blue hair",
            "green hair",
            "aqua hair",
            "teal hair",
            "violet hair",
            "magenta hair",
            "cyan hair",
            "mint hair",
            # Gradient/special styles
            "multicolored hair",
            "gradient hair",
            "streaked hair",
            "rainbow hair",
            "two-tone hair",
            "ombre hair",
        ],
        "hair_len": [
            "short hair",
            "medium hair",
            "long hair",
            "very long hair",
            "waist-length hair",
        ],
        "hair_style": [
            "twin tails",
            "ponytail",
            "side ponytail",
            "pigtails",
            "braid",
            "twin braids",
            "bun",
            "double bun",
            "messy hair",
            "straight hair",
            "wavy hair",
            "curly hair",
            "drill hair",
            "ahoge",
            "hair over one eye",
            "hair ribbon",
            "hair ornament",
        ],
        "eyes": [
            "blue eyes",
            "green eyes",
            "brown eyes",
            "purple eyes",
            "red eyes",
            "gold eyes",
            "grey eyes",
            "amber eyes",
            "violet eyes",
            "pink eyes",
            "silver eyes",
            "aqua eyes",
            "heterochromia",
            "glowing eyes",
            "star-shaped pupils",
            "heart-shaped pupils",
        ],
        "age_safe": ["woman", "man", "girl", "boy"],
        "expressions": [
            # Keep these generic to avoid overlap with explicit emotion/emote inputs
            "neutral expression",
            "gentle smile",
            "stoic",
            "calm",
        ],
        "extras": [
            "freckles",
            "beauty mark",
            "long eyelashes",
            # Avoid explicit facial expressions here to reduce overlap with emotion nodes
            "glasses",
            "sunglasses",
            "eye patch",
            "bandaid on face",
            "scar",
            "fang",
            "cat ears",
            "animal ears",
            "horns",
            "wings",
            "halo",
        ],
    }

    POSES = [
        # Basic poses
        "standing",
        "sitting",
        "walking",
        "running",
        "jumping",
        "flying",
        "floating",
        "lying down",
        "kneeling",
        "crouching",
        "leaning",
        "stretching",
        # Looking/facing
        "looking at viewer",
        "looking back",
        "looking away",
        "looking up",
        "looking down",
        "profile view",
        "from behind",
        "from side",
        # Hand gestures
        "waving",
        "peace sign",
        "thumbs up",
        "pointing",
        "reaching",
        "grabbing",
        "hands on hips",
        "arms crossed",
        "hands behind back",
        "hands clasped",
        "finger to mouth",
        "covering mouth",
        "hand on chin",
        "saluting",
        # Anime-specific poses
        "magical girl pose",
        "combat stance",
        "ninja pose",
        "idol pose",
        "school girl pose",
        "maid pose",
        "princess pose",
        "warrior pose",
        # Actions
        "dancing",
        "singing",
        "playing instrument",
        "reading",
        "writing",
        "studying",
        "cooking",
        "eating",
        "drinking",
        "sleeping",
        "yawning",
        "stretching",
        "exercising",
        "yoga pose",
        "meditation",
        "praying",
    ]

    CLOTHING_TOP = [
        # Casual tops
        "t-shirt",
        "tank top",
        "crop top",
        "tube top",
        "halter top",
        "off-shoulder top",
        "blouse",
        "shirt",
        "dress shirt",
        "button-up shirt",
        # Warm clothing
        "sweater",
        "pullover",
        "hoodie",
        "cardigan",
        "jacket",
        "blazer",
        "coat",
        "windbreaker",
        "bomber jacket",
        "denim jacket",
        "leather jacket",
        # School/uniform
        "sailor shirt",
        "school shirt",
        "uniform top",
        "vest",
        "waistcoat",
        "sailor collar",
        "serafuku",
        "blazer uniform top",
        # Special/anime
        "magical girl outfit top",
        "idol outfit",
        "cosplay outfit",
        "battle outfit",
    ]

    CLOTHING_BOTTOM = [
        # Skirts
        "skirt",
        "mini skirt",
        "pleated skirt",
        "long skirt",
        "maxi skirt",
        "pencil skirt",
        "frilly skirt",
        "tutu",
        "circle skirt",
        "asymmetrical skirt",
        # Shorts/pants
        "shorts",
        "hot pants",
        "denim shorts",
        "bike shorts",
        "gym shorts",
        "pants",
        "jeans",
        "leggings",
        "tights",
        "yoga pants",
        "sweatpants",
        "dress pants",
        "skinny jeans",
        "wide leg pants",
        "cargo pants",
        "bloomers",
    ]

    CLOTHING_OUTFIT = [
        # School/uniform
        "school uniform",
        "sailor uniform",
        "blazer uniform",
        "gym uniform",
        "summer uniform",
        "winter uniform",
        "private school uniform",
        # Dresses
        "dress",
        "summer dress",
        "sundress",
        "evening gown",
        "cocktail dress",
        "wedding dress",
        "party dress",
        "casual dress",
        "formal dress",
        # Traditional/cultural
        "kimono",
        "yukata",
        "hakama",
        "cheongsam",
        "qipao",
        "hanbok",
        "sari",
        # Occupational/role
        "maid outfit",
        "nurse outfit",
        "police uniform",
        "military uniform",
        "chef outfit",
        "waitress outfit",
        "librarian outfit",
        # Fantasy/anime
        "magical girl outfit",
        "witch outfit",
        "princess dress",
        "queen dress",
        "ninja outfit",
        "samurai outfit",
        "shrine maiden outfit",
        "miko outfit",
        "idol outfit",
        "performer outfit",
        "cosplay outfit",
        "gothic lolita",
        # Casual sets
        "business suit",
        "casual outfit",
        "sporty outfit",
        "beach outfit",
        "winter outfit",
        "autumn outfit",
        "spring outfit",
        "summer outfit",
    ]

    CLOTHING_EXTRAS = [
        # Neckwear
        "necktie",
        "bowtie",
        "ribbon",
        "choker",
        "necklace",
        "pendant",
        "collar",
        "scarf",
        "bandana",
        "ascot",
        # Accessories
        "hair ribbon",
        "hair band",
        "headband",
        "hair clip",
        "hair ornament",
        "scrunchie",
        "hair scrunchie",
        "barrette",
        # Simple headwear (kept for backward compatibility; full set below in HEADWEAR)
        "hat",
        "cap",
        "beret",
        "sun hat",
        "witch hat",
        "cat ears headband",
        # Jewelry
        "earrings",
        "bracelet",
        "ring",
        "anklet",
        "brooch",
        "pin",
        # Bags/items
        "handbag",
        "backpack",
        "school bag",
        "randoseru",
        "purse",
        "messenger bag",
        # Other
        "gloves",
        "mittens",
        "armband",
        "wristband",
        "belt",
        "sash",
        "stockings",
        "thigh highs",
        "knee socks",
        "ankle socks",
    ]

    # General clothing aesthetics/styles (derived from common Danbooru tag usage)
    GENERAL_STYLES = [
        "casual clothes",
        "streetwear",
        "techwear",
        "sportswear",
        "business attire",
        "formal wear",
        "gothic fashion",
        "gothic lolita",
        "lolita fashion",
        "classic lolita",
        "sweet lolita",
        "punk fashion",
        "grunge fashion",
        "preppy",
        "bohemian",
        "retro fashion",
        "vintage fashion",
        "steampunk",
        "dieselpunk",
        "cyberpunk fashion",
        "futuristic fashion",
        "military style",
        "school uniform style",
        "idol style",
        "magical girl style",
        "traditional japanese clothing",
        "traditional chinese clothing",
        "traditional korean clothing",
        "swimwear",
        "beachwear",
        "winter fashion",
        "summer fashion",
        "autumn fashion",
        "spring fashion",
    ]

    # Headwear taxonomy expanded (aligned with Danbooru tag conventions)
    HEADWEAR = [
        # Caps/Hats
        "baseball cap",
        "trucker cap",
        "visor",
        "beanie",
        "beret",
        "boater hat",
        "bonnet",
        "bowler hat",
        "bucket hat",
        "cowboy hat",
        "fedora",
        "flat cap",
        "newsboy cap",
        "sun hat",
        "straw hat",
        "top hat",
        "tricorn hat",
        "kepi",
        "ushanka",
        "tam o' shanter",
        # Themed
        "witch hat",
        "wizard hat",
        "party hat",
        "santa hat",
        # Crowns/tiaras
        "crown",
        "tiara",
        "circlet",
        "laurel wreath",
        "flower crown",
        # Bands/Head ties
        "headband",
        "ribbon headband",
        "bow headband",
        "hairband",
        "hair ribbon",
        "bandana",
        "headscarf",
        "kerchief",
        "hijab",
        "veil",
        "bridal veil",
        # Hoods/helmets
        "hood",
        "hood up",
        "hood down",
        "helmet",
        "bike helmet",
        "military helmet",
        "construction helmet",
        # On-head accessories
        "goggles on head",
        "sunglasses on head",
        "maid headband",
        "cat ears headband",
        "animal ears headband",
        # Traditional/special
        "komuso tengai",
    ]

    PROPS = [
        # School/study items
        "school bag",
        "backpack",
        "book",
        "notebook",
        "textbook",
        "pencil case",
        "pen",
        "pencil",
        "eraser",
        "ruler",
        "calculator",
        "laptop",
        "tablet",
        # Technology
        "phone",
        "smartphone",
        "camera",
        "headphones",
        "earbuds",
        "game console",
        "handheld gaming device",
        "mp3 player",
        "radio",
        "television",
        "flip phone",
        "portable cassette player",
        # Weather/outdoor
        "umbrella",
        "parasol",
        "sunglasses",
        "hat",
        "water bottle",
        "picnic basket",
        "sensu",
        "uchiwa",
        "geta",
        "zori",
        "tabi",
        # Food/drink
        "bento box",
        "lunch box",
        "tea cup",
        "coffee cup",
        "water bottle",
        "juice box",
        "ice cream",
        "cake",
        "sandwich",
        "rice ball",
        "onigiri",
        "takoyaki",
        "taiyaki",
        "dango",
        "yakisoba",
        "ramen",
        "chopsticks",
        # Decorative/atmospheric
        "flower petals",
        "sakura petals",
        "fallen leaves",
        "snow",
        "bubbles",
        "paper lanterns",
        "string lights",
        "candles",
        "incense",
        "wind chimes",
        # Furniture/environment
        "bench",
        "chair",
        "desk",
        "table",
        "bed",
        "couch",
        "pillow",
        "blanket",
        "vending machine",
        "street lamp",
        "traffic light",
        "mailbox",
        "fountain",
        "torii",
        "shrine bell",
        "ema",
        "omikuji",
        # Musical instruments
        "guitar",
        "piano",
        "violin",
        "flute",
        "drums",
        "microphone",
        # Sports/activities
        "ball",
        "basketball",
        "soccer ball",
        "tennis racket",
        "bicycle",
        "skateboard",
        "jump rope",
        "frisbee",
        "kite",
        # Magical/fantasy
        "magic wand",
        "crystal ball",
        "spell book",
        "potion bottle",
        "magic circle",
        "staff",
        "sword",
        "bow",
        "shield",
        "gem",
        "amulet",
        # Seasonal/holiday
        "christmas tree",
        "presents",
        "fireworks",
        "sparklers",
        "festival mask",
        "fan",
        "traditional fan",
        "paper fan",
        "bamboo",
        "lotus flower",
        "wind chime",
        "koinobori",
    ]

    CATEGORIES: Dict[str, Dict[str, List[str]]] = {
        "Daily Life": {
            "env": [
                # Urban/city
                "street",
                "city street",
                "suburban street",
                "alley",
                "crosswalk",
                "sidewalk",
                "convenience store",
                "supermarket",
                "shopping mall",
                "department store",
                "cafe",
                "restaurant",
                "fast food restaurant",
                "bakery",
                "bookstore",
                "library",
                "post office",
                "bank",
                "hospital",
                "pharmacy",
                # Home
                "bedroom",
                "living room",
                "kitchen",
                "bathroom",
                "dining room",
                "study room",
                "balcony",
                "garden",
                "backyard",
                "front yard",
                "garage",
                "attic",
                # School
                "classroom",
                "school hallway",
                "school entrance",
                "school courtyard",
                "cafeteria",
                "gym",
                "school library",
                "computer lab",
                "art room",
                "science lab",
                "home economics room",
                "music room",
                "club room",
                "shoe locker area",
                # Public spaces
                "bus stop",
                "train station",
                "subway station",
                "airport",
                "park bench",
                "public square",
                "market",
                "food court",
                "waiting room",
            ],
            "events": [
                "morning routine",
                "after school",
                "lunch break",
                "study session",
                "homework time",
                "family dinner",
                "weekend shopping",
                "daily commute",
                "evening walk",
                "bedtime routine",
                "cooking together",
                "cleaning house",
                "watching tv",
                "playing games",
                "reading time",
                "phone call with friends",
            ],
        },
        "Outdoor": {
            "env": [
                # Nature
                "park",
                "public park",
                "botanical garden",
                "flower garden",
                "rose garden",
                "beach",
                "sandy beach",
                "rocky shore",
                "pier",
                "boardwalk",
                "forest",
                "bamboo forest",
                "pine forest",
                "woods",
                "jungle",
                "mountain",
                "hill",
                "valley",
                "meadow",
                "field",
                "grassland",
                "lake",
                "pond",
                "river",
                "stream",
                "waterfall",
                "hot spring",
                # Cultural/religious
                "shrine",
                "temple",
                "torii gate",
                "stone steps",
                "prayer hall",
                "cemetery",
                "memorial",
                "statue",
                "monument",
                # Urban outdoor
                "rooftop",
                "balcony",
                "terrace",
                "patio",
                "courtyard",
                "city square",
                "plaza",
                "fountain",
                "bridge",
                "overpass",
                "festival street",
                "market street",
                "food stall area",
                "shotengai",
                "covered shopping street",
                "arcade street",
                # Sports/recreation
                "playground",
                "sports field",
                "tennis court",
                "basketball court",
                "swimming pool",
                "skating rink",
                "amusement park",
                "theme park",
            ],
            "events": [
                "picnic",
                "barbecue",
                "camping trip",
                "hiking",
                "nature walk",
                "street festival",
                "outdoor concert",
                "fireworks display",
                "fireworks night",
                "cosplay event",
                "hanami",
                "cherry blossom viewing",
                "autumn leaf viewing",
                "stargazing",
                "sunrise viewing",
                "sunset viewing",
                "morning jog",
                "evening stroll",
                "outdoor sports",
                "flying kites",
                "feeding birds",
                "photography walk",
            ],
        },
        "Indoor": {
            "env": [
                # Transportation
                "train interior",
                "subway car",
                "bus interior",
                "airplane cabin",
                "subway platform",
                "train platform",
                "waiting area",
                # Entertainment
                "movie theater",
                "concert hall",
                "theater",
                "opera house",
                "arcade",
                "game center",
                "karaoke box",
                "bowling alley",
                "museum",
                "art gallery",
                "exhibition hall",
                "aquarium",
                "planetarium",
                # Food/drink
                "teahouse",
                "coffee shop",
                "cafe corner",
                "restaurant interior",
                "bar",
                "pub",
                "izakaya",
                "ramen shop",
                "sushi restaurant",
                # Shopping
                "shop",
                "boutique",
                "clothing store",
                "electronics store",
                "bookstore interior",
                "record store",
                "antique shop",
                # Work/study
                "office",
                "studio",
                "art studio",
                "music studio",
                "workshop",
                "laboratory",
                "clinic",
                "salon",
                "spa",
                # Residential
                "apartment",
                "mansion",
                "traditional house",
                "modern house",
                "hotel room",
                "inn",
                "dormitory",
                "guest room",
            ],
            "events": [
                "quiet afternoon",
                "cozy evening",
                "late night study",
                "movie night",
                "game night",
                "karaoke session",
                "art class",
                "music lesson",
                "cooking class",
                "tea ceremony",
                "book club meeting",
                "closing time",
                "opening ceremony",
                "exhibition opening",
                "business meeting",
                "job interview",
                "doctor visit",
            ],
        },
        "Seasonal": {
            "env": [
                # Spring
                "cherry blossoms",
                "sakura trees",
                "blooming flowers",
                "spring garden",
                "rain-washed streets",
                "fresh green leaves",
                "flower field",
                # Summer
                "summer festival",
                "beach resort",
                "pool area",
                "summer camp",
                "sunflower field",
                "cicada-filled forest",
                "hot summer street",
                "fireworks venue",
                "outdoor festival",
                "summer night market",
                # Autumn
                "autumn leaves",
                "maple forest",
                "golden ginkgo trees",
                "harvest festival",
                "autumn park",
                "persimmon tree",
                # Winter
                "snow-covered landscape",
                "snowy street",
                "winter wonderland",
                "ice-covered lake",
                "ski resort",
                "hot spring in snow",
                "christmas market",
                "new year shrine",
                "lantern path",
                "illuminated street",
                "winter festival",
            ],
            "events": [
                # Spring
                "cherry blossom festival",
                "spring cleaning",
                "school entrance ceremony",
                "golden week",
                "children's day",
                "mother's day",
                # Summer
                "summer vacation",
                "beach trip",
                "summer festival",
                "bon festival",
                "fireworks festival",
                "camping trip",
                "pool party",
                "summer job",
                "heat haze",
                "cicada chorus",
                "watermelon splitting",
                # Autumn
                "autumn festival",
                "harvest time",
                "school festival",
                "sports day",
                "moon viewing",
                "autumn cleaning",
                "leaf collecting",
                # Winter
                "first snow",
                "christmas celebration",
                "new year celebration",
                "winter illumination",
                "skiing trip",
                "hot pot party",
                "valentine's day",
                "white day",
                "graduation ceremony",
            ],
        },
        "Atmospheric": {
            "env": [
                # Night scenes
                "neon-lit alley",
                "moonlit shrine",
                "starlit sky",
                "city lights at night",
                "lantern-lit street",
                "illuminated bridge",
                "night market",
                "glowing windows",
                "street lamp glow",
                "neon signs",
                # Weather effects
                "foggy forest",
                "misty mountain",
                "rainy window",
                "storm clouds",
                "rainbow after rain",
                "sun rays through clouds",
                "golden hour light",
                "blue hour twilight",
                "dawn breaking",
                "dusk settling",
                # Magical/dreamy
                "fairy tale forest",
                "enchanted garden",
                "crystal cave",
                "floating islands",
                "magical library",
                "starry void",
                "aurora borealis",
                "meteor shower",
                "lunar eclipse",
                # Urban atmospheric
                "empty subway platform",
                "abandoned building",
                "rooftop garden",
                "underground passage",
                "glass elevator",
                "modern architecture",
                "traditional architecture",
                "historical district",
            ],
            "events": [
                "soft drizzle",
                "gentle rain",
                "heavy downpour",
                "thunderstorm",
                "light mist",
                "thick fog",
                "morning dew",
                "frost formation",
                "aurora display",
                "meteor shower",
                "solar eclipse",
                "lunar eclipse",
                "rainbow appearance",
                "double rainbow",
                "sun dog",
                "light pillar",
                "quiet contemplation",
                "peaceful moment",
                "serene atmosphere",
                "mysterious encounter",
                "magical moment",
                "dream sequence",
            ],
        },
        "Action": {
            "env": [
                # Urban action
                "rain-soaked street",
                "busy intersection",
                "crowded station",
                "rooftop chase scene",
                "fire escape",
                "construction site",
                "parking garage",
                "alley chase",
                "bridge walkway",
                # Natural action
                "windy cliff",
                "rocky outcrop",
                "rushing river",
                "steep mountain path",
                "dense jungle",
                "desert dunes",
                "icy glacier",
                "active volcano",
                # Sports venues
                "stadium",
                "gymnasium",
                "dojo",
                "training ground",
                "race track",
                "swimming pool",
                "martial arts arena",
                "boxing ring",
                # Fantasy/adventure
                "ancient ruins",
                "mysterious cave",
                "haunted mansion",
                "magical battlefield",
                "sky fortress",
                "underwater palace",
                "dimensional rift",
                "portal gate",
                "floating platform",
            ],
            "events": [
                # Weather action
                "sudden downpour",
                "lightning strike",
                "tornado approach",
                "earthquake tremor",
                "avalanche",
                "landslide",
                "flood",
                "gust front",
                "sandstorm",
                "blizzard",
                # Physical action
                "chase sequence",
                "rescue mission",
                "race against time",
                "sports competition",
                "martial arts tournament",
                "dance battle",
                "cooking competition",
                "academic competition",
                # Adventure/fantasy
                "magical duel",
                "monster encounter",
                "treasure hunt",
                "exploration",
                "quest beginning",
                "final battle",
                "power awakening",
                "transformation sequence",
            ],
        },
        "Emotional": {
            "env": [
                # Intimate/personal spaces
                "empty classroom",
                "sunset rooftop",
                "quiet street",
                "bus stop",
                "park bench",
                "hospital room",
                "cemetery",
                "memorial site",
                "childhood home",
                "old neighborhood",
                "abandoned playground",
                # Transitional spaces
                "train platform",
                "airport departure gate",
                "school entrance",
                "graduation hall",
                "wedding chapel",
                "funeral home",
                # Contemplative spaces
                "library corner",
                "cafe by window",
                "garden pavilion",
                "shrine prayer area",
                "church pew",
                "meditation room",
                "art gallery",
                "music room",
                "study corner",
                # Memory/nostalgia spaces
                "old photo album",
                "childhood bedroom",
                "family restaurant",
                "school festival booth",
                "graduation ceremony",
                "first date location",
            ],
            "events": [
                # Parting/meeting
                "farewell moment",
                "goodbye scene",
                "reunion",
                "first meeting",
                "long-awaited reunion",
                "unexpected encounter",
                "chance meeting",
                # Achievement/growth
                "graduation day",
                "award ceremony",
                "job acceptance",
                "confession scene",
                "proposal moment",
                "wedding day",
                "birthday celebration",
                "coming of age ceremony",
                # Reflection/realization
                "moment of realization",
                "epiphany",
                "self-reflection",
                "looking back at memories",
                "contemplating future",
                "making important decision",
                "overcoming fear",
                # Relationships
                "heart-to-heart talk",
                "making up after fight",
                "first love",
                "friendship moment",
                "family bonding",
                "reconciliation",
                "supporting each other",
                "shared secret",
                "trust building",
            ],
        },

        "School Life": {
            "env": [
                "classroom",
                "school hallway",
                "library",
                "cafeteria",
                "gymnasium",
                "school courtyard",
                "music room",
                "art room",
                "science lab",
                "rooftop",
                "school festival booth",
                "club room",
                "student council room",
                "infirmary",
                "principal's office",
                "teacher's lounge",
                "storage room",
                "shoe locker area",
                "home economics room",
            ],
            "events": [
                "school festival",
                "sports festival",
                "cultural festival",
                "entrance exam",
                "graduation ceremony",
                "class trip",
                "club activities",
                "student council meeting",
                "exam period",
                "lunch break",
                "after school activities",
                "cleaning time",
            ],
        },
        "Fantasy Adventure": {
            "env": [
                "magical forest",
                "ancient castle",
                "dragon's lair",
                "mystical lake",
                "enchanted garden",
                "wizard's tower",
                "fairy village",
                "crystal cave",
                "floating island",
                "magical academy",
                "elf kingdom",
                "dwarf mine",
                "demon realm",
                "celestial palace",
                "time vortex",
                "parallel dimension",
            ],
            "events": [
                "magical awakening",
                "quest beginning",
                "boss battle",
                "power training",
                "artifact discovery",
                "spell casting",
                "summoning ritual",
                "transformation sequence",
                "magical duel",
                "prophecy fulfillment",
            ],
        },
        "Romance": {
            "env": [
                "romantic cafe",
                "moonlit park",
                "flower garden",
                "beach sunset",
                "ferris wheel",
                "restaurant terrace",
                "cozy apartment",
                "starlit rooftop",
                "cherry blossom path",
                "romantic bridge",
                "candle-lit room",
                "fireplace lounge",
                "rose garden",
                "love hotel",
            ],
            "events": [
                "first date",
                "confession scene",
                "first kiss",
                "proposal",
                "valentine's date",
                "anniversary celebration",
                "romantic dinner",
                "couple's trip",
                "wedding ceremony",
                "honeymoon",
                "romantic surprise",
                "intimate moment",
                "lovers' quarrel",
            ],
        },
        "Convention": {
            "env": [
                "convention center",
                "exhibition hall",
                "artist alley",
                "vendor hall",
                "panel room",
                "photo area",
                "autograph area",
                "registration desk",
                "cosplay gathering spot",
                "merch booth",
                "food court",
                "main stage",
                "gaming lounge",
                "screening room",
                "workshop space",
                "backstage area",
                "green room",
                "press room",
                "VIP lounge",
                "outdoor plaza",
                "hotel lobby",
                "hotel room",
                "parking garage",
                "loading dock",
                "storage room",
                "security checkpoint",
                "information booth",
                "lost and found",
                "first aid station",
                "quiet room",
                "nursing room",
                "restroom area",
                "merchandise pickup",
                "will call desk",
                "con suite",
                "tabletop gaming area",
                "video game arcade",
                "karaoke room",
                "dance floor",
                "outdoor courtyard",
                "balcony area",
                "corridor",
                "elevator lobby",
                "stairwell",
                "costume repair station",
                "prop check area",
                "streaming booth",
                "podcast studio"
            ],
            "events": [
                "anime convention",
                "comic convention",
                "cosplay event",
                "artist alley browsing",
                "panel discussion",
                "autograph session",
                "photo shoot",
                "merch shopping",
                "line waiting",
                "badge pickup",
                "gaming tournament",
                "anime screening",
                "cosplay contest",
                "voice acting panel",
                "art workshop",
                "meet and greet",
                "vendor hall browsing",
                "karaoke event",
                "dance competition",
                "trivia contest",
                "live performance",
                "premiere screening",
                "Q&A session",
                "art demonstration",
                "costume judging",
                "fan meetup",
                "room party",
                "networking event",
                "exhibition browsing",
                "food court dining",
                "charity auction",
                "speed dating",
                "scavenger hunt",
                "craft workshop",
                "gaming demo",
                "podcast recording",
                "interview session",
                "fashion show",
                "dance party",
                "outdoor photoshoot"
            ],
        },
    }

    # Deduplicate options within each category's env/events while preserving order
    for _cat_name, _cat_data in CATEGORIES.items():
        if isinstance(_cat_data, dict):
            if "env" in _cat_data and isinstance(_cat_data["env"], list):
                _seen_env = set()
                _env_out = []
                for _item in _cat_data["env"]:
                    if _item not in _seen_env:
                        _seen_env.add(_item)
                        _env_out.append(_item)
                _cat_data["env"] = _env_out
            if "events" in _cat_data and isinstance(_cat_data["events"], list):
                _seen_evt = set()
                _evt_out = []
                for _item in _cat_data["events"]:
                    if _item not in _seen_evt:
                        _seen_evt.add(_item)
                        _evt_out.append(_item)
                _cat_data["events"] = _evt_out

    TIME_WEATHER = [
        # Time of day
        "dawn",
        "early morning",
        "morning",
        "late morning",
        "noon",
        "midday",
        "afternoon",
        "late afternoon",
        "evening",
        "dusk",
        "twilight",
        "night",
        "midnight",
        "late night",
        "blue hour",
        "golden hour",
        # Weather conditions
        "sunny",
        "bright",
        "cloudy",
        "overcast",
        "partly cloudy",
        "light rain",
        "rainy",
        "heavy rain",
        "drizzle",
        "downpour",
        "storm",
        "thunderstorm",
        "snow",
        "snowing",
        "light snow",
        "heavy snow",
        "blizzard",
        "fog",
        "foggy",
        "mist",
        "misty",
        "haze",
        "humid",
        "windy",
        "breezy",
        "gust",
        "calm",
        "still air",
        # Seasonal weather
        "spring rain",
        "summer heat",
        "autumn breeze",
        "winter chill",
        "heat wave",
        "cold snap",
        "indian summer",
        "monsoon",
        # Atmospheric effects
        "rainbow",
        "double rainbow",
        "aurora",
        "meteor shower",
        "milky way",
        "starry sky",
        "full moon",
        "new moon",
        "crescent moon",
        "eclipse",
        "sunrise",
        "sunset",
    ]

    AMBIENCE = [
        # Peaceful/calm
        "peaceful",
        "serene",
        "tranquil",
        "calm",
        "quiet",
        "still",
        "zen",
        "meditative",
        "relaxing",
        "soothing",
        "gentle",
        "soft",
        # Emotional/reflective
        "melancholic",
        "nostalgic",
        "wistful",
        "bittersweet",
        "sentimental",
        "contemplative",
        "introspective",
        "thoughtful",
        "reflective",
        # Romantic/dreamy
        "romantic",
        "dreamy",
        "ethereal",
        "whimsical",
        "enchanting",
        "magical",
        "fairytale-like",
        "mystical",
        "otherworldly",
        # Energetic/dynamic
        "energetic",
        "vibrant",
        "lively",
        "bustling",
        "dynamic",
        "exciting",
        "thrilling",
        "electric",
        "pulsing",
        "rhythmic",
        # Dramatic/intense
        "dramatic",
        "intense",
        "powerful",
        "epic",
        "grand",
        "majestic",
        "awe-inspiring",
        "breathtaking",
        "spectacular",
        "cinematic",
        # Mysterious/dark
        "mysterious",
        "enigmatic",
        "secretive",
        "shadowy",
        "dark",
        "ominous",
        "foreboding",
        "eerie",
        "haunting",
        "spooky",
        # Cheerful/bright
        "cheerful",
        "bright",
        "sunny",
        "optimistic",
        "uplifting",
        "joyful",
        "happy",
        "festive",
        "festival atmosphere",
        "celebratory",
        "warm",
        # Cozy/intimate
        "cozy",
        "intimate",
        "homey",
        "comfortable",
        "snug",
        "welcoming",
        "familiar",
        "safe",
        "nurturing",
        "embracing",
        "moe",
    ]

    DENSITY = [
        # Solo/minimal
        "solo",
        "alone",
        "solitary",
        "isolated",
        "empty space",
        # Small groups
        "duo",
        "couple",
        "pair",
        "small group",
        "intimate gathering",
        "few people",
        "sparse crowd",
        "quiet crowd",
        # Medium crowds
        "moderate crowd",
        "gathering",
        "social setting",
        "group activity",
        "classroom full",
        "cafe busy",
        "park populated",
        # Large crowds
        "crowded",
        "bustling",
        "packed",
        "busy street",
        "festival crowd",
        "concert crowd",
        "rush hour",
        "packed train",
        "stadium full",
        "mass gathering",
        "overwhelming crowd",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Existing text before scene (chain-aware).",
                    },
                ),
                "suffix": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Existing text after scene."},
                ),
                "tipo_extra_negatives": (
                    "STRING",
                    {
                        "forceInput": False,
                        "tooltip": "Comma-separated terms to downweight/remove during TIPO optimization.",
                    },
                ),
            },
            "required": {
                # Scene knobs
                "Category": (list(cls.CATEGORIES.keys()), {"default": "Outdoor"}),
                "Complexity": (["simple", "medium", "detailed"], {"default": "medium"}),
                "Include Time/Weather": ("BOOLEAN", {"default": True}),
                "Include Ambience": ("BOOLEAN", {"default": True}),
                "Include Event": ("BOOLEAN", {"default": False}),
                "Include Prop": ("BOOLEAN", {"default": True}),
                "Include Density": ("BOOLEAN", {"default": False}),
                # Person/pose/clothes (all optional)
                "Include Person Description": ("BOOLEAN", {"default": False}),
                "Include Pose/Action": ("BOOLEAN", {"default": True}),
                "Include Clothing": ("BOOLEAN", {"default": False}),
                # Selectors (mirroring Scenes): exposed but optional via "-"
                "Outfits": (
                    ["-"] + cls.CLOTHING_OUTFIT,
                    {"default": "-", "tooltip": "Pick a full outfit or leave '-' for random."},
                ),
                "Top": (
                    ["-"] + cls.CLOTHING_TOP,
                    {"default": "-", "tooltip": "Pick a top; used if no Outfit is chosen."},
                ),
                "Bottoms": (
                    ["-"] + cls.CLOTHING_BOTTOM,
                    {"default": "-", "tooltip": "Pick bottoms; used if no Outfit is chosen."},
                ),
                "General Style": (
                    ["-"] + cls.GENERAL_STYLES,
                    {"default": "-", "tooltip": "Overall clothing style/aesthetic."},
                ),
                "Headwear": (
                    ["-"] + cls.HEADWEAR,
                    {"default": "-", "tooltip": "Specific headwear item to include."},
                ),
                "Hair Colors": (
                    ["-"] + cls.PERSON_DESC["hair_color"],
                    {"default": "-", "tooltip": "Force a specific hair color or leave '-' for random."},
                ),
                "Safe Adult Subject": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use woman/man instead of girl/boy tags.",
                    },
                ),
                # Behavior
                "Use Chain Insert": ("BOOLEAN", {"default": True}),
                "Strict Tags (no phrases)": ("BOOLEAN", {"default": True}),
                "De-duplicate With Prefix/Suffix": ("BOOLEAN", {"default": True}),
                "Danbooru Tag Style": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Format output to lowercase_underscore Danbooru tags.",
                    },
                ),
                # Token budget (maps to a soft character cap used by TIPO)
                "Token Count": (
                    ["-", "77", "150", "250", "300", "500"],
                    {"default": "77", "tooltip": "Approx. token budget for the prompt"},
                ),
                # TIPO optimization knobs (optional)
                "Enable TIPO Optimization": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Generate prompt candidates and pick the best for Illustrious.",
                    },
                ),
                "TIPO Candidates": (
                    "INT",
                    {
                        "default": 8,
                        "min": 3,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Number of variants to consider.",
                    },
                ),
                "TIPO Flavor": (
                    ["balanced", "vibrant", "soft", "natural"],
                    {
                        "default": "balanced",
                        "tooltip": "Illustrious-oriented flavor to bias scoring.",
                    },
                ),
                "TIPO Max Length": (
                    "INT",
                    {
                        "default": 320,
                        "min": 80,
                        "max": 800,
                        "step": 10,
                        "tooltip": "Soft cap for final prompt length (characters).",
                    },
                ),
                "TIPO Seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Randomization seed (0 = auto).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    # --- helpers ---
    def _pick(self, pool: List[str]) -> str:
        return random.choice(pool) if pool else ""

    def _join_commas(self, parts: List[str]) -> str:
        parts = [p.strip().strip(",") for p in parts if p and p.strip()]
        s = ", ".join(parts)
        s = re.sub(r"\s*,\s*,\s*", ", ", s).strip(", ").strip()
        return s

    def _format_danbooru_tags(self, text: str) -> str:
        """Convert phrases to danbooru-style tags: lowercase, underscores, minimal punctuation, de-dupe."""
        if not text:
            return text
        # split on commas, normalize each
        parts = [p.strip() for p in text.split(",") if p and p.strip()]
        norm = []
        seen = set()
        for p in parts:
            # strip weighting like (tag:1.2) or (tag)
            p = re.sub(r"[()\[\]{}]", "", p)
            p = re.sub(r"\b:\s*\d+(?:\.\d+)?", "", p)
            # spaces and hyphens to underscores, collapse repeats
            p = p.lower().strip()
            p = re.sub(r"[\s\-]+", "_", p)
            p = re.sub(r"_{2,}", "_", p)
            p = p.strip("_").strip()
            if p and p not in seen:
                seen.add(p)
                norm.append(p)
        return ", ".join(norm)

    def _enforce_hard_cap(self, text: str, max_chars: int, strict: bool) -> str:
        """Hard-truncate to max_chars. For strict tag mode, trim to last full token (comma boundary)."""
        if not text or not max_chars or max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        if strict:
            # Prefer the last comma to avoid cutting a tag midway
            last = cut.rfind(",")
            if last > 0:
                return cut[:last].rstrip(" ,")
            return cut.rstrip(" ,")
        # Sentence mode: prefer period, else space
        last_period = cut.rfind(".")
        if last_period > 0:
            return cut[:last_period + 1].strip()
        last_space = cut.rfind(" ")
        if last_space > 0:
            return cut[:last_space].strip()
        return cut.strip()

    # --- sentence helpers for non-strict mode ---
    def _a_an(self, phrase: str) -> str:
        w = (phrase or "").strip()
        if not w:
            return w
        article = "an" if re.match(r"^[aeiou]", w, re.I) else "a"
        return f"{article} {w}"

    def _assemble_sentence(
        self,
        subject_token: str,
        person_bits: List[str],
        pose: str,
        env: str,
        time_val: str,
        mood_val: str,
        prop_val: str,
        evt: str,
        density_val: str,
        clothes_bits: List[str],
    ) -> str:
        # subject
        subject = subject_token or "character"
        # Build descriptors from person bits excluding the age/subject word
        descriptors = [b for b in (person_bits or []) if b and b.lower() not in {"woman", "man", "girl", "boy"}]
        desc_phrase = ", ".join(descriptors) if descriptors else ""

        # Pose
        pose_phrase = f"is {pose}" if pose else "is present"

        # Clothes
        clothes_phrase = ""
        if clothes_bits:
            clothes_phrase = "wearing " + ", ".join(clothes_bits)

        # Location/time/mood
        parts = []
        lead = f"A {subject}"
        if desc_phrase:
            lead += f" with {desc_phrase}"
        parts.append(lead)
        parts.append(pose_phrase)
        if env:
            parts.append(f"in the {env}")
        if time_val:
            parts.append(f"during {time_val}")
        if mood_val:
            parts.append(f"with a {mood_val} atmosphere")
        if clothes_phrase:
            parts.append(clothes_phrase)
        if prop_val:
            parts.append(f"with {prop_val}")
        if evt:
            parts.append(f"while {evt}")
        if density_val and density_val not in ("solo", "alone"):
            parts.append(f"amid {density_val}")

        sentence = ", ".join(p for p in parts if p)
        sentence = sentence.strip()
        if sentence and not sentence.endswith("."):
            sentence += "."
        return sentence

    # ---- TIPO internals (text-only, Illustrious-flavored) ----
    def _tipo_split(self, text: str, strict: bool) -> List[str]:
        parts = [t.strip() for t in (text or "").split(",")]
        parts = [p for p in parts if p]
        if strict:
            tokens = []
            for p in parts:
                tokens.extend([t for t in re.split(r"[;|/]+", p) if t.strip()])
            parts = [t.strip() for t in tokens if t.strip()]
        return parts

    def _tipo_dedupe(self, tokens: List[str]) -> List[str]:
        seen, out = set(), []
        for t in tokens:
            key = re.sub(r"\s+", " ", t.lower())
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _tipo_perturb(
        self,
        rng: random.Random,
        tokens: List[str],
        flavor: str,
        flavor_boosts: Dict[str, List[str]],
        include_time_weather: bool,
        include_ambience: bool,
        banned: set,
        strict: bool,
    ) -> List[str]:
        cand = tokens[:]
        if cand:
            head = cand[0:1]
            tail = cand[1:]
            rng.shuffle(tail)
            cand = head + tail

        for _ in range(min(3, max(1, len(cand) // 6))):
            i = rng.randrange(0, len(cand)) if cand else 0
            j = rng.randrange(0, len(cand)) if cand else 0
            if i < len(cand) and j < len(cand):
                cand[i], cand[j] = cand[j], cand[i]

        boosts = flavor_boosts.get(flavor, [])
        for b in boosts:
            if b not in cand and random.random() < 0.6:
                cand.append(b)

        if include_time_weather and not any(
            ("sunset" in t or "night" in t or "morning" in t or "weather" in t)
            for t in cand
        ):
            if rng.random() < 0.5:
                cand.append(
                    rng.choice(["sunset", "golden hour", "overcast", "night city lights"])
                )
        if include_ambience and not any(
            ("ambient" in t or "mood" in t or "atmosphere" in t) for t in cand
        ):
            if rng.random() < 0.5:
                cand.append(
                    rng.choice(["cinematic atmosphere", "serene mood", "dramatic atmosphere"])
                )

        def emphasize(t: str) -> str:
            if not strict:
                return t
            if "(" in t or ")" in t:
                return t
            if len(t) <= 40:
                return f"({t}:1.1)"
            return t

        for idx in range(min(2, len(cand))):
            if rng.random() < 0.6:
                cand[idx] = emphasize(cand[idx])

        cand = [t for t in cand if t.strip().lower() not in banned]
        return self._tipo_dedupe(cand)

    def _tipo_score(
        self,
        tokens: List[str],
        flavor: str,
        category: str,
        strict_tags: bool,
        max_len: int,
        flavor_boosts: Dict[str, List[str]],
        banned: set,
    ) -> Tuple[float, Dict[str, int]]:
        text = ", ".join(tokens)
        score = 0.0
        why: Dict[str, int] = {}

        boosts = flavor_boosts.get(flavor, [])
        present = sum(1 for b in boosts if any(b.lower() in t.lower() for t in tokens))
        score += present * 1.2
        why["flavor_hits"] = present

        cat_hits = sum(1 for t in tokens if category.lower() in t.lower())
        score += cat_hits * 0.5
        why["category_hits"] = cat_hits

        uniq = len(set(t.lower() for t in tokens))
        dup_penalty = max(0, len(tokens) - uniq) * 0.6
        score -= dup_penalty
        why["dup_penalty"] = int(round(dup_penalty))

        length_pen = max(0, len(text) - max_len) / 40.0
        score -= length_pen
        why["length_penalty"] = int(round(length_pen))

        ban_pen = sum(1 for t in tokens if t.strip().lower() in banned) * 0.8
        score -= ban_pen
        why["banned_penalty"] = int(round(ban_pen))

        ambience_hits = sum(
            1 for t in tokens if any(k in t.lower() for k in ["atmosphere", "ambient", "mood"])
        )
        tw_hits = sum(
            1
            for t in tokens
            if t.lower() in [
                "sunset",
                "golden hour",
                "overcast",
                "night city lights",
                "night",
                "morning",
            ]
        )
        score += min(1, ambience_hits) * 0.4 + min(1, tw_hits) * 0.4
        why["context_bonus"] = min(1, ambience_hits) + min(1, tw_hits)

        il_bonus = (
            sum(
                1
                for k in ["clean lineart", "balanced colors", "natural color grading"]
                if any(k in t.lower() for t in tokens)
            )
            * 0.5
        )
        score += il_bonus
        why["illustrious_bonus"] = int(round(il_bonus))

        return score, why

    def _optimize_prompt_tipo(
        self,
        prompt: str,
        category: str,
        flavor: str,
        k: int,
        strict_tags: bool,
        max_len: int,
        seed: int,
        include_time_weather: bool,
        include_ambience: bool,
        extra_negatives: str,
    ) -> str:
        rng = random.Random(seed)
        base_tokens = self._tipo_split(prompt, strict=strict_tags)
        base_tokens = self._tipo_dedupe(base_tokens)

        flavor_boosts = {
            "balanced": ["balanced colors", "clean lineart", "consistent shading"],
            "vibrant": ["vibrant colors", "high saturation control", "dynamic lighting"],
            "soft": ["soft lighting", "pastel tones", "gentle shading"],
            "natural": ["natural color grading", "realistic lighting", "neutral tones"],
        }
        banned = set(t.strip().lower() for t in self._tipo_split(extra_negatives, strict=False))

        candidates = []
        for _ in range(max(3, k)):
            cand = self._tipo_perturb(
                rng,
                base_tokens,
                flavor,
                flavor_boosts,
                include_time_weather,
                include_ambience,
                banned,
                strict_tags,
            )
            cand_s = ", ".join(cand)
            score, _ = self._tipo_score(
                cand, flavor, category, strict_tags, max_len, flavor_boosts, banned
            )
            candidates.append((score, cand_s))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_prompt = candidates[0]
        if len(best_prompt) > max_len:
            best_prompt = best_prompt[:max_len].rstrip(", ")
        return best_prompt

    def _dedupe_against(self, text: str, tokens: List[str]) -> List[str]:
        if not text:
            return tokens
        lower = text.lower()
        out = []
        for t in tokens:
            if t and t.lower() not in lower:
                out.append(t)
        return out

    def _person_block(self, safe_adult: bool, strict: bool, hair_color_sel: str = None) -> List[str]:
        bits = []
        # age/subject word
        if safe_adult:
            bits.append(self._pick(self.PERSON_DESC["age_safe"]))
        # hair + eyes
        if hair_color_sel and hair_color_sel != "-":
            bits.append(hair_color_sel)
        else:
            bits.append(self._pick(self.PERSON_DESC["hair_color"]))
        bits.append(self._pick(self.PERSON_DESC["hair_len"]))
        bits.append(self._pick(self.PERSON_DESC["eyes"]))
        # one soft descriptor
        bits.append(self._pick(self.PERSON_DESC["extras"]))
        if not strict:
            # tiny phrase that still reads like a tag
            # (keep it minimal to stay Illustrious-friendly)
            pass
        return [b for b in bits if b]

    def _clothes_block(self, outfits: str, top: str, bottoms: str, style: str, headwear: str) -> List[str]:
        """Prefer explicit selections; otherwise fall back to random picks."""
        picks = []
        if outfits and outfits != "-":
            picks.append(outfits)
        else:
            if top and top != "-":
                picks.append(top)
            if bottoms and bottoms != "-":
                picks.append(bottoms)
            if not picks:  # fallback randoms
                # either an outfit, or top+bottoms
                if random.random() < 0.45:
                    picks.append(self._pick(self.CLOTHING_OUTFIT))
                else:
                    picks.append(self._pick(self.CLOTHING_TOP))
                    picks.append(self._pick(self.CLOTHING_BOTTOM))
        # Optional style aesthetic
        if style and style != "-":
            picks.append(style)
        # Accessories sometimes
        if headwear and headwear != "-":
            picks.append(headwear)
        else:
            if random.random() < 0.5:
                # Prefer explicit headwear pool; fall back to extras if empty
                pool = self.HEADWEAR if self.HEADWEAR else self.CLOTHING_EXTRAS
                picks.append(self._pick(pool))
        return [p for p in picks if p]

    def _pose_block(self) -> List[str]:
        return [self._pick(self.POSES)]

    def construct(self, **kwargs) -> Tuple[str]:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()
        cat = kwargs.get("Category", "Outdoor")
        complexity = kwargs.get("Complexity", "medium")
        add_time = kwargs.get("Include Time/Weather", True)
        add_mood = kwargs.get("Include Ambience", True)
        add_event = kwargs.get("Include Event", False)
        add_prop = kwargs.get("Include Prop", True)
        add_density = kwargs.get("Include Density", False)

        add_person = kwargs.get("Include Person Description", False)
        add_pose = kwargs.get("Include Pose/Action", True)
        add_clothes = kwargs.get("Include Clothing", False)

        outfits = kwargs.get("Outfits", "-")
        top = kwargs.get("Top", "-")
        bottoms = kwargs.get("Bottoms", "-")
        style = kwargs.get("General Style", "-")
        headwear = kwargs.get("Headwear", "-")
        hair_color_sel = kwargs.get("Hair Colors", "-")

        safe_adult = kwargs.get("Safe Adult Subject", True)
        use_chain = kwargs.get("Use Chain Insert", True)
        strict = kwargs.get("Strict Tags (no phrases)", True)
        dedupe = kwargs.get("De-duplicate With Prefix/Suffix", True)
        danbooru = kwargs.get("Danbooru Tag Style", True)

        pools = self.CATEGORIES.get(cat, {})
        env = self._pick(pools.get("env", []))

        # Token Count to soft char cap mapping (for TIPO)
        token_count_sel = str(kwargs.get("Token Count", "77") or "-")
        token_to_char = {"77": 320, "150": 650, "250": 1100, "300": 1300, "500": 2200}
        tipo_cap = token_to_char.get(token_count_sel)

        # Determine if user explicitly chose clothing; if so, include even if toggle is off.
        explicit_clothes_selected = (
            (outfits and outfits != "-")
            or (top and top != "-")
            or (bottoms and bottoms != "-")
            or (style and style != "-")
            or (headwear and headwear != "-")
        )

        tokens: List[str] = []
        person_bits: List[str] = []
        subject_token: str = ""
        pose_val: str = ""
        clothes_bits: List[str] = []
        time_val: str = ""
        mood_val: str = ""
        prop_val: str = ""
        density_val: str = ""
        evt: str = ""

        # Person bits first (subject details), then pose, then scene nouns
        if add_person:
            person_bits = self._person_block(safe_adult, strict, hair_color_sel)
            # Extract subject token if present
            for w in ["woman", "man", "girl", "boy"]:
                if any(w == b for b in person_bits):
                    subject_token = w
                    break
            tokens += person_bits
        if add_pose:
            pose_val = self._pose_block()[0]
            tokens.append(pose_val)
        # Auto-include clothes when user makes an explicit selection, even if the toggle is off
        if add_clothes or explicit_clothes_selected:
            clothes_bits = self._clothes_block(outfits, top, bottoms, style, headwear)
            tokens += clothes_bits

        # Scene core
        tokens.append(env)
        if add_time:
            time_val = self._pick(self.TIME_WEATHER)
            tokens.append(time_val)
        if add_mood:
            mood_val = self._pick(self.AMBIENCE)
            tokens.append(mood_val)
        if add_prop:
            prop_val = self._pick(self.PROPS)
            tokens.append(prop_val)
        if add_event:
            evt = self._pick(pools.get("events", []))
            if evt:
                tokens.append(evt)
        if add_density:
            density_val = self._pick(self.DENSITY)
            tokens.append(density_val)

        # Trim by complexity (ensure env stays)
        keep = {"simple": 4, "medium": 6, "detailed": 8}.get(complexity, 6)
        tokens = [t for t in tokens if t]
        if env in tokens:
            # keep env, constrain others
            base = [env]
            rest = [t for t in tokens if t != env]
            # If the user explicitly chose clothing, prioritize those tokens so they survive trimming
            if explicit_clothes_selected:
                explicit_tokens = []
                if outfits and outfits != "-":
                    explicit_tokens.append(outfits)
                else:
                    if top and top != "-":
                        explicit_tokens.append(top)
                    if bottoms and bottoms != "-":
                        explicit_tokens.append(bottoms)
                # Also preserve style and headwear
                if style and style != "-":
                    explicit_tokens.append(style)
                if headwear and headwear != "-":
                    explicit_tokens.append(headwear)
                # Preserve order: first any explicit clothing tokens (in their original order), then the rest
                prioritized = [t for t in rest if t in explicit_tokens]
                others = [t for t in rest if t not in explicit_tokens]
                rest = prioritized + others
            tokens = base + rest[: max(0, keep - 1)]
        else:
            tokens = tokens[:keep]

        # Dedupe against prefix/suffix to avoid overlaps when chained
        if dedupe:
            tokens = self._dedupe_against(prefix, tokens)
            tokens = self._dedupe_against(suffix, tokens)

        scene_str = self._join_commas(tokens)

        if use_chain and CHAIN_INSERT_TOKEN in prefix:
            head, tail = prefix.split(CHAIN_INSERT_TOKEN, 1)
            out = self._join_commas([head, scene_str, tail, suffix])
        else:
            out = self._join_commas([prefix, scene_str, suffix])

        # Optional TIPO optimization (only in strict/tag mode)
        if strict and kwargs.get("Enable TIPO Optimization", True):
            out = self._optimize_prompt_tipo(
                prompt=out,
                category=kwargs.get("Category", "Outdoor"),
                flavor=kwargs.get("TIPO Flavor", "balanced"),
                k=int(kwargs.get("TIPO Candidates", 8)),
                strict_tags=True,
                max_len=(int(tipo_cap) if tipo_cap else int(kwargs.get("TIPO Max Length", 320))),
                seed=int(kwargs.get("TIPO Seed", 0)) or (int(time.time()) & 0x7FFFFFFF),
                include_time_weather=kwargs.get("Include Time/Weather", True),
                include_ambience=kwargs.get("Include Ambience", True),
                extra_negatives=kwargs.get("tipo_extra_negatives", "") or "",
            )

        # Sentence mode when not strict: convert to readable sentence
        if not strict:
            sentence = self._assemble_sentence(
                subject_token=subject_token,
                person_bits=person_bits,
                pose=pose_val,
                env=env,
                time_val=time_val,
                mood_val=mood_val,
                prop_val=prop_val,
                evt=evt,
                density_val=density_val,
                clothes_bits=clothes_bits,
            )
            if use_chain and CHAIN_INSERT_TOKEN in prefix:
                head, tail = prefix.split(CHAIN_INSERT_TOKEN, 1)
                out = ", ".join([head.strip(), sentence.strip(), tail.strip(), suffix.strip()]).strip(", ")
            else:
                out = ", ".join([prefix.strip(), sentence.strip(), suffix.strip()]).strip(", ")

        if danbooru and strict:
            out = self._format_danbooru_tags(out)

        # Enforce absolute cap if Token Count is set
        if tipo_cap:
            out = self._enforce_hard_cap(out, int(tipo_cap), strict)

        return (out if out else " ",)
