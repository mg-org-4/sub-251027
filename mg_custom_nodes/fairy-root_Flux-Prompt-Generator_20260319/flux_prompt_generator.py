# flux_prompt_generator.py
import subprocess
import sys
import random
import json
import os
import re

# --- Installation and JSON Loading (Keep as is) ---
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

def load_json_file(file_name):
    # Add basic error handling for file not found
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    if not os.path.exists(file_path):
        print(f"Warning: JSON file not found at {file_path}. Returning empty list.")
        return []
    try:
        with open(file_path, "r", encoding='utf-8') as file: # Added encoding
            data = json.load(file)

        if isinstance(data, list):
            # Simpler duplicate removal for lists of strings/simple objects
            # If items are dicts, the original method is better, but assumes hashable items after json.dumps
            try:
                data = list(dict.fromkeys(data)) # Faster for simple types
            except TypeError: # Fallback for unhashable types like dicts if json was complex
                 data = list({json.dumps(item, sort_keys=True) for item in data})
                 data = [json.loads(item) for item in data]
        elif isinstance(data, dict):
             # For now, assume lists are expected based on the sample.
             print(f"Warning: Expected a list in {file_name}, but got dict. Using keys or values might be needed.")
             # Example: return list(data.keys()) or list(data.values())
             return [] # Return empty list if structure is unexpected

        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Returning empty list.")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Returning empty list.")
        return []


# --- Load Data (Keep as is, but consider adding error checks) ---
ARTFORM = load_json_file("artform.json")
ACCESSORIES = load_json_file("accessories.json")
ADDITIONAL_DETAILS = load_json_file("additional_details.json")
AGE_GROUP = load_json_file("age_group.json")
ARTIST = load_json_file("artist.json")
BACKGROUND = load_json_file("background.json")
BODY_MARKINGS = load_json_file("body_markings.json")
BODY_TYPES = load_json_file("body_types.json")
CLOTHING = load_json_file("clothing.json")
COMPOSITION = load_json_file("composition.json")
DEFAULT_TAGS = load_json_file("default_tags.json")
DEVICE = load_json_file("device.json")
DIGITAL_ARTFORM = load_json_file("digital_artform.json")
ETHNICITY = load_json_file("ethnicity.json")
EXPRESSION = load_json_file("expression.json")
EYE_COLORS = load_json_file("eye_colors.json")
FACE_FEATURES = load_json_file("face_features.json")
FACIAL_HAIR = load_json_file("facial_hair.json")
HAIR_COLOR = load_json_file("hair_color.json")
HAIRSTYLES = load_json_file("hairstyles.json")
LIGHTING = load_json_file("lighting.json")
MAKEUP_STYLES = load_json_file("makeup_styles.json")
PHOTO_TYPE = load_json_file("photo_type.json")
PHOTOGRAPHER = load_json_file("photographer.json")
PHOTOGRAPHY_STYLES = load_json_file("photography_styles.json")
PLACE = load_json_file("place.json")
POSE = load_json_file("pose.json")
ROLES = load_json_file("roles.json")
SKIN_TONE = load_json_file("skin_tone.json")
TATTOOS_SCARS = load_json_file("tattoos_scars.json")


# --- Helper Function for Cleaner Joining ---
def smart_join(elements, separator=", "):
    """Joins non-empty elements with a separator."""
    return separator.join(filter(None, elements))

# --- PromptGenerator Class (Refactored) ---
class PromptGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def _get_choice(self, input_value, default_choices):
        """Internal helper to get a single choice, handling random/disabled."""
        if not default_choices: # Handle empty JSON lists
            return input_value if input_value.lower() not in ["random", "disabled"] else ""

        input_lower = input_value.lower()
        if input_lower == "disabled":
            return ""
        elif "," in input_value: # Allow comma-separated specific choices from user
            choices = [choice.strip() for choice in input_value.split(",")]
            # Return one random choice from the user's list
            return self.rng.choice(choices) if choices else ""
        elif input_lower == "random":
            return self.rng.choice(default_choices)
        else:
            # Return the specific value provided by the user
            return input_value

    def _get_multiple_choices(self, input_value, default_choices, min_count=1, max_count=1):
        """Helper to get multiple choices, useful for things like lighting."""
        if not default_choices:
            return ""

        input_lower = input_value.lower()
        if input_lower == "disabled":
            return ""
        elif "," in input_value: # User provided specific items
             # Use user's specific items directly, joined
             return ", ".join(filter(None, [choice.strip() for choice in input_value.split(",")]))
        elif input_lower == "random":
            count = self.rng.randint(min_count, max_count)
            # Ensure sample size doesn't exceed population size
            actual_count = min(count, len(default_choices))
            if actual_count == 0: return "" # Handle case where default_choices is empty
            chosen = self.rng.sample(default_choices, actual_count)
            return ", ".join(chosen)
        else:
            # Specific single value chosen
            return input_value

    def clean_prompt_string(self, text):
        """Cleans up common prompt string issues."""
        if not text: return ""
        # Remove extra spaces around commas, then replace multiple commas with one
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r',+', ',', text)
        # Remove leading/trailing commas and spaces
        text = text.strip(', ')
        # Remove spaces before BREAK markers
        text = re.sub(r'\s+(BREAK_CLIP[GL])', r' \1', text)
         # Remove spaces after BREAK markers
        text = re.sub(r'(BREAK_CLIP[GL])\s+', r'\1 ', text)
        # Remove spaces before periods
        text = re.sub(r'\s+\.', '.', text)
        # Ensure space after period before word (but not multiple spaces)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        # Fix double periods
        text = re.sub(r'\.\.+', '.', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Specific cleanup like 'of a as' -> 'of a' etc.
        text = text.replace(" of as ", " of ")
        text = text.replace(" a as ", " as ")
        text = text.replace(" as as ", " as ")  # Fix "working as as a"
        text = text.replace(" with with ", " with ")  # Fix "with with"
        # Remove empty segments like ", ," or leading ", " after splits
        text = re.sub(r'^,\s*', '', text)
        text = re.sub(r'\s*,\s*$', '', text)
        text = re.sub(r',(\s*,)+', ',', text)
        return text

    def strip_weights_for_natural_language(self, text):
        """Removes parentheses and weights from text for natural language (T5-XXL)."""
        if not text: return ""
        # Remove weights like (text:1.5) -> text
        text = re.sub(r'\(([^:)]+):[0-9.]+\)', r'\1', text)
        # Remove simple parentheses (text) -> text
        text = re.sub(r'\(([^)]+)\)', r'\1', text)
        return text

    def _build_natural_language_sentence(self, category, value, context=None):
        """Converts a category value into a natural language fragment."""
        if not value:
            return ""

        templates = {
            'role': f"working as {value}",
            'hairstyle': f"with {value}",
            'clothing': f"dressed in {value}",
            'composition': f"The composition follows {value}",
            'pose': f"The subject is {value}",
            'face_features': f"with {value}",
            'eye_color': f"with {value} eyes",
            'skin_tone': f"with {value} skin",
            'age_group': f"appearing {value}",
            'ethnicity': f"of {value} descent",
            'accessories': f"wearing {value}",
            'expression': f"with a {value} expression",
            'tattoos_scars': f"featuring {value}",
            'hair_color': f"with {value} hair",
            'body_markings': f"displaying {value}",
            'facial_hair': f"with {value}",
            'makeup': f"styled with {value} makeup"
        }

        return templates.get(category, value)

    def _format_debug_info(self, debug_info):
        """Format debug info as readable string"""
        lines = []
        lines.append("=== DEBUG INFO - CATEGORY USAGE ===\n")

        # Categories Used
        lines.append(f"✅ CATEGORIES USED ({len(debug_info['used'])})")
        for category, value in sorted(debug_info['used'].items()):
            # Truncate long values
            display_value = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
            lines.append(f"  • {category:20s} = {display_value}")

        # Categories Not Used
        lines.append(f"\n❌ CATEGORIES NOT USED ({len(debug_info['not_used'])})")
        for category, reason in sorted(debug_info['not_used'].items()):
            lines.append(f"  • {category:20s} : {reason}")

        return "\n".join(lines)

    def process_string_v2(self, combined_prompt, seed, debug_output=""):
        """Uses regex to split the prompt based on BREAK markers."""
        clip_l_content = ""
        clip_g_content = ""
        t5xxl_content = combined_prompt # Start with full prompt for T5

        # Extract CLIP L content
        match_l = re.search(r'BREAK_CLIPL(.*?)BREAK_CLIPL', combined_prompt, re.DOTALL)
        if match_l:
            clip_l_content = match_l.group(1).strip()
            # Remove CLIP L block from T5
            t5xxl_content = t5xxl_content.replace(match_l.group(0), '', 1)

        # Extract CLIP G content
        match_g = re.search(r'BREAK_CLIPG(.*?)BREAK_CLIPG', combined_prompt, re.DOTALL)
        if match_g:
            clip_g_content = match_g.group(1).strip()
             # Remove CLIP G block from T5
            t5xxl_content = t5xxl_content.replace(match_g.group(0), '', 1)

        # Original prompt is T5 content with markers removed
        original_content = t5xxl_content.replace('BREAK_CLIPL', '').replace('BREAK_CLIPG', '')

        # Strip weights/parentheses from T5-XXL content (natural language doesn't need weights)
        original_content = self.strip_weights_for_natural_language(original_content)
        t5xxl_content = self.strip_weights_for_natural_language(t5xxl_content)

        # Clean all parts
        original_clean = self.clean_prompt_string(original_content)
        t5xxl_clean = self.clean_prompt_string(t5xxl_content)
        clip_l_clean = self.clean_prompt_string(clip_l_content)
        clip_g_clean = self.clean_prompt_string(clip_g_content)

        return original_clean, seed, t5xxl_clean, clip_l_clean, clip_g_clean, debug_output

    def generate_prompt(self, seed, **kwargs):
        # Use kwargs directly, simplifies passing arguments
        self.rng = random.Random(seed) # Re-seed for each generation if seed changes

        components = []

        # Debug tracking: Track which categories are used vs not used
        debug_info = {
            'used': {},      # Categories that had values and were used
            'not_used': {},  # Categories that were disabled/empty/ignored
            'all_inputs': {} # All input values for reference
        }

        # --- 1. Custom Prompt ---
        custom = kwargs.get("custom", "")
        if custom:
            components.append(custom)

        # --- 2. Artform / Style Lead-in (with photo_type integrated) ---
        artform = self._get_choice(kwargs.get("artform", "disabled"), ARTFORM)
        is_photographer = (artform.lower() == "photography")

        # Get photo_type early to integrate into opening
        photo_type = self._get_choice(kwargs.get("photo_type", "random"), PHOTO_TYPE)

        if is_photographer:
            photo_style = self._get_choice(kwargs.get("photography_styles", "random"), PHOTOGRAPHY_STYLES)
            # Build opening with optional photo_type
            opening_parts = []
            if photo_type:
                opening_parts.append(f"A {photo_type}")
            opening_parts.append(photo_style if photo_style else "photography")
            components.append(" ".join(opening_parts))

            # Add "of" if a subject or default tag will follow
            if kwargs.get("subject") or kwargs.get("default_tags", "disabled").lower() != "disabled":
                 components.append("of") # Add connector word

        elif artform and artform.lower() != "disabled":
             # Build opening with optional photo_type
             opening_parts = []
             if photo_type:
                 opening_parts.append(f"A {photo_type}")
             opening_parts.append(artform)
             components.append(" ".join(opening_parts))

             # Add "of" if a subject or default tag will follow and artform isn't inherently descriptive like 'illustration'
             if kwargs.get("subject") or kwargs.get("default_tags", "disabled").lower() != "disabled":
                 # Could refine this list if needed
                 if artform.lower() not in ["illustration", "painting", "drawing", "sketch"]:
                     components.append("of")

        elif photo_type:
            # Standalone photo_type when artform is disabled
            components.append(f"A {photo_type}")
            # Add "of" if a subject or default tag will follow
            if kwargs.get("subject") or kwargs.get("default_tags", "disabled").lower() != "disabled":
                components.append("of")

        # --- 3. Subject Definition ---
        subject = kwargs.get("subject", "")
        default_tags_input = kwargs.get("default_tags", "random")
        body_type_input = kwargs.get("body_types", "random")

        chosen_subject_elements = []
        # User provided subject takes precedence (but not if it's "random" or "disabled")
        if subject and subject.lower() not in ["random", "disabled"]:
            chosen_body_type = self._get_choice(body_type_input, BODY_TYPES)
            if chosen_body_type:
                 chosen_subject_elements.extend(["a", chosen_body_type]) # e.g., "a muscular"
            chosen_subject_elements.append(subject) # e.g., "a muscular woman"
        else: # No specific subject, use default tags
            chosen_default_tag = self._get_choice(default_tags_input, DEFAULT_TAGS)
            if chosen_default_tag: # Only proceed if default tag isn't disabled/empty
                chosen_body_type = self._get_choice(body_type_input, BODY_TYPES)
                # Check if tag starts with a/an, handle body type insertion
                starts_with_article = chosen_default_tag.lower().startswith(("a ", "an "))
                if chosen_body_type:
                    if starts_with_article:
                        # Insert body type after article: "a muscular woman"
                        parts = chosen_default_tag.split(" ", 1)
                        chosen_subject_elements.extend([parts[0], chosen_body_type, parts[1]])
                    else:
                        # Prepend "a" and body type: "a muscular subject"
                         chosen_subject_elements.extend(["a", chosen_body_type, chosen_default_tag])
                else:
                    # Just use the default tag
                     chosen_subject_elements.append(chosen_default_tag)

        if chosen_subject_elements:
            components.append(" ".join(chosen_subject_elements))

        # Store chosen tag for gender check later
        resolved_subject_desc = " ".join(chosen_subject_elements).lower()

        # --- 4. Core Details (Roles, Hairstyles, Additional Details) ---
        # Build natural language sentences instead of comma-joining
        role = self._get_choice(kwargs.get("roles", "random"), ROLES)
        hairstyle = self._get_choice(kwargs.get("hairstyles", "random"), HAIRSTYLES)
        additional_details = self._get_choice(kwargs.get("additional_details", "random"), ADDITIONAL_DETAILS)

        core_parts = []
        if role:
            core_parts.append(self._build_natural_language_sentence('role', role))
        if hairstyle:
            core_parts.append(self._build_natural_language_sentence('hairstyle', hairstyle))

        if core_parts:
            components.append(" ".join(core_parts))

        if additional_details:
            components.append(f". {additional_details}")

        # --- 5. Clothing ---
        clothing = self._get_choice(kwargs.get("clothing", "random"), CLOTHING)
        if clothing:
            components.append(f". Dressed in {clothing}")

        # --- 6. Composition & Pose ---
        # Build natural language sentences with periods
        composition = self._get_choice(kwargs.get("composition", "random"), COMPOSITION)
        pose = self._get_choice(kwargs.get("pose", "random"), POSE)

        comp_pose_parts = []
        if pose:
            comp_pose_parts.append(f"The subject is {pose}")
        if composition:
            comp_pose_parts.append(f"The composition follows {composition}")

        if comp_pose_parts:
            components.append(". " + ". ".join(comp_pose_parts))

        # --- BREAK CLIP G 1 ---
        components.append("BREAK_CLIPG")

        # --- 7. Environment (Background, Place) ---
        environment = [
             self._get_choice(kwargs.get("background", "random"), BACKGROUND),
             self._get_choice(kwargs.get("place", "random"), PLACE),
        ]
        components.append(smart_join(environment))


        # --- 8. Lighting ---
        # Use multi-choice helper for lighting if random
        lighting_input = kwargs.get("lighting", "random")
        if lighting_input.lower() == "random":
             lighting = self._get_multiple_choices(lighting_input, LIGHTING, min_count=2, max_count=4) # Example: 2-4 items
        else:
             lighting = self._get_choice(lighting_input, LIGHTING)

        if lighting:
            components.append(lighting)


        # --- BREAK CLIP G 2 ---
        components.append("BREAK_CLIPG")

        # --- 9. Physical Features ---
        # Get all feature values
        face_features = self._get_choice(kwargs.get("face_features", "random"), FACE_FEATURES)
        eye_color = self._get_choice(kwargs.get("eye_colors", "random"), EYE_COLORS)
        skin_tone = self._get_choice(kwargs.get("skin_tone", "random"), SKIN_TONE)
        age_group = self._get_choice(kwargs.get("age_group", "random"), AGE_GROUP)
        ethnicity = self._get_choice(kwargs.get("ethnicity", "random"), ETHNICITY)
        accessories = self._get_choice(kwargs.get("accessories", "random"), ACCESSORIES)
        expression = self._get_choice(kwargs.get("expression", "random"), EXPRESSION)
        tattoos_scars = self._get_choice(kwargs.get("tattoos_scars", "random"), TATTOOS_SCARS)
        hair_color = self._get_choice(kwargs.get("hair_color", "random"), HAIR_COLOR)
        body_markings = self._get_choice(kwargs.get("body_markings", "random"), BODY_MARKINGS)

        # Facial hair and makeup - now independent of gender for modern/creative contexts
        facial_hair = self._get_choice(kwargs.get("facial_hair", "random"), FACIAL_HAIR)
        makeup = self._get_choice(kwargs.get("makeup_styles", "random"), MAKEUP_STYLES)

        # Group features into coherent sentences
        feature_sentences = []

        # Sentence 1: Face, eyes, expression
        face_parts = []
        if face_features:
            face_parts.append(face_features)
        if eye_color:
            face_parts.append(f"{eye_color} eyes")
        if expression:
            face_parts.append(f"a {expression} expression")
        if face_parts:
            feature_sentences.append("They have " + " and ".join(face_parts))

        # Sentence 2: Skin, age, ethnicity
        identity_parts = []
        if skin_tone:
            identity_parts.append(f"{skin_tone} skin")
        if age_group:
            # Age groups that are nouns need articles
            age_nouns = ["adult", "child", "infant", "preteen", "senior", "teenager", "toddler", "young adult"]
            if age_group in age_nouns:
                article = "an" if age_group[0].lower() in "aeiou" else "a"
                identity_parts.append(f"appearing as {article} {age_group}")
            else:
                # Adjectives don't need articles
                identity_parts.append(f"appearing {age_group}")
        if ethnicity:
            identity_parts.append(f"of {ethnicity} descent")
        if identity_parts:
            feature_sentences.append("The subject has " + ", ".join(identity_parts))

        # Sentence 3: Hair, accessories, markings, gender-specific (only one of facial_hair OR makeup)
        detail_parts = []
        if hair_color:
            detail_parts.append(f"{hair_color} hair")
        if accessories:
            detail_parts.append(f"wearing {accessories}")
        if tattoos_scars:
            detail_parts.append(f"featuring {tattoos_scars}")
        if body_markings:
            detail_parts.append(f"displaying {body_markings}")
        if facial_hair and not makeup:  # Only add if makeup isn't present
            detail_parts.append(f"{facial_hair}")
        elif makeup:  # Only add if facial_hair isn't present
            detail_parts.append(f"{makeup}")
        if detail_parts:
            feature_sentences.append("Notable features include " + ", ".join(detail_parts))

        if feature_sentences:
            components.append(". " + ". ".join(feature_sentences))

        # --- 10. Camera/Device (for T5-XXL natural language) ---
        device = self._get_choice(kwargs.get("device", "random"), DEVICE)
        if device:
            components.append(f". The image was captured using a {device}")

        # --- BREAK CLIP L 1 ---
        components.append("BREAK_CLIPL")

        # --- 11. Technical/Artistic Details (for CLIP_L keywords) ---
        # All categories now work independently - users control via their selections
        tech_artist_details = []

        # Framing (weighted) - reuse the same value fetched for T5-XXL opening
        # This ensures consistency between T5-XXL and CLIP_L
        if photo_type:
            random_value = round(self.rng.uniform(1.1, 1.5), 1)
            tech_artist_details.append(f"({photo_type}:{random_value})")

        # Device - already fetched above for T5-XXL, reuse the same value
        if device:
            tech_artist_details.append(f"shot on {device}")

        # Digital artform - works independently
        digital_artform = self._get_choice(kwargs.get("digital_artform", "random"), DIGITAL_ARTFORM)
        if digital_artform:
            tech_artist_details.append(digital_artform)

        # Photographer - works independently
        photographer = self._get_choice(kwargs.get("photographer", "random"), PHOTOGRAPHER)
        if photographer:
            tech_artist_details.append(f"photo by {photographer}")

        # Artist - works independently
        artist = self._get_choice(kwargs.get("artist", "random"), ARTIST)
        if artist:
            tech_artist_details.append(f"by {artist}")

        components.append(smart_join(tech_artist_details))

        # --- BREAK CLIP L 2 ---
        components.append("BREAK_CLIPL")

        # --- Final Assembly & Processing ---
        # Join all collected components, using smart_join to handle potential empty strings between sections
        full_prompt_string = smart_join(components, separator=" ")

        # --- Debug Info: Track all category usage ---
        all_categories = [
            'custom', 'subject', 'default_tags', 'body_types', 'artform', 'photography_styles',
            'digital_artform', 'artist', 'photographer', 'roles', 'hairstyles', 'hair_color',
            'additional_details', 'lighting', 'clothing', 'composition', 'pose', 'background',
            'place', 'age_group', 'ethnicity', 'accessories', 'expression', 'face_features',
            'eye_colors', 'skin_tone', 'facial_hair', 'body_markings', 'makeup_styles',
            'tattoos_scars', 'photo_type', 'device'
        ]

        for category in all_categories:
            input_value = kwargs.get(category, "disabled")
            debug_info['all_inputs'][category] = input_value

            # Determine if category was actually used and provide specific reason if not
            if not input_value or input_value == "":
                debug_info['not_used'][category] = "empty (not provided)"
            elif input_value.lower() == "disabled":
                debug_info['not_used'][category] = "disabled (explicitly set)"
            elif input_value.lower() == "random":
                # Random means it was used (a value was selected)
                debug_info['used'][category] = "random (value selected)"
            else:
                # Specific value was used
                debug_info['used'][category] = input_value

        # Special cases: Check for conditional logic
        # default_tags is ignored when subject is provided
        if kwargs.get('subject', '') and kwargs.get('subject', '').lower() not in ['random', 'disabled']:
            if 'default_tags' in debug_info['used']:
                debug_info['not_used']['default_tags'] = "ignored (subject takes precedence)"
                del debug_info['used']['default_tags']

        # photography_styles only used when artform is photography
        if kwargs.get('photography_styles', '') and kwargs.get('photography_styles', '').lower() not in ['disabled', '']:
            if artform.lower() != "photography":
                if 'photography_styles' in debug_info['used']:
                    debug_info['not_used']['photography_styles'] = f"ignored (artform={artform}, not photography)"
                    del debug_info['used']['photography_styles']

        # Format debug output as readable string
        debug_output = self._format_debug_info(debug_info)

        # Process using the V2 splitter
        return self.process_string_v2(full_prompt_string, seed, debug_output)


# --- ComfyUI Node Class (Updated RETURN_TYPES) ---
class FluxPromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": random.randint(0, 30000), "min": 0, "max": 30000, "step": 1}),
                "custom": ("STRING", {"multiline": True, "default": ""}),
                "subject": ("STRING", {"multiline": True, "default": ""}),
                "accessories": (["disabled", "random"] + ACCESSORIES, {"default": "disabled"}),
                "additional_details": (["disabled", "random"] + ADDITIONAL_DETAILS, {"default": "disabled"}),
                "age_group": (["disabled", "random"] + AGE_GROUP, {"default": "disabled"}),
                "artform": (["disabled", "random"] + ARTFORM, {"default": "disabled"}),
                "artist": (["disabled", "random"] + ARTIST, {"default": "disabled"}),
                "background": (["disabled", "random"] + BACKGROUND, {"default": "disabled"}),
                "body_markings": (["disabled", "random"] + BODY_MARKINGS, {"default": "disabled"}),
                "body_types": (["disabled", "random"] + BODY_TYPES, {"default": "disabled"}),
                "clothing": (["disabled", "random"] + CLOTHING, {"default": "disabled"}),
                "composition": (["disabled", "random"] + COMPOSITION, {"default": "disabled"}),
                "default_tags": (["disabled", "random"] + DEFAULT_TAGS, {"default": "disabled"}),
                "device": (["disabled", "random"] + DEVICE, {"default": "disabled"}),
                "digital_artform": (["disabled", "random"] + DIGITAL_ARTFORM, {"default": "disabled"}),
                "ethnicity": (["disabled", "random"] + ETHNICITY, {"default": "disabled"}),
                "expression": (["disabled", "random"] + EXPRESSION, {"default": "disabled"}),
                "eye_colors": (["disabled", "random"] + EYE_COLORS, {"default": "disabled"}),
                "face_features": (["disabled", "random"] + FACE_FEATURES, {"default": "disabled"}),
                "facial_hair": (["disabled", "random"] + FACIAL_HAIR, {"default": "disabled"}),
                "hair_color": (["disabled", "random"] + HAIR_COLOR, {"default": "disabled"}),
                "hairstyles": (["disabled", "random"] + HAIRSTYLES, {"default": "disabled"}),
                "lighting": (["disabled", "random"] + LIGHTING, {"default": "disabled"}),
                "makeup_styles": (["disabled", "random"] + MAKEUP_STYLES, {"default": "disabled"}),
                "photographer": (["disabled", "random"] + PHOTOGRAPHER, {"default": "disabled"}),
                "photography_styles": (["disabled", "random"] + PHOTOGRAPHY_STYLES, {"default": "disabled"}),
                "photo_type": (["disabled", "random"] + PHOTO_TYPE, {"default": "disabled"}),
                "place": (["disabled", "random"] + PLACE, {"default": "disabled"}),
                "pose": (["disabled", "random"] + POSE, {"default": "disabled"}),
                "roles": (["disabled", "random"] + ROLES, {"default": "disabled"}),
                "skin_tone": (["disabled", "random"] + SKIN_TONE, {"default": "disabled"}),
                "tattoos_scars": (["disabled", "random"] + TATTOOS_SCARS, {"default": "disabled"}),
            }
        }

    # Correct RETURN_TYPES and add RETURN_NAMES
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "t5xxl", "clip_l", "clip_g", "seed", "debug_info")

    FUNCTION = "execute"
    CATEGORY = "Prompt"

    def execute(self, **kwargs):
        # Pass all arguments using kwargs
        seed = kwargs.get('seed', 0) # Extract seed separately if needed elsewhere
        prompt_generator = PromptGenerator(seed)
        prompt = prompt_generator.generate_prompt(**kwargs)
        # Unpack the 6-tuple and return all 6 outputs including seed_used
        original_clean, seed_used, t5xxl_clean, clip_l_clean, clip_g_clean, debug_output = prompt
        # Concatenate all sections into a single combined prompt
        combined_prompt = ", ".join(filter(None, [t5xxl_clean, clip_l_clean, clip_g_clean]))
        return (combined_prompt, t5xxl_clean, clip_l_clean, clip_g_clean, str(seed_used), debug_output)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Trigger refresh based on seed or any other input change
        # Simple approach: always return True if any input changes (including seed)
        # You could implement more complex logic comparing old/new kwargs if needed
        return float('nan') # Standard ComfyUI way to indicate always refresh


# Node export details
NODE_CLASS_MAPPINGS = {"FluxPromptGenerator": FluxPromptGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxPromptGenerator": "Flux Prompt Generator"}