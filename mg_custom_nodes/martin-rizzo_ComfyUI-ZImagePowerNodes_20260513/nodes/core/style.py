"""
File    : style.py
Purpose : Core classes for working with visual styles.
           `Style`   : A class for storing information about a visual style.
           `StyleSet`: A container for storing multiple styles.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 30, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from __future__      import annotations
from typing          import Iterator
from collections.abc import KeysView, ValuesView
import re
import unicodedata

# Pattern used to remove non-alphanumeric characters.
_CLEAN_PATTERN = re.compile(r"[^a-z0-9_]")

# For undefined order, we use a tuple with two very high values.
_UNDEFINED_ORDER = (999_999, 999_999)


#=============================== Style CLASS ===============================#

class Style:
    """Represents a visual style with its name, template, description and tags.

    Attributes:
        name        : The name of the style.
        template    : A string with the template to be injected into the prompt.
        description : A string with the description for the style.
        tags        : A tuple with string tags associated with the style.
        category    : A string with the category name to which this style belongs.
        version     : A string indicating the version to which this style belongs, e.g. "2.1.0"
        order       : An optional tuple used to determine the order of the styles;
                      This attribute is used internally by the `StyleSet` class
                      when sorting the styles and should not be set by the user

    Properties:
        comma_separated_tags: A string containing the style's tags separated by commas.
        quoted_name         : A string containing the style's name but surrounded by double quotes.
        slug                : A string with the slug version of the style's name for URL use.

    Example:
        >>> style = Style(
        ...     "cinematic",
        ...     template    = "{$@}, rendered in epic cinematic lighting",
        ...     description = "Ultra-realistic blockbuster look",
        ...     tags        = ("photoreal", "movie", "8k"),
        ...     category    = "photo",
        ...     version     = "1.0.0"
        ... )
        >>> style.comma_separated_tags
        'photoreal, movie, 8k'
    """
    def __init__(self,
                 name       : str,
                 *,
                 template   : str,
                 description: str                        = "",
                 tags       : list[str] | tuple[str,...] = tuple(),
                 category   : str                        = "",
                 version    : str | tuple[int,...]       = "0.0.0",
                 order      : tuple[int, int]            = _UNDEFINED_ORDER,
                 ):
        self.name       : str                  = name
        self.template   : str                  = template
        self.description: str                  = description
        self.tags       : tuple[str, ...]      = tuple(t for t in tags if isinstance(t, str) and len(t) > 0)
        self.category   : str                  = category
        self.version    : tuple[int, int, int] = self.normalize_version(version)
        self.order      : tuple[int, int]      = order



    @property
    def comma_separated_tags(self) -> str:
        """A string containing the style's tags separated by commas."""
        return ", ".join(self.tags)


    @property
    def quoted_name(self) -> str:
        """A string containing the style's name but surrounded by double quotes."""
        return f'"{self.name}"'


    @property
    def slug(self) -> str:
        """A string with the slug version of the style's name for URL use."""
        return self.canonicalize_name(self.name)


    @staticmethod
    def canonicalize_name(name: str) -> str:
        """
        Normalize te style name

        The transformation pipeline is:
          1. Return empty string immediately if the name is empty.
          2. Unicode NFD normalization removing diacritics such as accents, tildes, etc.
          3. Lower-case, strip surrounding whitespace.
          4. Replace spaces and hyphens with underscores.
          5. Remove quotation (`'`, `"`)
          6. Guard against null cases ("-", "none", empty string).
          7. Replace any remaining non-alphanumeric character with "x".

        Args:
            name: The style name to canonicalize.

        Returns:
            The canonicalized style name, that contains only lowercase ASCII
            letters, digits, and underscores; suitable for use in URLs, ids, etc;
            or empty string if the input name is invalid or empty.

        Examples:
            >>> Style.canonicalize_name("Cinematic Look")
            'cinematic_look'
            >>> Style.canonicalize_name("8k PHOTO STYLE")
            '8k_photo_style'
            >>> Style.canonicalize_name("Fotográfico")
            'fotografico'
            >>> Style.canonicalize_name("")
            ''
            >>> Style.canonicalize_name("None")
            ''
        """
        if not isinstance(name, str) or not name:
            return ""

        # step 1: unicode normalization
        # strip the combining marks to remove accents, tildes, etc.
        name = unicodedata.normalize("NFD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))

        # step 2: case normalization, whitespace cleanup,
        # map spaces and hyphens to underscores, then drop quote characters
        name = name.lower().strip().replace(" ", "_").replace("-", "_")
        name = re.sub(r"['\"`]", "", name)

        # step 3: Reject invalid or null names.
        if name in ("", "-", "none", "null"):
            return ""

        # step 4: any character not in [a‑z 0‑9 _] becomes "x".
        return _CLEAN_PATTERN.sub("x", name)


    @staticmethod
    def normalize_version(version: str | tuple[int,...]) -> tuple[int, int, int]:
        """
        Parse a version string or tuple into a tuple of 3 integers.

        This method is used internally by Style to ensure that the version
        number is in a consistent format. Any external code should use this
        method to ensure matching versions with the `style.version` attribute.

        Handles version numbers with optional 'v' prefix and 1, 2 or 3 elements
        separated by periods, missing elements are filled with zeros.

        Args:
            version: A version string like "3.1", "4.2.1", "v2.3", "v2.3.1"
                     or a tuple of 3 integers.
        Returns:
            A tuple of 3 integers (major, minor, patch).

        Examples:
            >>> _parse_version("3.1")
            (3, 1, 0)
            >>> _parse_version("v2.3.1")
            (2, 3, 1)
            >>> _parse_version((1, 2, 3))
            (1, 2, 3)
        """
        # if already a tuple, force to 3 elements
        if isinstance(version, tuple):
            v = (*version, 0, 0, 0)[:3]
        # if version is a string, parse it into a tuple of integers
        elif isinstance(version, str):
            parts = [int(p) if p.isdigit() else 0 for p in version.lstrip('vV').split('.')]
            v = (*parts, 0, 0, 0)[:3]
        else:
            raise TypeError("Version must be a string or tuple")
        return (v[0], v[1], v[2])




    def apply_to_prompt(self,
                        prompt : str,
                        *,
                        spicy_impact_booster: bool = False
                        ) -> str:
        """
        Applies the style to a prompt with optional spicy content boost.

        Args:
            prompt              : The input text prompt to be styled.
            spicy_impact_booster: If True, adds spicy content to the output. Default is False.

        Returns:
            A string containing the styled prompt.
        """
        spicy_content = ""
        if spicy_impact_booster:
            spicy_content = "attractive and spicy content, where any woman is sexy and provocative, with"

        result = self.template
        result = result.replace("{$spicy-content-with}", spicy_content) #< the secret spicy dressing
        result = result.replace("{$@}"                 , prompt       ) #< prompt to be styled
        result = result.replace("  ", " ")                              #< fix double spaces
        return result



    def __repr__(self) -> str:
        """Return an unambiguous string representation of the Style instance that can be used to recreate it."""
        return (
            f"Style({self.name!r}, "
            f"template={self.template!r}, "
            f"description={self.description!r}, "
            f"tags={self.tags!r}, "
            f"version={self.version!r})"
        )


#============================= StyleSet CLASS ==============================#

class StyleSet:
    """
    A class representing a set of visual styles.
    """

    # Order for categories, lower values come first.
    # (unlisted categories will be placed after these, order = 999)
    CATEGORY_SORT_ORDER = {
        "photo"       : 0,
        "illustration": 1,
        "wild"        : 2
    }

    def __init__(self,
                 styles: StyleSet | None = None,
                 ) -> None:
        """
        Initialize the StyleSet with an optional dictionary of styles.

        Args:
            styles: An optional StyleSet instance to copy from,
                    or None to start with an empty set.
        """
        self._styles_by_canonical: dict[str, Style]   = {}
        self._sorted_list        : list[Style] | None = None
        self._sequence_index     : int                = 0

        if styles is None:
            pass

        elif isinstance(styles, StyleSet):
            # copy the internal dictionary directly (it always is canonicalized)
            self._styles_by_canonical = dict(styles._styles_by_canonical)
            self._sorted_list         = list(styles._sorted_list) if styles._sorted_list is not None else None

        else:
            raise TypeError("Invalid type for styles argument.")


    @property
    def sorted_list(self) -> list[Style]:
        """
        Returns a list of styles sorted by its order property.
        """
        if  self._sorted_list is None:
            self._sorted_list = sorted(
                self._styles_by_canonical.values(),
                key = lambda style: style.order
                )
        return self._sorted_list


    @classmethod
    def from_string(cls,
                    string  : str,
                    category: str                  = "",
                    version : str | tuple[int,...] = "0.0.0"
                    ) -> StyleSet:
        """
        Create a new StyleSet instance from a multi-line string containing style definitions.

        Args:
           string: A multi-line string containing style definitions.
        Returns:
           A new `StyleSet` instance with styles loaded from the provided string.
        """
        if not isinstance(string, str) or len(string) == 0:
            return StyleSet()
        style_set = cls()
        style_set.load_from_string(string, category=category, version=version)
        return style_set


    def load_from_string(self,
                         string : str,
                         /,*,
                         category : str                  = "",
                         version  : str | tuple[int,...] = "0.0.0",
                         ) -> int:
        """
        Parse style definitions from a multi-line string and load them into the style set.

        This method scans the input string for special marker prefixes that delimit style
        definitions. Content between markers is treated as the template for the preceding
        style.

        Marker Reference:
            ">>>" - Style definition marker.
                    Indicates the start of a new style. Everything that follows (lines,
                    blank lines, etc.) belongs to this style's template until another
                    marker is encountered. The text after ">>>" becomes the style name.
                    Example: ">>>My Style" -> creates a style named "My Style"
            "#!"  - Shebang line (must be the first line of the input string).
                    The shebang itself is ignored during parsing.
                    Example: "#!ZSTYLES"
            "{#"  - Variable definition marker.
                    A compatibility marker with "Amazing Z-Image Workflow". Variables
                    are parsed and stored separately; the line itself is NOT included
                    in the style template.
                    Example: "{#MY_BAR}"
            ">::" - Workflow modifier marker.
                    A compatibility marker with "Amazing Z-Image Workflow". These
                    lines are parsed as configuration directives and NOT included
                    in the style template.
                    Example: ">::PIN-GROUP"

        Automatic Fallback Behavior:
            If the input string contains NO ">>>" markers, the entire string is treated
            as a single style called "Custom 1".

        Args:
            string  : The input string containing style definitions to be loaded.
            category: Default category assigned to styles that do not specify one.
            version : Default version string assigned to styles that do not specify one.
        Returns:
            The total number of styles that were successfully parsed and added to
            the style set. Returns 0 if the string was empty or contained markers
            not related to style definitions.

        Example:
            >>> input_text = '''
            ... #!ZSTYLES
            ...
            ... >>>Phone Photo
            ... Your photographs has android phone cam-quality.
            ... YOUR PHOTO: {$@}
            ...
            ... >>>Selfie
            ... The subject takes a selfie with their limb outstretched.
            ... THE SELFIE: {$@}
            ... '''
            >>> style_set = StyleSet()
            >>> count = style_set.load_from_string(input_text)
            # Loads 2 styles: "Phone Photo" and "Selfie"
        """
        all_lines = string.strip().splitlines()

        # skip shebang line if present (must be the first line)
        # (compatibility with "Amazing Z-Image Workflow")
        if all_lines and all_lines[0].startswith("#!"):
            all_lines.pop(0)

        # if the input string does not contain any explicit style definitions (>>>),
        # it is assumed to be plain text and automatically assigned to a default
        # style "Custom 1".
        if not any(line.startswith(">>>") for line in all_lines):
            all_lines.insert(0, ">>>Custom 1")

        # add sentinel element to force processing of last style
        all_lines.append(">>>EOF")

        load_count   : int       = 0
        marker       : str       = ""
        content      : list[str] = []

        for line in all_lines:
            line                = line.rstrip()  #< trailing whitespaces are lost at the end of each line
            new_marker          = line
            new_marker_detected = (
                new_marker.startswith("{#")  or  #< variable definition       (compatibility with Amazing Z-Image Workflow)
                new_marker.startswith(">::") or  #< action to modify workflow (compatibility with Amazing Z-Image Workflow)
                new_marker.startswith(">>>")     #< style definition !!
            )

            # if the line does not contain any new marker then
            # it is text that must be added to the content of the previous marker
            if not new_marker_detected:
                content.append(line)
                continue

            # at this point a new marker is detected
            # so the previous pending marker/content must be processed
            if marker.startswith(">>>"):
                style_name = marker[3:].strip()
                style      = Style(style_name,
                                   template = "\n".join(content).strip(),
                                   category = category,
                                   version  = version,
                                   )
                # try to add the new style
                if self.add_style(style):
                    load_count += 1

            # the new marker is stored as pending
            marker, content = new_marker, []

        return load_count


    def add_style(self, style: Style) -> bool:
        """
        Adds a new Style object to the set.

        The style is indexed by its canonical name internally.
        If the canonical name is empty (invalid style name), the style is silently rejected.

        Args:
            style: The `Style` object to be added.
        Returns:
            True if the style was added, False otherwise.
        """
        canonical_name = Style.canonicalize_name(style.name)
        # reject styles with invalid names or with empty templates
        if not canonical_name or not style.template:
            return False

        # retrieve the order assigned to the style category (defaults to 999)
        category_order = self.CATEGORY_SORT_ORDER.get(style.category, 999)

        # style.order is a tuple = (category_order, sequence_index)
        # this ensures that styles are grouped by category first,
        # and then sorted by their addition time
        if style.order == _UNDEFINED_ORDER:
            style.order = (category_order, self._sequence_index)
        self._styles_by_canonical[canonical_name] = style

        # reset sorted list cache and increment the sequence index
        self._sorted_list = None
        self._sequence_index += 1
        return True


    def get(self, style_name: str, default: Style | None = None) -> Style | None:
        """
        Retrieves a style by name, returning a default value if not found.

        This is a safe alternative to indexing with [] that won't raise a
        KeyError when the style name is missing.

        Args:
            style_name: The name of the style to retrieve.
            default   : Optional value to return if the style is not found.
        Returns:
            The `Style` object if found, otherwise the default value of None.

        Example:
            >>> # Get a non-existing style with default (returns default)
            >>> default_style = Style("fallback")
            >>> style_set = StyleSet()
            >>> style = style_set.get("nonexistent", default_style)
            >>> style.name
            'fallback'
            >>>
            >>> # Get a non-existing style without default (returns None)
            >>> result = style_set.get("nonexistent")
            >>> result is None
            True
        """
        canonical_name = Style.canonicalize_name(style_name)
        if not canonical_name:
            return default
        return self._styles_by_canonical.get(canonical_name, default)


    def by_version(self, version: str | tuple[int,...]) -> StyleSet:
        """
        Returns a StyleSet that contains every style tagged with `version`.

        Args:
            version: A semantic version string such as "1.0.0".
        Returns:
            A `StyleSet` instance holding styles for the given version.
            If no such version exists, an empty `StyleSet` is returned.
        """
        normalized_version = Style.normalize_version(version)
        styles = StyleSet()
        for style in self._styles_by_canonical.values():
            if style.version == normalized_version:
                styles.add_style(style)
        return styles


    def by_category(self, category: str) -> StyleSet:
        """
        Returns a StyleSet that contains every style with the specified category.

        Args:
            category: The category string used to filter styles.
        Returns:
            A `StyleSet` instance holding styles for the given category.
            If no such category exists, an empty `StyleSet` is returned.
        """
        styles = StyleSet()
        for style in self._styles_by_canonical.values():
            if style.category == category:
                styles.add_style(style)
        return styles


    def names(self) -> list[str]:
        """
        Return a list of all style names stored in the set.

        Returns:
            A list of style names sorted by the style order attribute.

        Example:
            >>> for name in style_set.names():
            ...     print(name)
        """
        return [style.name for style in self.sorted_list]


    def quoted_names(self) -> list[str]:
        """
        Return a list of all quoted style names stored in the set.

        Returns:
            A list of quoted style names sorted by the style order attribute.

        Example:
            >>> for quoted_name in style_set.quoted_names():
            ...     print(quoted_name)
        """
        return [style.quoted_name for style in self.sorted_list]


    def canonical_names(self) -> KeysView[str]:
        """
        Return a dynamic view of all canonical style names stored in the set.

        Canonical names are normalized versions of the style names.
        They are converted to lowercase with:
          - spaces replaced by underscores
          - accent marks removed
          - rare characters converted to 'x'

        Important:
            The returned view is **NOT** sorted by style order attribute.

        Returns:
            A view object containing all canonical style names.

        Example:
            >>> for canonical_name in style_set.canonical_names():
            ...     print(canonical_name)
        """
        return self._styles_by_canonical.keys()


    def categories(self) -> list[str]:
        """
        Returns a list of all categories of the styles stored in the set.
        """
        # get unique categories
        unique_cats = {style.category for style in self._styles_by_canonical.values()}

        # sort using the CATEGORY_SORT_ORDER map,
        # default 999 ensures unlisted categories appear at the end.
        return sorted(
            unique_cats,
            key=lambda c: (self.CATEGORY_SORT_ORDER.get(c, 999), c)
        )


    def __getitem__(self, style_name: str) -> Style:
        """
        Enables access to a specific style by its name.

        The input name is canonicalized before lookup.
        Example usage: `style_set["cinematic_look"]`
        """
        canonical_name = Style.canonicalize_name(style_name)
        return self._styles_by_canonical[canonical_name]


    def __delitem__(self, style_name: str) -> None:
        """
        Removes a style from the set by its name.

        The input name is canonicalized before deletion.
        Example usage: `del style_set["cinematic_look"]`
        """
        canonical_name = Style.canonicalize_name(style_name)
        del self._styles_by_canonical[canonical_name]


    def __contains__(self, style_name: str) -> bool:
        """
        Checks if a specific style exists in the set.

        The input name is canonicalized before checking.
        Example usage: `"oil_painting" in style_set`
        """
        canonical_name = Style.canonicalize_name(style_name)
        return canonical_name in self._styles_by_canonical


    def __len__(self) -> int:
        """
        Returns the number of styles currently stored in the set.

        Example usage: `len(style_set)`
        """
        return len(self._styles_by_canonical)


    def __iter__(self) -> Iterator[Style]:
        """
        Allows iteration over the Style objects within this set.

        Example usage: `for style in style_set:`
        """
        return iter(self.sorted_list)


    def __repr__(self) -> str:
        """
        Provides a string representation of the StyleSet instance.
        """
        return f"StyleSet(styles={list(self._styles_by_canonical.keys())})"


#================================= HELPERS =================================#

def remove_style_from_text(text: str, name: str) -> str:
    """
    Removes a style definition from the given text.

    Args:
        text: The original text containing multiple style definitions.
        name: The name of the style to be removed.

    Returns:
        The modified text with the specified style removed.
    """
    name_to_find = Style.canonicalize_name(name)
    if not name_to_find:
        return text

    lines = text.splitlines()

    # find the beginning of the style definition
    start = None
    for i, line in enumerate(lines):
        if line.startswith(">>>") and Style.canonicalize_name(line[3:]) == name_to_find:
            start = i
            break
    else:
        return text #<< loop finished without break: name not found

    # find the end of the style definition
    end = None
    for i, line in enumerate(lines[start + 1:], start=start + 1):
        if line.startswith((">>>", ">::", "{#")):
            end = i
            break

    # remove the style block (inclusive start, exclusive end)
    kept_lines = lines[:start] if end is None else lines[:start] + lines[end:]
    return "\n".join(kept_lines)


def append_style_to_text(text: str, name: str, template: str) -> str:
    """
    Appends a style definition to the given text.
    Args:
        text    : The original text to which the style will be added.
        name    : The name of the new style.
        template: The content defining the new style.
    Returns:
        The modified text with the new style appended.
    """
    return f"{text}>>>{name}\n{template}\n\n\n"



