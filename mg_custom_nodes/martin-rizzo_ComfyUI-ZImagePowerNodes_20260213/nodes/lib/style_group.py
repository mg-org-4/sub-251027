"""
File    : styles/style_group.py
Purpose : A class for managing and retrieving style templates based on their names.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import re


#================================== STYLE ==================================#

class Style:
    """
    Represents a style with a name, template, description and tags.

    Attributes:
        name        (str): The name of the style.
        template    (str): The template associated with this style.
        description (str): A description for the style.
        tags       (list): Tags or categories associated with the style.

    Properties:
        comma_separated_tags (str): Returns a comma-separated string of all tags.
        quoted_name          (str): Returns the style name enclosed in double quotes.
        slug                 (str): Returns a slug version of the style's name for URL use.
    """
    def __init__(self,
                 name       : str, /,*,
                 template   : str,
                 description: str = "",
                 tags       : list = []
                 ):
        self.name        = name
        self.template    = template
        self.description = description
        self.tags        = tags


    @property
    def comma_separated_tags(self) -> str:
        """Returns a comma-separated string of all tags associated with the style."""
        return ", ".join(self.tags)


    @property
    def quoted_name(self) -> str:
        """Returns the name of the style enclosed in double quotes."""
        return f'"{self.name}"'


    @property
    def slug(self) -> str:
        """Generates a slug from the style's name for use in URLs or file names."""
        name = self.name.strip().lower()
        name = name.replace(" ", "_")           #< spaces to underscores
        name = re.sub(r"['\"`]", "", name)      #< remove any quotes
        name = re.sub(r"[^a-z0-9_]", "x", name) #< any strange character will be converted to 'x'
        return name



#=============================== STYLE GROUP ===============================#

class StyleGroup:
    """
    A group of style templates indexed by name.

    A class for managing and retrieving style templates based on their names.
    The indexation is case-insensitive and when indexing the names can be provided quoted or not.

    Args:
        category              (str): The category of the style group. Defaults to an empty string.
        version               (str): The version of the style group. Defaults to an empty string.
        styles        (dict | None): A dictionary mapping style names to their template. Defaults to None.
        ordered_names (list | None): The ordered list of style names. Defaults to None.

    Attributes:
        category (str): The category of the style group.
        version  (str): The version of the style group.
    """

    def __init__(self,
                 category     : str = "",
                 version      : str = "",
                 styles       : dict[str, str] | None = None,
                 ordered_names: list[str]      | None = None,
                 ):

        self.category = category
        self.version  = version
        self._styles_by_lowername = {}
        self._names_by_lowername  = {}
        self._ordered_names       = []

        # if no styles are provided then leave everything empty and do nothing
        if styles is None:
            return

        # if no ordered names are provided then add styles unordered from the dictionary
        if ordered_names is None:
            for name, template in styles.items():
                self.add_style(name, template)
            return

        # we have the ordered list of names then add styles in order
        for name in ordered_names:
            if name in styles:
                self.add_style(name, styles[name])


    @classmethod
    def from_string(cls,
                    string : str,
                    /,*,
                    category : str = "",
                    version  : str = "",
                    ) -> "StyleGroup":
        """
        Creates a StyleGroup instance from a string containing style definitions.

        Args:
            string   (str): The input string containing style definitions.
            category (str): The category of the style group. Defaults to an empty string.
            version  (str): The version of the style group. Defaults to an empty string.

        Returns:
            A new instance of `StyleGroup` parsed from the input string.
        """
        style_group = StyleGroup(category=category, version=version)
        action  = None
        content = ""
        all_lines = string.splitlines()

        # if the input string does not contain any explicit style definitions (>>>),
        # it is assumed to be plain text and automatically assigned to a default
        # style "Custom 1".
        if not any(line.startswith(">>>") for line in all_lines):
            all_lines.insert(0, ">>>Custom 1")

        is_first_line = True
        for line in all_lines:
            is_shebang_line = is_first_line and line.startswith("#!")
            is_first_line   = False

            line = line.rstrip() #< trailing whitespaces are lost at the end of each line
            if ( is_shebang_line        or #< sheban "#!ZCONFIG"        (compatibility with Amazing Z-Image Workflow)
                 line.startswith("{#")  or #< variable definition       (compatibility with Amazing Z-Image Workflow)
                 line.startswith(">::") or #< action to modify workflow (compatibility with Amazing Z-Image Workflow)
                 line.startswith(">>>")    #< style definition !!
               ):
                # a new action is detected, so the previous pending one is processed
                if action and action.startswith(">>>"):
                    style_name = action[3:].strip()
                    style_group.add_style(style_name, content.strip())

                # the new action is stored as pending
                action, content = line, ""
            else:
                content += line + "\n"

        # before ending, process any pending action
        if action and action.startswith(">>>"):
            style_name = action[3:].strip()
            style_group.add_style(style_name, content.strip())

        return style_group

    @staticmethod
    def apply_style_template(prompt        : str,
                             style_template: str,
                             /,*,
                             spicy_impact_booster: bool = False) -> str:
        """
        Applies a given style template to a prompt with optional spicy content boost.

        Args:
            prompt         (str): The input text prompt to be styled.
            style_template (str): The template style into which the prompt will be inserted.
                                  This is a string template that should contain "{$@}" and
                                  can be obtained with `style_group.get_style_template(name)`
            spicy_impact_booster (optional): If True, adds spicy content to the output. Default is False.

        Returns:
            The final styled prompt ready for use.
        """
        spicy_content = ""
        if spicy_impact_booster:
            spicy_content = "attractive and spicy content, where any woman is sexy and provocative, with"

        result = style_template
        result = result.replace("{$spicy-content-with}", spicy_content) #< the secret spicy dressing
        result = result.replace("{$@}"                 , prompt       ) #< prompt to be styled
        result = result.replace("  ", " ")                              #< fix double spaces
        return result



    def contains(self, name: str) -> bool:
        """Check whether the group contains a given style or not."""
        return name.lower() in self._styles_by_lowername


    def get_style(self, name: str) -> Style | None:
        """Return the style for a given name. If it doesn't exist, returns None."""
        lowername = name.strip().lower()
        # the name can be quoted with single or double quotes
        if(  ( lowername.startswith("'") and lowername.endswith("'") )  or
             ( lowername.startswith('"') and lowername.endswith('"') )  ):
            lowername = lowername[1:-1]
        return self._styles_by_lowername.get(lowername)


    def get_style_template(self, name: str, default: str = "") -> str:
        """Return the style template for a given name. If it doesn't exist, returns `default` or empty string."""
        style = self.get_style(name)
        return style.template if style else default


    def add_style(self, name: str, style: Style|str):
        """Add a new style or update an existing one."""
        lowername = name.strip().lower()

        # if instead of an object, a string is passed as parameter,
        # then it is assumed that this string is the template
        if isinstance(style, str):
            style = Style(name, template=style)

        # if the style already exists, then it is only updated
        if lowername in self._styles_by_lowername:
            self._styles_by_lowername[lowername] = style
            return
        # adds a new style
        self._styles_by_lowername[lowername] = style
        self._names_by_lowername[lowername] = name
        self._ordered_names.append(name)


    def remove_style(self, name: str):
        """Remove a style by its name."""
        lowername = name.strip().lower()
        if lowername not in self._styles_by_lowername:
            return
        name = self._names_by_lowername[lowername] or name
        del self._styles_by_lowername[lowername]
        del self._names_by_lowername[lowername]
        self._ordered_names.remove(name)


    def update(self, style_group: "StyleGroup"):
        """Update this style group with another one."""
        for name in style_group.get_names():
            style = style_group.get_style(name)
            if style: self.add_style(name, style)


    def get_names(self, /,*, quoted: bool | str = False) -> list[str]:
        """
        Returns a list of all style names.
        Args:
            quoted (optional):
                if True then each key is returned as a text with double quotes around it.
                if a string is passed then it's used as the quote char. Defaults to False.
        """
        if not quoted:
            return self._ordered_names
        quote_char = quoted if isinstance(quoted, str) else '"'
        return [ f'{quote_char}{x}{quote_char}' for x in self.get_names() ]


    def __len__(self):
        """Return the number of styles."""
        return len(self._styles_by_lowername)


    def __contains__(self, name):
        """Check if a style exists by its name."""
        return self.contains(name)


    def __str__(self) -> str:
        return f"StyleGroup({len(self._styles_by_lowername)} styles)"

