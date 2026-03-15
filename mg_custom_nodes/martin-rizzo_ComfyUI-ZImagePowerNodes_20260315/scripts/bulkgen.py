"""
File    : bulkgen.py
Purpose : Script to create a batch of images based on a configuration file.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 14, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import argparse
from typing import overload
from copy import deepcopy

DEFAULT_PROMPT       =""
DEFAULT_CHECKPOINT   =os.getenv("BULKGEN_CHECKPOINT"   , "z_image_turbo-Q5_K_S.gguf")
DEFAULT_STYLE        =os.getenv("BULKGEN_STYLE"        , "Casual Photo"             )
DEFAULT_STEPS        =os.getenv("BULKGEN_STEPS"        , "9"                        )
DEFAULT_SEED         =os.getenv("BULKGEN_SEED"         , "1"                        )
DEFAULT_INCALIBRATION=os.getenv("BULKGEN_INCALIBRATION", "1.0"                      )
FILE_PREFIX = "bgen"


# ANSI escape codes for colored terminal output
RED      = '\033[91m'
DKRED    = '\033[31m'
YELLOW   = '\033[93m'
DKYELLOW = '\033[33m'
GREEN    = '\033[92m'
CYAN     = '\033[96m'
DKGRAY   = '\033[90m'
RESET    = '\033[0m'

#============================= ERROR MESSAGES ==============================#

def disable_colors():
    global RED, DKRED, YELLOW, DKYELLOW, GREEN, CYAN, DKGRAY, RESET
    RED, DKRED, YELLOW, DKYELLOW, GREEN, CYAN, DKGRAY, RESET = "", "", "", "", "", "", "", ""


def info(message: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an informational message to the error stream.
    """
    print(f"{" "*padding}{CYAN}\u24d8 {message}{RESET}", file=file)


def warning(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a warning message to the standard error stream.
    """
    print(f"{" "*padding}{CYAN}[{YELLOW}WARNING{CYAN}]{DKYELLOW} {message}{RESET}", file=file)
    for info_message in info_messages:
        info(info_message, padding=padding, file=file)


def error(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an error message to the standard error stream.
    """
    print(f"{" "*padding}{DKRED}[{RED}ERROR!{DKRED}]{DKYELLOW} {message}{RESET}", file=file)
    for info_message in info_messages:
        info(info_message, padding=padding, file=file)


def fatal_error(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a fatal error message to the standard error stream and exits with status code 1.
    """
    error(message, *info_messages, padding=padding, file=file)
    sys.exit(1)


#================================= HELPERS =================================#

def _to_float(value: str) -> float:
    """Converts a string to float, handling optional percentage signs."""
    value = value.strip()
    if not value: raise ValueError("Empty string provided for float conversion.")
    elif value.endswith('%')  : return float(value[:-1]) / 100.0
    elif value.startswith('%'): return float(value[1:] ) / 100.0
    return float(value)


def _to_int(value: str) -> int:
    """Converts a string to int."""
    value = value.strip()
    if not value: raise ValueError("Empty string provided for int conversion.")
    return int(float(value))


@overload
def to_float(values: str) -> float: ...

@overload
def to_float(values: list[str]) -> list[float]: ...

def to_float(values: str | list[str]) -> float | list[float]:
    """
    Converts a string (or list of strings) to float(s), supporting percentages.
    Raises:
        ValueError: If the input contains invalid characters for numbers/percentages.
    """
    try:
        if isinstance(values, str):
            return _to_float(values)
        return [_to_float(value) for value in values]
    except ValueError as e:
        raise ValueError(f"Invalid float/percent values: {values}") from e


@overload
def to_int(values: str) -> int: ...

@overload
def to_int(values: list[str]) -> list[int]: ...

def to_int(values: str | list[str]) -> int | list[int]:
    """
    Converts a string (or list of strings) to int(s).
    Raises:
        ValueError: If the input contains invalid characters for integer numbers.
    """
    try:
        if isinstance(values, str):
            return _to_int(values)
        return [_to_int(value) for value in values]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid int values: {values}") from e


#======================= GenParams & GenMultiParams ========================#

# class GenParams:
#     """
#     Represents a single generation parameter configuration.

#     This class holds specific values for text-to-image generation parameters 
#     without combinatorial support. It is used when a single unique image 
#     needs to be generated with fixed settings.

#     Example Usage:
#         params = GenParams(
#             prompt="A cyberpunk city",
#             style="anime"
#             checkpoint="z_image_turbo-Q5_K_S.gguf",
#         )

#     Attributes:
#         prompt        (str)  : The text prompt for the image generation.
#         checkpoint    (str)  : The model checkpoint to use.
#         style         (str)  : The artistic style to apply.
#         steps         (int)  : Number of denoising steps.
#         seed          (int)  : Random seed (-1 for random).
#         incalibration (float): Initial noise calibration percentage (0.0 - 1.0).
#     """
#     def __init__(self, *,
#                  prompt       : str   | None = None,
#                  checkpoint   : str   | None = None,
#                  style        : str   | None = None,
#                  steps        : int   | None = None,
#                  seed         : int   | None = None,
#                  incalibration: float | None = None,
#     ) -> None:
#         #nota: incalibration es la abreviacion de 'initial noise calibration' y es un porcentaje en este caso expresado en el rango 0.0, 1.0
#         self.prompt        = prompt        if type(prompt)        == str  else DEFAULT_PROMPT
#         self.checkpoint    = checkpoint    if type(checkpoint)    == str  else DEFAULT_CHECKPOINT
#         self.style         = style         if type(style)         == str  else DEFAULT_STYLE
#         self.steps         = steps         if type(steps)         == int  else DEFAULT_STEPS
#         self.seed          = seed          if type(seed)          == int  else DEFAULT_SEED
#         self.incalibration = incalibration if type(incalibration) == list else DEFAULT_INCALIBRATION

#     def __str__(self) -> str:
#         """Returns a readable string representation of the parameters."""
#         return f"GenParams(prompt='{self.prompt}', style='{self.style}', checkpoint='{self.checkpoint}', steps={self.steps}, seed={self.seed}, incalibration={self.incalibration})"


class GenMultiParams:
    """
    Represents a multi-value generation parameter configuration.

    This class allows storing multiple values for each parameter to support
    combinatorial generation. For example, if you want to test different
    checkpoints, distinct steps, and different seeds but with the same prompt.

    Example Usage:
        multi_params = GenMultiParams(
            prompt="A cyberpunk city",
            style=["anime", "phone photo"]
            checkpoint=["z_image_turbo-Q5_K_S.gguf", "z_image_turbo_bf16.safetensors"],
            steps=[5,7,9],
            seed=[1]
        )
        # Later logic could generate all images using `self.generate_combinations()`

    Attributes:
        prompt        (str)        : The single prompt text shared across all variations.
        checkpoint    (List[str])  : List of model checkpoint identifiers.
        style         (List[str])  : List of artistic styles to test.
        steps         (List[int])  : List of denoising step counts.
        seed          (List[int])  : List of random seeds (-1 is not valid inside a list logic usually).
        incalibration (List[float]): List of initial noise calibration values.
    """
    def __init__(self, *,
                 prompt       : str         | None = None,
                 checkpoint   : list[str]   | None = None,
                 style        : list[str]   | None = None,
                 steps        : list[int]   | None = None,
                 seed         : list[int]   | None = None,
                 incalibration: list[float] | None = None,
    ) -> None:
        #nota: incalibration es la abreviacion de 'initial noise calibration' y es un porcentaje en este caso expresado en el rango 0.0, 1.0
        self.prompt        = prompt        if type(prompt)        == str  else DEFAULT_PROMPT
        self.checkpoint    = checkpoint    if type(checkpoint)    == list else [DEFAULT_CHECKPOINT]
        self.style         = style         if type(style)         == list else [DEFAULT_STYLE]
        self.steps         = steps         if type(steps)         == list else [int(DEFAULT_STEPS)]
        self.seed          = seed          if type(seed)          == list else [int(DEFAULT_SEED)]
        self.incalibration = incalibration if type(incalibration) == list else [float(DEFAULT_INCALIBRATION)]

    def set_param(self, param_and_values: str, forced_values: str | None = None):
        param, _, values = param_and_values.partition("=")
        param  = param.strip().lower()
        if param.startswith('$'): param = param[1:]

        # builds the list of values taking into account if the values were forced
        if forced_values: values = forced_values
        values = [v.strip() for v in values.split(",")]

        if param == "ckpt":
            self.checkpoint = values
        elif param == "style":
            self.style = values
        elif param == "steps":
            self.steps = to_int(values)
        elif param == "seed":
            self.seed  = to_int(values)
        elif param == "inc" or param == "incalibration" or param == "initial_noise_calibration":
            self.incalibration = to_float(values)
        else:
            raise ValueError(f"Unknown generation parameter: {param}")

    def copy(self) -> 'GenMultiParams':
        """Returns a deep copy with all attributes duplicated independently.
        Example:
            original = GenMultiParams(prompt="A cyberpunk city", steps=[5, 7, 9])
            copy_instance = original.copy()
            # Modifying copy_instance.steps will not affect original
            original.steps.append(10)
            assert copy_instance.steps == [5, 7, 9]  # True
        """
        return deepcopy(self)

    def __str__(self) -> str:
        """Returns a readable string representation of the multi-value configuration."""
        return f"GenMultiParams(checkpoint={self.checkpoint}, style={self.style}, "         \
               f"steps={self.steps}, seed={self.seed}, incalibration={self.incalibration}, "\
               f"prompt=\"{self.prompt}\")"


#================================ PROCESSOR ================================#

class Processor:
    def __init__(self):
        self._action             = ""
        self._content            = ""
        self._is_first_line      = True
        self._is_shebang_line    = False
        self._is_comment_zone    = True
        self._global_params      = GenMultiParams()
        self._local_params       = GenMultiParams()
        self._custom_styles      = {}

    @classmethod
    def from_file(cls, path: str) -> "Processor":
        processor = cls()
        with open(path, "r") as file:
            for line in file: processor.digest_text_line(line)
            processor.end()
        return processor


    def digest_text_line(self, line: str):
        self._is_shebang_line = self._is_first_line and line.startswith("#!")
        self._is_first_line   = False

        # note: trailing whitespaces are lost at the end of each line
        line = line.rstrip()

        # discard comments if it is in a zone that can have comments
        if self._is_comment_zone and line.startswith(">>#"):
            return

        # verify if the line is a new action to be processed
        if ( self._is_shebang_line  or #< sheban "#!ZCONFIG"        (compatibility with Amazing Z-Image Workflow)
             line.startswith(">::") or #< action to modify workflow (compatibility with Amazing Z-Image Workflow)
             line.startswith(">>>") or #< style selection / prompt start
             line.startswith(">>$")    #< parameter configuration
           ):
            self.process_action(self._action, self._content)
            self._action, self._content = line, ""
        else:
            self._content += line + "\n"


    def end(self):
        self.process_action(self._action, self._content)
        self._action, self._content = "", ""


    def process_action(self, action: str, content: str, /):
        """
        Process configuration commands based on the action syntax.

        Parses the provided action to determine if it is a parameter assignment, a
        prompt definition with parameters, or a simple prompt block, and executes
        the corresponding logic.

        Args:
            action  (str): The action to be processed (case inseensitive)
            content (str): The accompanying content associated with the action.
        """
        action       = action.strip()
        action_lower = action.lower()
        content      = content.strip()

        # the action of ">>>Prompt" can have an optional ':' on the right side
        if action_lower.startswith(">>>prompt"):
            action       = action.rstrip(":")
            action_lower = action_lower.rstrip(":")

        # processes global parameter assignment
        # format: ">>>$param = valueA,valueB,...,valueN"
        if action.startswith(">>>$"):
            self._global_params.set_param( action[3:] )

        # processes a prompt with parameters
        # format: ">>>prompt{ param=valueA,valueB,...; param2=value2; ... }"
        #         and following lines are the full prompt
        elif action_lower.startswith(">>>prompt{") and action.endswith("}"):
            local_params        = self._global_params.copy()
            local_params.prompt = content.strip()
            for param_and_values in action[10:-1].split(";"):
                local_params.set_param( param_and_values )
            print("##>>----------")
            print(local_params)

        # processes prompt without parameters
        # format: ">>>" or ">>>prompt:"
        #         and following lines are the full prompt
        elif action_lower == ">>>prompt" or action == ">>>":
            local_params = self._global_params.copy()
            local_params.prompt = content.strip()
            print("##>>----------")
            print(local_params)

        # processes classical style selection
        # format: ">>>Style Name"
        elif action.startswith(">>>"):
            style = action[3:].strip()
            self._global_params.set_param("style", style)
            if content:
                # TODO: verify that there are no commas
                self._custom_styles[style] = content











# def parse_config_file(file: str) -> dict[str, list]:
#     string = ""
#     with open(file, 'r', encoding='utf-8') as f:
#         string = f.read()
#     return parse_config(string) if string else {}


# def parse_config_lines(config_lines: list[str]) -> dict[str, list]:
#     """Parses a configuration file and returns a dictionary of the parsed data.
#     Args:
#         config_file (str): The path to the configuration file to parse.
#     Returns:
#         dict: A dictionary containing the parsed data from the
#     """

#     # ">::" = action to modify workflow (compatibility with Amazing Z-Image Workflow)
#     # ">>>" = style selection (e.g. ">>>Phone Photo")
#     # ">>>" = prompt start (e.g. ">>>", "A man in a red shirt is standing in front of a bar.")
#     # ">>$" = parameter configuration (e.g. ">>$ ckpt=zimage_alt.safetensors")
#     # ">>#" = comment (e.g. ">># This is a comment")

#     action             = ""
#     content            = ""
#     is_first_line      = True
#     is_comment_allowed = True
#     for line in config_lines:

#         is_shebang_line = is_first_line and line.startswith("#!")
#         is_first_line   = False

#         # note: trailing whitespaces are lost at the end of each line
#         line = line.rstrip()

#         # verify if the line is a new action to be processed
#         if ( is_shebang_line        or #< sheban "#!ZCONFIG"        (compatibility with Amazing Z-Image Workflow)
#              line.startswith(">::") or #< action to modify workflow (compatibility with Amazing Z-Image Workflow)
#              line.startswith(">>>") or #< style selection / prompt start
#              line.startswith(">>$") or #< parameter configuration
#              (is_comment_allowed and line.startswith(">>#"))
#            ):
#             process_config_action(action, content)
#             action, content = line, ""
#         else:
#             content += line + "\n"

#     # before ending, process any pending action
#     process_config_action(action, content)
#     return style_group


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args=None, parent_script=None):
    """
    Main entry point for the script.
    Args:
        args          (optional): List of arguments to parse. Default is None, which will use the command line arguments.
        parent_script (optional): The name of the calling script if any. Used for customizing help output.
    """
    prog = None
    if parent_script:
        prog = parent_script + " " + os.path.basename(__file__).split('.')[0]

    # set up argument parser for the script
    parser = argparse.ArgumentParser(
        prog            = prog,
        description     = "Create a batch of images based on a configuration file",
        formatter_class = argparse.RawTextHelpFormatter
        )
    parser.add_argument('--no-color'       , action='store_true', help="Disable colored output.")
    parser.add_argument('config_files'     , nargs='+', metavar='FILE', help="One or more configuration files to generate images from.")
    args = parser.parse_args(args=args)

    # if the user requested to disable colors, call disable_colors()
    if args.no_color:
        disable_colors()

    # loop through the configuration files and generate images
    for config_file in args.config_files:
        processor = Processor.from_file(config_file)



if __name__ == "__main__":
    main()