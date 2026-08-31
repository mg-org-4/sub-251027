import copy
import logging
import sys
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[0;36m",
        "INFO": "\033[0;32m",
        "WARNING": "\033[0;33m",
        "ERROR": "\033[0;31m",
        "CRITICAL": "\033[0;37;41m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        colored = copy.copy(record)
        seq = self.COLORS.get(colored.levelname, self.COLORS["RESET"])
        colored.levelname = f"{seq}{colored.levelname}{self.COLORS['RESET']}"
        return super().format(colored)


logger = logging.getLogger("TIPO-installer")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            "[%(name)s]-|%(asctime)s|-%(levelname)s: %(message)s", "%H:%M:%S"
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
