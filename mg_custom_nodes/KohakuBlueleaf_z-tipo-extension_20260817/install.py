"""A1111 / Forge install hook.

Only tipo-kgen is installed here, because scripts/tipo.py imports it at module
scope. llama-cpp-python is deliberately left to first use: picking its wheel
needs a network round trip, and paying that on every webui launch would stall
startup for users who never enable TIPO.
"""

import os
import sys

if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from tipo_installer import ensure_tipo_kgen, logger

if not ensure_tipo_kgen():
    logger.warning("tipo-kgen is not available; the TIPO tab will not work.")
