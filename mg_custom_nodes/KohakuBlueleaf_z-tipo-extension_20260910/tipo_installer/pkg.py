"""Install packages through whichever package manager this host actually has.

The resolution order mirrors ComfyUI-Manager's, because that is the order the
real hosts satisfy: ComfyUI Desktop ships uv and no pip at all, portable builds
ship an embedded interpreter that needs -s so user site-packages stay out, and
A1111 wants its own run_pip so the user's index-url settings are honoured.
"""

import subprocess
import sys

from .log import logger

_UNSET = object()
_resolved = _UNSET


def host():
    if "launch" in sys.modules and hasattr(sys.modules["launch"], "run_pip"):
        return "webui"
    if "folder_paths" in sys.modules:
        return "comfyui"
    return "standalone"


def _embedded():
    return "python_embeded" in sys.executable or "python_embedded" in sys.executable


def _candidates():
    base = [sys.executable] + (["-s"] if _embedded() else [])
    yield base + ["-m", "pip", "--version"], base + ["-m", "pip", "install"]
    yield (
        base + ["-m", "uv", "--version"],
        base + ["-m", "uv", "pip", "install", "--python", sys.executable],
    )
    yield ["uv", "--version"], ["uv", "pip", "install", "--python", sys.executable]


def _probe(command):
    try:
        subprocess.check_output(command, stderr=subprocess.DEVNULL, timeout=10)
        return True
    except Exception:
        return False


def installer():
    """Install-command prefix for this environment, or None if nothing works."""
    global _resolved
    if _resolved is _UNSET:
        _resolved = None
        for probe, install in _candidates():
            if _probe(probe):
                _resolved = install
                break
    return _resolved


def describe(args):
    command = installer() or ["uv", "pip", "install", "--python", sys.executable]
    return subprocess.list2cmdline(command + list(args))


def install(args, description):
    args = list(args)
    run_pip = getattr(sys.modules.get("launch"), "run_pip", None)
    if run_pip is not None:
        try:
            run_pip(f"install {' '.join(args)}", description)
            return True
        except Exception as error:
            logger.warning(f"{description}: webui run_pip failed ({error})")
            return False

    command = installer()
    if command is None:
        logger.warning("No package manager found (tried pip, python -m uv, uv).")
        logger.warning(f"To install {description} yourself, run:\n    {describe(args)}")
        return False

    logger.info(f"Installing {description} via {command[0]}")
    result = subprocess.run(
        command + args, capture_output=True, text=True, errors="ignore", check=False
    )
    if result.returncode == 0:
        return True

    logger.warning(
        f"{description}: install failed\n{(result.stderr or '').strip()[-2000:]}"
    )
    logger.warning(f"To retry by hand:\n    {describe(args)}")
    return False
