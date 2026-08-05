"""Where prebuilt llama-cpp-python wheels come from, and which one each GPU wants.

This is the only file that needs editing when upstream moves the release feed,
renames a variant, or ships a new accelerator. Everything here is data; the
matching logic lives in `wheels.py` and reads these names without interpreting
them further.

Variant tags are discovered from the live release feed, never listed here, so a
newly published `cu135` or `rocm80` is picked up with no code change.
"""

SCHEME_VERSION = 1

DISTRIBUTION = "llama-cpp-python"
IMPORT_NAME = "llama_cpp"

REPO = "abetlen/llama-cpp-python"
RELEASES_API = "https://api.github.com/repos/{repo}/releases?per_page=100"
RELEASE_PAGES = 2

TAG_PATTERN = r"^v(?P<version>\d+(?:\.\d+)*)(?:-(?P<variant>.+))?$"

VERSION_FLOOR = "0.3.0"
VERSION_CEILING = None

CPU_VARIANT = ""

# major_bound: a cu12 wheel cannot load against a cu13 runtime.
NUMERIC_VARIANTS = {
    "cuda": {"pattern": r"^cu(?P<major>\d+)(?P<minor>\d)$", "major_bound": True},
    "rocm": {"pattern": r"^rocm(?P<major>\d+)(?P<minor>\d)$", "major_bound": True},
}

# Ordered fallback chain per detected accelerator. `@cuda` / `@rocm` resolve
# through NUMERIC_VARIANTS; bare strings are literal release tags.
PREFERENCE = {
    "cuda": ["@cuda", CPU_VARIANT],
    "rocm": ["@rocm", "hip-radeon", "vulkan", CPU_VARIANT],
    "hip": ["hip-radeon", "@rocm", "vulkan", CPU_VARIANT],
    "metal": ["metal", CPU_VARIANT],
    "xpu": ["vulkan", CPU_VARIANT],
    "vulkan": ["vulkan", CPU_VARIANT],
    "cpu": [CPU_VARIANT],
}

# Consulted before any detection, so a user can pin a variant that the probe
# would never choose (Vulkan on an NVIDIA card, CPU on a flaky driver).
ENV_VARIANT = "TIPO_LLAMA_VARIANT"
ENV_VERSION_CEILING = "TIPO_LLAMA_MAX_VERSION"
ENV_DISABLE = "TIPO_NO_AUTO_INSTALL"

NETWORK_TIMEOUT = 15

KGEN_DISTRIBUTION = "tipo-kgen"
KGEN_IMPORT_NAME = "kgen"
KGEN_MIN_VERSION = "0.3.0"
