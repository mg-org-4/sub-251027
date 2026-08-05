"""Pick the best prebuilt llama-cpp-python wheel for this machine.

Variants and versions come from the live release feed rather than a table in
the repo, so a newly published accelerator needs no code change here.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import NamedTuple

from . import scheme
from .hardware import Accelerator, detect, forced_variant
from .log import logger

try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    sys_tags = parse_wheel_filename = None

_TAG_RE = re.compile(scheme.TAG_PATTERN)
_catalog_cache = None


class Candidate(NamedTuple):
    url: str
    version: str
    variant: str
    asset: str

    def describe(self):
        return f"{self.version} ({self.variant or 'cpu'})"


def _as_version(text):
    return tuple(int(part) for part in re.findall(r"\d+", str(text)))


def _ceiling():
    override = os.environ.get(scheme.ENV_VERSION_CEILING, "").strip()
    return override or scheme.VERSION_CEILING


def _within_bounds(version):
    if version < _as_version(scheme.VERSION_FLOOR):
        return False
    ceiling = _ceiling()
    return not (ceiling and version > _as_version(ceiling))


def _request(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "tipo-installer"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=scheme.NETWORK_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_releases():
    base = scheme.RELEASES_API.format(repo=scheme.REPO)
    releases = []
    for page in range(1, scheme.RELEASE_PAGES + 1):
        try:
            batch = _request(f"{base}&page={page}")
        except (urllib.error.URLError, OSError, ValueError) as error:
            logger.warning(f"Could not read the llama-cpp-python release feed: {error}")
            break
        if not batch:
            break
        releases.extend(batch)
    return releases


def _catalog():
    """variant -> [(version tuple, tag, [asset names])], newest first."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    grouped = {}
    for release in _fetch_releases():
        if release.get("draft") or release.get("prerelease"):
            continue
        matched = _TAG_RE.match(release.get("tag_name", ""))
        if not matched:
            continue
        version = _as_version(matched.group("version"))
        if not _within_bounds(version):
            continue
        assets = [a.get("name", "") for a in release.get("assets") or []]
        variant = matched.group("variant") or scheme.CPU_VARIANT
        grouped.setdefault(variant, []).append((version, release["tag_name"], assets))

    for entries in grouped.values():
        entries.sort(key=lambda item: item[0], reverse=True)

    _catalog_cache = grouped
    return grouped


def _supported_tags():
    if sys_tags is None:
        return None
    try:
        return {str(tag) for tag in sys_tags()}
    except Exception:
        return None


def _compatible_asset(assets):
    wheels = [name for name in assets if name.endswith(".whl")]
    supported = _supported_tags()
    for name in wheels:
        if supported is None:
            if _crude_match(name):
                return name
            continue
        try:
            tags = parse_wheel_filename(name)[3]
        except Exception:
            continue
        if any(str(tag) in supported for tag in tags):
            return name
    return None


def _crude_match(name):
    """Tag check for environments without `packaging` installed."""
    platform_tag = {
        "win32": "win_amd64",
        "darwin": "macosx",
        "linux": "linux_x86_64" if sys.maxsize > 2**32 else "linux_i686",
    }.get(sys.platform, "")
    if platform_tag and platform_tag.split("_")[0] not in name:
        return False
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return "-py3-" in name or python_tag in name


def _numeric_choices(kind, runtime_version, available):
    spec = scheme.NUMERIC_VARIANTS.get(kind)
    if spec is None or runtime_version is None:
        return []
    pattern = re.compile(spec["pattern"])
    scored = []
    for variant in available:
        matched = pattern.match(variant)
        if not matched:
            continue
        value = (int(matched.group("major")), int(matched.group("minor")))
        if spec.get("major_bound") and value[0] != runtime_version[0]:
            continue
        if value > runtime_version:
            continue
        scored.append((value, variant))
    return [variant for _, variant in sorted(scored, reverse=True)]


def _chain(accelerator, available):
    ordered = []
    for entry in scheme.PREFERENCE.get(accelerator.family, [scheme.CPU_VARIANT]):
        if entry.startswith("@"):
            ordered.extend(_numeric_choices(entry[1:], accelerator.version, available))
        else:
            ordered.append(entry)
    return list(dict.fromkeys(ordered))


def resolve(accelerator: Accelerator = None) -> "Candidate | None":
    accelerator = accelerator or detect()
    catalog = _catalog()
    if not catalog:
        return None

    forced = forced_variant()
    if forced is not None:
        chain = [forced]
        logger.info(f"{scheme.ENV_VARIANT} pins the wheel variant to {forced or 'cpu'}")
    else:
        chain = _chain(accelerator, catalog.keys())

    for variant in chain:
        for version, tag, assets in catalog.get(variant, []):
            asset = _compatible_asset(assets)
            if asset is None:
                continue
            url = f"https://github.com/{scheme.REPO}/releases/download/{tag}/{asset}"
            return Candidate(url, ".".join(str(p) for p in version), variant, asset)
    return None
