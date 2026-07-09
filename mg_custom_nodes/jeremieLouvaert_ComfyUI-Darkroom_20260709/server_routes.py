"""
Backend HTTP routes for ComfyUI-Darkroom.
Registers aiohttp endpoints on ComfyUI's PromptServer for the folder picker
used by LUT Export.
"""

import json
import os
import string

from aiohttp import web
from server import PromptServer

try:
    import folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False


def _norm(path):
    return path.replace("\\", "/") if path else path


# --- Pinned folders (cross-node quick-access) -----------------------------
#
# User-added favourites, shared by every node that opens the path picker.
# Lives under ComfyUI's user/ dir so it survives ComfyUI-Darkroom updates
# and follows the user's settings-follow-dotfiles workflow if any.

_PINS_REL_PATH = os.path.join("default", "darkroom", "pinned_folders.json")


def _pins_file():
    if not _HAS_FOLDER_PATHS:
        return None
    try:
        return os.path.join(folder_paths.get_user_directory(), _PINS_REL_PATH)
    except Exception:
        return None


def _load_pins():
    p = _pins_file()
    if not p or not os.path.isfile(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("path"):
            path = _norm(str(item["path"]))
            name = str(item.get("name") or os.path.basename(path.rstrip("/")) or path)
            out.append({"name": name, "path": path})
        elif isinstance(item, str) and item:
            path = _norm(item)
            out.append({"name": os.path.basename(path.rstrip("/")) or path, "path": path})
    return out


def _save_pins(pins):
    p = _pins_file()
    if not p:
        return False
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pins, f, indent=2)
    return True


def _get_roots():
    roots = []

    home = os.path.expanduser("~")
    if home and os.path.isdir(home):
        roots.append({"name": "Home", "path": _norm(home)})

    if _HAS_FOLDER_PATHS:
        try:
            out = folder_paths.get_output_directory()
            roots.append({"name": "ComfyUI output", "path": _norm(out)})
            lut_dir = os.path.join(out, "luts")
            roots.append({"name": "Darkroom LUTs", "path": _norm(lut_dir)})
        except Exception:
            pass

    if os.name == "nt":
        for c in string.ascii_uppercase:
            drive = f"{c}:/"
            if os.path.exists(drive):
                roots.append({"name": f"{c}: drive", "path": drive})
    else:
        roots.append({"name": "/", "path": "/"})

    return roots


def _list_entries(path, extensions=None):
    subdirs, files = [], []
    with os.scandir(path) as it:
        for entry in it:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not entry.name.startswith("."):
                        subdirs.append({
                            "name": entry.name,
                            "path": _norm(entry.path),
                        })
                elif extensions and entry.is_file(follow_symlinks=False):
                    name_lower = entry.name.lower()
                    if any(name_lower.endswith(ext) for ext in extensions):
                        files.append({
                            "name": entry.name,
                            "path": _norm(entry.path),
                        })
            except (PermissionError, OSError):
                continue
    subdirs.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["name"].lower())
    return subdirs, files


def _parent_of(path):
    if not path:
        return ""
    stripped = path.rstrip("/\\")
    parent = os.path.dirname(stripped)
    if not parent or parent == stripped:
        return ""
    if os.name == "nt" and len(parent) == 2 and parent[1] == ":":
        parent = parent + "/"
    return _norm(parent)


def _parse_extensions(raw):
    if not raw:
        return None
    parts = [e.strip().lower() for e in raw.split(",") if e.strip()]
    if not parts:
        return None
    return tuple(e if e.startswith(".") else "." + e for e in parts)


@PromptServer.instance.routes.get("/darkroom/list_dir")
async def darkroom_list_dir(request):
    raw_path = request.query.get("path", "").strip()
    extensions = _parse_extensions(request.query.get("extensions", ""))
    path = _norm(raw_path) if raw_path else ""

    if path and os.path.isfile(path):
        path = _norm(os.path.dirname(path))

    if not path and _HAS_FOLDER_PATHS:
        try:
            lut_dir = os.path.join(folder_paths.get_output_directory(), "luts")
            if os.path.isdir(lut_dir):
                path = _norm(lut_dir)
            else:
                path = _norm(folder_paths.get_output_directory())
        except Exception:
            pass

    roots = _get_roots()

    if not path or not os.path.isdir(path):
        return web.json_response({
            "path": "",
            "parent": "",
            "subdirs": [],
            "files": [],
            "roots": roots,
            "writable": False,
        })

    try:
        subdirs, files = _list_entries(path, extensions)
    except (PermissionError, OSError, FileNotFoundError) as e:
        return web.json_response({
            "path": path,
            "parent": _parent_of(path),
            "subdirs": [],
            "files": [],
            "roots": roots,
            "writable": False,
            "error": f"{type(e).__name__}: {e}",
        })

    return web.json_response({
        "path": path,
        "parent": _parent_of(path),
        "subdirs": subdirs,
        "files": files,
        "roots": roots,
        "writable": os.access(path, os.W_OK),
    })


@PromptServer.instance.routes.post("/darkroom/mkdir")
async def darkroom_mkdir(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON body"}, status=400)

    path = (data.get("path") or "").strip()
    if not path:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)

    path = _norm(path)

    try:
        os.makedirs(path, exist_ok=False)
    except FileExistsError:
        return web.json_response({"ok": False, "error": "Folder already exists"}, status=400)
    except (PermissionError, OSError) as e:
        return web.json_response(
            {"ok": False, "error": f"{type(e).__name__}: {e}"},
            status=500,
        )

    return web.json_response({"ok": True, "path": path})


@PromptServer.instance.routes.get("/darkroom/pins")
async def darkroom_pins_get(_request):
    return web.json_response({"ok": True, "pins": _load_pins()})


@PromptServer.instance.routes.post("/darkroom/pins")
async def darkroom_pins_add(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON body"}, status=400)

    raw_path = (data.get("path") or "").strip()
    if not raw_path:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)
    path = _norm(raw_path)
    if not os.path.isdir(path):
        return web.json_response({"ok": False, "error": "Path is not a directory"}, status=400)

    name = (data.get("name") or os.path.basename(path.rstrip("/")) or path).strip()
    pins = _load_pins()
    if any(p["path"].lower() == path.lower() for p in pins):
        return web.json_response({"ok": True, "pins": pins, "duplicate": True})
    pins.append({"name": name, "path": path})
    _save_pins(pins)
    return web.json_response({"ok": True, "pins": pins})


@PromptServer.instance.routes.delete("/darkroom/pins")
async def darkroom_pins_remove(request):
    raw_path = (request.query.get("path") or "").strip()
    if not raw_path:
        return web.json_response({"ok": False, "error": "Missing path"}, status=400)
    path = _norm(raw_path).lower()
    pins = _load_pins()
    new_pins = [p for p in pins if p["path"].lower() != path]
    _save_pins(new_pins)
    return web.json_response({"ok": True, "pins": new_pins})


@PromptServer.instance.routes.get("/darkroom/camera_looks")
async def darkroom_camera_looks(request):
    """Return the prettified look names for one brand, used by the RAW Load
    node's frontend to narrow the camera_look combo after the brand changes."""
    brand = (request.query.get("brand") or "").strip()
    try:
        from .utils.dcp import list_looks_for_brand
        looks = list_looks_for_brand(brand)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"{type(e).__name__}: {e}"}, status=500)
    return web.json_response({"ok": True, "brand": brand, "looks": looks})


print("[ComfyUI-Darkroom] registered HTTP routes: /darkroom/list_dir, /darkroom/mkdir, "
      "/darkroom/camera_looks, /darkroom/pins (GET/POST/DELETE)")
