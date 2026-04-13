"""
Backend HTTP routes for ComfyUI-Darkroom.
Registers aiohttp endpoints on ComfyUI's PromptServer for the folder picker
used by LUT Export.
"""

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


print("[ComfyUI-Darkroom] registered HTTP routes: /darkroom/list_dir, /darkroom/mkdir")
