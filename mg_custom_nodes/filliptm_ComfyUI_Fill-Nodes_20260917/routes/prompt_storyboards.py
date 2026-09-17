import asyncio
import io

from PIL import Image, ImageOps

from aiohttp import web
from server import PromptServer
import execution

from ..nodes.audio.prompt_storyboards import storyboard_store, storyboard_graph, storyboard_batch_graph, extract_panels
from ..nodes.audio.prompt_references import reference_path


PREFIX = "/fl/audio-prompt-timeline/storyboards"


def thumbnail_bytes(path):
    with Image.open(path) as source:
        source.thumbnail((256, 256))
        image = ImageOps.exif_transpose(source).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="WEBP", quality=78)
        return output.getvalue()


@PromptServer.instance.routes.get(PREFIX + "/thumbnail")
async def storyboard_thumbnail(request):
    try:
        path = reference_path(dict(request.query))
        stat = path.stat()
        etag = f'"{stat.st_mtime_ns}-{stat.st_size}-256"'
        headers = {"ETag": etag, "Cache-Control": "private, max-age=0, must-revalidate"}
        if request.headers.get("If-None-Match") == etag:
            return web.Response(status=304, headers=headers)
        return web.Response(body=await asyncio.to_thread(thumbnail_bytes, path), content_type="image/webp", headers=headers)
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.get(PREFIX)
async def list_storyboards(request):
    return web.json_response(await asyncio.to_thread(storyboard_store.list, request.query.get("scheduler_id", "")))


@PromptServer.instance.routes.post(PREFIX)
async def create_storyboard(request):
    try:
        result = await asyncio.to_thread(storyboard_store.create, await request.json())
        return web.json_response(result)
    except (ValueError, TypeError, KeyError) as error:
        return web.json_response({"error": str(error)}, status=400)


@PromptServer.instance.routes.post(PREFIX + "/{job_id}/{action}")
async def storyboard_action(request):
    try:
        job_id, action = request.match_info["job_id"], request.match_info["action"]
        if job_id == "batch" and action == "submit":
            ids = (await request.json()).get("job_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(i,str) for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("Submit a nonempty list of distinct storyboard jobs.")
            jobs = [await asyncio.to_thread(storyboard_store.get, i) for i in ids]
            if len({j["spec"]["scheduler_id"] for j in jobs}) != 1:
                raise ValueError("A storyboard batch must belong to one scheduler.")
            graph = storyboard_batch_graph(jobs)
            valid, error, _outputs, node_errors = await execution.validate_prompt("storyboard_batch", graph, None)
            if not valid:
                return web.json_response({"error": "Storyboard batch validation failed", "details": error, "node_errors": node_errors}, status=400)
            await asyncio.to_thread(storyboard_store.claim_batch, ids)
            return web.json_response({"graph": graph})
        job = await asyncio.to_thread(storyboard_store.get, job_id)
        if action in {"validate", "submit"}:
            graph = storyboard_graph(job_id, job["spec"])
            valid, error, _outputs, node_errors = await execution.validate_prompt(job_id, graph, None)
            if not valid:
                return web.json_response({"error": "Storyboard graph validation failed", "details": error, "node_errors": node_errors}, status=400)
            if action == "validate":
                return web.json_response({"valid": True})
            graph = await asyncio.to_thread(storyboard_store.claim, job_id)
            return web.json_response({"graph": graph})
        if action == "receipt":
            body = await request.json()
            prompt_id = body.get("prompt_id")
            if job["state"] not in {"submitted", "unknown"} or not isinstance(prompt_id, str) or len(prompt_id) > 128:
                raise ValueError("Invalid storyboard queue receipt.")
            return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "submitted", {"prompt_id": prompt_id}))
        if action == "refresh":
            if job["state"] in {"proposed", "cancelled", "complete"}:
                return web.json_response(job)
            history = PromptServer.instance.prompt_queue.get_history()
            for prompt_id, entry in history.items():
                images = [image for output in entry.get("outputs", {}).values() for image in output.get("images", [])]
                source = next((image for image in images if image.get("subfolder", "").replace("\\", "/") == f"fl-storyboards/{job_id}"), None)
                if source:
                    return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "complete", {**job["result"], "source": source, "prompt_id": prompt_id}))
            prompt_id = job["result"].get("prompt_id")
            if prompt_id in history:
                return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "failed", {**job["result"], "error": "No contact sheet was saved. Check ComfyUI history; charges may apply."}))
            running, pending = PromptServer.instance.prompt_queue.get_current_queue()
            for item in running + pending:
                graph = item[2]
                if any(node.get("inputs", {}).get("filename_prefix") == f"fl-storyboards/{job_id}/sheet" for node in graph.values()):
                    return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "submitted", {"prompt_id": item[1]}))
            return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "unknown", {**job["result"], "error": "No queue or history record found. Check saved outputs before submitting a new paid job."}))
        if action == "extract":
            if job["state"] != "complete" or not job["result"].get("source"):
                raise ValueError("Wait for a completed contact sheet before extracting panels.")
            body = await request.json()
            result = await asyncio.to_thread(extract_panels, job, job["result"]["source"], body.get("bounds"))
            return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "complete", result))
        if action == "cancel":
            if job["state"] != "proposed":
                raise ValueError("Submitted jobs may already be billed. Use ComfyUI's queue to cancel pending work.")
            return web.json_response(await asyncio.to_thread(storyboard_store.update, job_id, "cancelled", {}))
        raise ValueError("Unknown storyboard action.")
    except (ValueError, TypeError, KeyError) as error:
        return web.json_response({"error": str(error)}, status=400)
