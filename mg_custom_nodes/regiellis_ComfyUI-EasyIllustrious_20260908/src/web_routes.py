"""Web routes for Illustrious API endpoints"""

import json
from aiohttp import web
from server import PromptServer
from .api_handler import IllustriousAPI
import logging

# Prefer Illustrious symbol; expose as legacy name for old references
## Color Corrector server removed; replaced by Color Suite

# Ensure TIPO endpoint is registered on startup
from .nodes.tipo_server import tipo_server as _tipo_server

# Import Illustrious smart cache server to auto-register its routes
from .nodes.smart_cache_server import illustrious_smart_cache_server

# Smart Scene Generator
from .nodes.smart_prompt_generator import IllustriousSmartSceneGenerator


# Register API routes with ComfyUI's web server
@PromptServer.instance.routes.get("/illustrious/get_full_list")
async def get_full_list(request):
    """Endpoint to get paginated character/artist lists"""
    try:
        data_type = request.query.get("data_type", "characters")
        search_query = request.query.get("search", "")
        page = int(request.query.get("page", "0"))
        page_size = int(request.query.get("page_size", "200"))

        # Validate data_type
        valid_types = ["characters", "artists", "e621_characters", "e621_artists"]
        if data_type not in valid_types:
            return web.json_response(
                {
                    "success": False,
                    "error": f"Invalid data_type. Must be one of: {valid_types}",
                },
                status=400,
            )

        result = IllustriousAPI.get_paginated_list(
            data_type, search_query, page, page_size
        )
        return web.json_response(result)

    except Exception as e:
        return web.json_response(
            {"success": False, "error": f"Server error: {str(e)}"}, status=500
        )


@PromptServer.instance.routes.get("/illustrious/search")
async def search_lists(request):
    """Endpoint to search within lists"""
    try:
        data_type = request.query.get("data_type", "characters")
        query = request.query.get("query", "")
        limit = int(request.query.get("limit", "50"))

        if not query or len(query) < 2:
            return web.json_response(
                {"success": False, "error": "Query must be at least 2 characters long"},
                status=400,
            )

        # Validate data_type
        valid_types = ["characters", "artists", "e621_characters", "e621_artists"]
        if data_type not in valid_types:
            return web.json_response(
                {
                    "success": False,
                    "error": f"Invalid data_type. Must be one of: {valid_types}",
                },
                status=400,
            )

        result = IllustriousAPI.search_lists(data_type, query, limit)
        return web.json_response(result)

    except Exception as e:
        return web.json_response(
            {"success": False, "error": f"Server error: {str(e)}"}, status=500
        )


@PromptServer.instance.routes.get("/illustrious/popular")
async def get_popular_items(request):
    """Endpoint to get popular items"""
    try:
        data_type = request.query.get("data_type", "characters")
        limit = int(request.query.get("limit", "100"))

        # Validate data_type
        valid_types = ["characters", "artists", "e621_characters", "e621_artists"]
        if data_type not in valid_types:
            return web.json_response(
                {
                    "success": False,
                    "error": f"Invalid data_type. Must be one of: {valid_types}",
                },
                status=400,
            )

        result = IllustriousAPI.get_popular_items(data_type, limit)
        return web.json_response(result)

    except Exception as e:
        return web.json_response(
            {"success": False, "error": f"Server error: {str(e)}"}, status=500
        )


# Smart Prompt Generator endpoints
_smart_generator = None


def get_smart_generator():
    """Get or create smart scene generator instance."""
    global _smart_generator
    if _smart_generator is None:
        _smart_generator = IllustriousSmartSceneGenerator()
    return _smart_generator


@PromptServer.instance.routes.post("/illustrious/smart_prompt/generate")
async def generate_smart_prompt(request):
    """Generate smart anime scene prompt"""
    try:
        data = await request.json()
        generator = get_smart_generator()

        # Generate prompt using the parameters directly
        prompt, negative_prompt, metadata = generator.generate_smart_prompt(**data)

        response = web.json_response(
            {
                "success": True,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "metadata": (
                    json.loads(metadata) if isinstance(metadata, str) else metadata
                ),
            }
        )
        
        logging.debug(
            f" INFO: generate_smart_prompt\n request: {data}\n response: {response.text}"
        )

        return response 

    except Exception as e:
        logging.error(
            f" ERROR: request: {request}\n Exception: {e}"
        )
        return web.json_response(
            {"success": False, "error": f"Generation error: {str(e)}"}, status=500
        )


@PromptServer.instance.routes.get("/illustrious/smart_prompt/stats")
async def get_smart_prompt_stats(request):
    """Get smart prompt generator statistics"""
    try:
        generator = get_smart_generator()
        stats = generator.get_current_stats()

        return web.json_response({"success": True, "stats": stats})

    except Exception as e:
        return web.json_response(
            {"success": False, "error": f"Stats error: {str(e)}"}, status=500
        )


@PromptServer.instance.routes.get("/illustrious/smart_prompt/health")
async def smart_prompt_health(request):
    """Health check for smart prompt generator"""
    try:
        generator = get_smart_generator()

        # Test generation with valid args
        _ = generator.generate_smart_prompt(
            Category="Outdoor",
            Complexity="simple",
            **{
                "Include Time/Weather": True,
                "Include Ambience": True,
                "Include Event": False,
                "Include Prop": True,
                "Include Density": False,
                "Include Person Description": False,
                "Include Pose/Action": True,
                "Include Clothing": False,
                "Safe Adult Subject": True,
                "Use Chain Insert": True,
                "Strict Tags (no phrases)": True,
                "De-duplicate With Prefix/Suffix": True,
                "Danbooru Tag Style": True,
                "Enable TIPO Optimization": True,
                "TIPO Candidates": 4,
                "TIPO Flavor": "balanced",
                "TIPO Max Length": 320,
                "TIPO Seed": 1,
            },
        )

        return web.json_response(
            {
                "success": True,
                "status": "healthy",
                "test_generation": "ok",
                "components": {
                    "scene_system": "ok",
                    "composition_system": "ok",
                    "stats_system": "ok",
                },
            }
        )

    except Exception as e:
        return web.json_response(
            {"success": False, "status": "unhealthy", "error": str(e)}, status=500
        )


print("[Illustrious] API routes registered (Smart Prompt, Lists, Smart Cache, TIPO)")
