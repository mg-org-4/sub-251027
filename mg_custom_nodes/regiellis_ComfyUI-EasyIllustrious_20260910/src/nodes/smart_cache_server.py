"""
Server endpoints for Illustrious Smart Cache system
Handles real-time preview adjustments and cache management
"""

import json
import base64
import io
import asyncio
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import time

from server import PromptServer
from aiohttp import web
from .smart_cache_corrector import IllustriousSmartCacheCorrector


class IllustriousSmartCacheServer:
    """Server component for smart cache management with real-time preview"""

    def __init__(self):
        # Global cache shared across all node instances
        self.global_cache = {}
        self.node_instances = {}  # Track active node instances
        self.preview_cache = {}  # Real-time preview cache
        self.settings_queue = {}  # Queued setting changes for real-time application

        # Initialize corrector for processing
        self.corrector = IllustriousSmartCacheCorrector()

        # Register server routes
        self.register_routes()

        # Background task for processing preview updates
        self.preview_update_task = None

    def register_routes(self):
        """Register all API endpoints"""
        routes = PromptServer.instance.routes

        # Real-time preview adjustment
        routes.post("/illustrious/smart_cache/adjust_preview")(self.adjust_preview)

        # Commit preview changes
        routes.post("/illustrious/smart_cache/commit_preview")(self.commit_preview)

        # Cache management
        routes.get("/illustrious/smart_cache/cache_status")(self.get_cache_status)
        routes.post("/illustrious/smart_cache/clear_cache")(self.clear_cache)

        # Node registration (called by nodes when they initialize)
        routes.post("/illustrious/smart_cache/register_node")(self.register_node)

        # Settings queue management
        routes.post("/illustrious/smart_cache/queue_settings")(
            self.queue_settings_change
        )

        print("🧠 Illustrious Smart Cache server routes registered")

    async def adjust_preview(self, request):
        """Apply setting changes to cached image in real-time"""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            setting_name = data.get("setting_name")
            setting_value = data.get("setting_value")
            current_settings = data.get("current_settings", {})

            if not node_id:
                return web.json_response({"error": "Missing node_id"}, status=400)

            print(
                f"🎨 Adjusting Illustrious preview for node {node_id}: {setting_name} = {setting_value}"
            )

            # Get cached data for this node
            if node_id not in self.global_cache:
                return web.json_response(
                    {"error": "No cached data for node"}, status=404
                )

            cached_data = self.global_cache[node_id]
            original_images = cached_data.get("original_images")
            original_analysis = cached_data.get("analysis", {})

            if not original_images:
                return web.json_response(
                    {"error": "No original images in cache"}, status=404
                )

            # Apply the setting change to cached images
            preview_result = await self.apply_settings_to_cached_images(
                original_images, original_analysis, current_settings, node_id
            )

            if preview_result:
                # Store in preview cache
                self.preview_cache[node_id] = {
                    "result": preview_result,
                    "settings": current_settings,
                    "timestamp": time.time(),
                }

                # Send preview update to frontend
                await self.send_preview_update(node_id, preview_result)

                return web.json_response(
                    {
                        "success": True,
                        "preview_image": preview_result["preview_b64"],
                        "timestamp": time.time(),
                        "model_version": current_settings.get("model_version", "auto"),
                    }
                )
            else:
                return web.json_response(
                    {"error": "Failed to generate preview"}, status=500
                )

        except Exception as e:
            print(f"[Illustrious Smart Cache] Preview adjustment error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def commit_preview(self, request):
        """Commit preview changes to main cache"""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            settings = data.get("settings", {})

            if not node_id:
                return web.json_response({"error": "Missing node_id"}, status=400)

            print(f"✅ Committing Illustrious preview changes for node {node_id}")

            # Move preview result to main cache
            if node_id in self.preview_cache:
                preview_data = self.preview_cache[node_id]

                # Update main cache with committed result
                if node_id in self.global_cache:
                    self.global_cache[node_id]["last_result"] = preview_data["result"]
                    self.global_cache[node_id]["committed_settings"] = settings
                    self.global_cache[node_id]["commit_timestamp"] = time.time()

                # Clear preview cache for this node
                del self.preview_cache[node_id]

                # Notify frontend of commit
                await self.send_commit_notification(node_id, True)

                return web.json_response(
                    {
                        "success": True,
                        "message": "Illustrious preview changes committed successfully",
                    }
                )
            else:
                return web.json_response(
                    {"error": "No preview data to commit"}, status=404
                )

        except Exception as e:
            print(f"[Illustrious Smart Cache] Commit error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def get_cache_status(self, request):
        """Get current cache status"""
        try:
            node_id = request.query.get("node_id")

            if node_id:
                # Status for specific node
                node_status = {
                    "node_id": node_id,
                    "has_cache": node_id in self.global_cache,
                    "has_preview": node_id in self.preview_cache,
                    "cache_age": None,
                    "preview_age": None,
                    "model_version": None,
                }

                if node_id in self.global_cache:
                    cache_time = self.global_cache[node_id].get("timestamp", 0)
                    node_status["cache_age"] = time.time() - cache_time
                    node_status["model_version"] = (
                        self.global_cache[node_id]
                        .get("committed_settings", {})
                        .get("model_version", "auto")
                    )

                if node_id in self.preview_cache:
                    preview_time = self.preview_cache[node_id].get("timestamp", 0)
                    node_status["preview_age"] = time.time() - preview_time

                return web.json_response(node_status)
            else:
                # Global status
                global_status = {
                    "total_cached_nodes": len(self.global_cache),
                    "active_previews": len(self.preview_cache),
                    "registered_nodes": len(self.node_instances),
                    "cache_memory_estimate": self.estimate_cache_memory_usage(),
                    "oldest_cache_age": self.get_oldest_cache_age(),
                    "newest_cache_age": self.get_newest_cache_age(),
                    "illustrious_optimized": True,
                }

                return web.json_response(global_status)

        except Exception as e:
            print(f"[Illustrious Smart Cache] Status query error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def clear_cache(self, request):
        """Clear cache data"""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            cache_type = data.get("cache_type", "all")  # 'all', 'main', 'preview'

            cleared = 0

            if node_id:
                # Clear specific node
                if cache_type in ["all", "main"] and node_id in self.global_cache:
                    del self.global_cache[node_id]
                    cleared += 1

                if cache_type in ["all", "preview"] and node_id in self.preview_cache:
                    del self.preview_cache[node_id]
                    cleared += 1

                message = (
                    f"Cleared {cleared} Illustrious cache entries for node {node_id}"
                )

            else:
                # Clear all cache
                if cache_type in ["all", "main"]:
                    cleared += len(self.global_cache)
                    self.global_cache.clear()

                if cache_type in ["all", "preview"]:
                    cleared += len(self.preview_cache)
                    self.preview_cache.clear()

                message = f"Cleared {cleared} total Illustrious cache entries"

            print(f"🗑️ Illustrious cache cleared: {message}")

            return web.json_response(
                {"success": True, "message": message, "cleared_count": cleared}
            )

        except Exception as e:
            print(f"[Illustrious Smart Cache] Clear cache error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def register_node(self, request):
        """Register a node instance for cache management"""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            node_config = data.get("config", {})

            if not node_id:
                return web.json_response({"error": "Missing node_id"}, status=400)

            self.node_instances[node_id] = {
                "config": node_config,
                "registered_time": time.time(),
                "last_activity": time.time(),
                "illustrious_optimized": True,
            }

            print(
                f"📝 Registered Illustrious node {node_id} for smart cache management"
            )

            return web.json_response(
                {
                    "success": True,
                    "message": f"Illustrious node {node_id} registered successfully",
                }
            )

        except Exception as e:
            print(f"[Illustrious Smart Cache] Node registration error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def queue_settings_change(self, request):
        """Queue settings change for batch processing"""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            settings_changes = data.get("changes", {})
            priority = data.get("priority", "normal")  # 'low', 'normal', 'high'

            if not node_id:
                return web.json_response({"error": "Missing node_id"}, status=400)

            # Add to settings queue
            if node_id not in self.settings_queue:
                self.settings_queue[node_id] = []

            self.settings_queue[node_id].append(
                {
                    "changes": settings_changes,
                    "priority": priority,
                    "timestamp": time.time(),
                    "processed": False,
                    "illustrious_specific": True,
                }
            )

            # Start background processing if not already running
            if not self.preview_update_task or self.preview_update_task.done():
                self.preview_update_task = asyncio.create_task(
                    self.process_settings_queue()
                )

            return web.json_response(
                {
                    "success": True,
                    "queued": len(self.settings_queue[node_id]),
                    "message": "Illustrious settings change queued for processing",
                }
            )

        except Exception as e:
            print(f"[Illustrious Smart Cache] Queue settings error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def apply_settings_to_cached_images(
        self, original_images, analysis, settings, node_id
    ):
        """Apply settings to cached images and return preview result"""
        try:
            # Use the corrector's cached processing method
            corrected_batch, report = await asyncio.get_event_loop().run_in_executor(
                None,
                self.corrector._apply_corrections_to_cached,
                original_images,
                analysis,
                settings,
            )

            # Convert first corrected image to base64 for preview
            if len(corrected_batch) > 0:
                first_image = corrected_batch[0]
                image_np = first_image.cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

                pil_image = Image.fromarray(image_np)

                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                preview_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                return {
                    "corrected_batch": corrected_batch,
                    "report": report,
                    "preview_b64": preview_b64,
                    "settings_applied": settings,
                    "model_version": settings.get("model_version", "auto"),
                }

        except Exception as e:
            print(f"[Illustrious Smart Cache] Apply settings error: {e}")
            return {
                "error": str(e),
                "corrected_batch": None,
                "report": f"Error applying settings: {e}",
                "preview_b64": None,
                "settings_applied": settings,
                "model_version": settings.get("model_version", "auto"),
            }

    async def send_preview_update(self, node_id, result):
        """Send preview update to frontend"""
        try:
            PromptServer.instance.send_sync(
                "illustrious.smart_cache_preview_update",
                {
                    "node_id": node_id,
                    "preview_image": result["preview_b64"],
                    "timestamp": time.time(),
                    "model_version": result.get("model_version", "auto"),
                },
            )
        except Exception as e:
            print(f"[Illustrious Smart Cache] Preview update send error: {e}")

    async def send_commit_notification(self, node_id, success):
        """Notify frontend of commit result"""
        try:
            PromptServer.instance.send_sync(
                "illustrious.smart_cache_commit",
                {"node_id": node_id, "success": success, "timestamp": time.time()},
            )
        except Exception as e:
            print(f"[Illustrious Smart Cache] Commit notification error: {e}")

    async def process_settings_queue(self):
        """Background task to process queued settings changes"""
        print("🔄 Starting Illustrious settings queue processor")

        try:
            while True:
                # Process all queued changes
                processed_any = False

                for node_id, queue in list(self.settings_queue.items()):
                    if not queue:
                        continue

                    # Process highest priority item first
                    queue.sort(
                        key=lambda x: {"high": 0, "normal": 1, "low": 2}[x["priority"]]
                    )

                    for item in queue:
                        if not item["processed"]:
                            await self.process_single_settings_change(node_id, item)
                            item["processed"] = True
                            processed_any = True
                            break  # Process one item per node per cycle

                    # Remove processed items
                    self.settings_queue[node_id] = [
                        item for item in queue if not item["processed"]
                    ]

                    if not self.settings_queue[node_id]:
                        del self.settings_queue[node_id]

                if not processed_any and not self.settings_queue:
                    # No more work to do
                    break

                # Brief pause between processing cycles
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"[Illustrious Smart Cache] Settings queue processor error: {e}")

    async def process_single_settings_change(self, node_id, change_item):
        """Process a single settings change"""
        try:
            changes = change_item["changes"]
            print(
                f"🔄 Processing Illustrious settings change for node {node_id}: {changes}"
            )

            # Apply changes and generate preview
            if node_id in self.global_cache:
                cached_data = self.global_cache[node_id]
                current_settings = cached_data.get("committed_settings", {})

                # Merge changes into current settings
                updated_settings = {**current_settings, **changes}

                # Apply to cached images
                result = await self.apply_settings_to_cached_images(
                    cached_data.get("original_images"),
                    cached_data.get("analysis", {}),
                    updated_settings,
                    node_id,
                )

                if result:
                    # Update preview cache
                    self.preview_cache[node_id] = {
                        "result": result,
                        "settings": updated_settings,
                        "timestamp": time.time(),
                    }

                    # Send update to frontend
                    await self.send_preview_update(node_id, result)

        except Exception as e:
            print(f"[Illustrious Smart Cache] Single settings change error: {e}")

    def estimate_cache_memory_usage(self):
        """Estimate memory usage of cache in MB"""
        total_size = 0

        for node_id, cache_data in self.global_cache.items():
            if "original_images" in cache_data:
                for image_tensor in cache_data["original_images"]:
                    if hasattr(image_tensor, "element_size"):
                        total_size += (
                            image_tensor.element_size() * image_tensor.nelement()
                        )

        for node_id, preview_data in self.preview_cache.items():
            if "result" in preview_data and "corrected_batch" in preview_data["result"]:
                batch = preview_data["result"]["corrected_batch"]
                for image_tensor in batch:
                    if hasattr(image_tensor, "element_size"):
                        total_size += (
                            image_tensor.element_size() * image_tensor.nelement()
                        )

        return round(total_size / (1024 * 1024), 2)  # Convert to MB

    def get_oldest_cache_age(self):
        """Get age of oldest cache entry in seconds"""
        if not self.global_cache:
            return 0  # Return 0 instead of None for empty cache

        oldest_time = min(
            data.get("timestamp", time.time()) for data in self.global_cache.values()
        )
        return time.time() - oldest_time

    def get_newest_cache_age(self):
        """Get age of newest cache entry in seconds"""
        if not self.global_cache:
            return 0  # Return 0 instead of None for empty cache

        newest_time = max(
            data.get("timestamp", 0) for data in self.global_cache.values()
        )
        return time.time() - newest_time

    def update_node_cache(self, node_id, cache_data):
        """Update cache for a specific node (called by node instances)"""
        self.global_cache[node_id] = {
            **cache_data,
            "timestamp": time.time(),
            "illustrious_optimized": True,
        }

        # Update node activity
        if node_id in self.node_instances:
            self.node_instances[node_id]["last_activity"] = time.time()


# Global server instance
illustrious_smart_cache_server = IllustriousSmartCacheServer()


# Enhanced node that integrates with the smart cache server
class IllustriousSmartCacheIntegratedCorrector(IllustriousSmartCacheCorrector):
    """Enhanced corrector with server integration"""

    def __init__(self):
        super().__init__()
        self._node_id = None

    def auto_correct_colors(self, *args, **kwargs):
        """Enhanced version with server integration"""
        # Call parent method
        result = super().auto_correct_colors(*args, **kwargs)

        # Update server cache if we have a node ID
        if hasattr(self, "_node_id") and self._node_id:
            # Extract cache data from result
            cache_data = {
                "original_images": (
                    args[0] if args else None
                ),  # First argument is images
                "last_result": result[0],  # Corrected images
                "last_report": result[1],  # Report
                "analysis": {},  # Would need to extract from processing
                "cache_used": result[2] if len(result) > 2 else False,
            }

            illustrious_smart_cache_server.update_node_cache(self._node_id, cache_data)

        return result

    def set_node_id(self, node_id):
        """Set the node ID for server integration"""
        self._node_id = node_id


print("🧠 Illustrious Smart Cache server integration loaded")
