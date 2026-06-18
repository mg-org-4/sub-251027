"""
Server endpoints for TIPOPromptOptimizer (headless optimization for UI hooks)
"""

import json
from aiohttp import web
from server import PromptServer

# Simple, text-only optimizer import
from .tipo_optimizer import TIPOPromptOptimizer


class TIPOServer:
    def __init__(self):
        self.optimizer = TIPOPromptOptimizer()
        PromptServer.instance.routes.post("/illustrious/tipo/optimize")(self.optimize)
        print("🧠 TIPO server routes registered")

    async def optimize(self, request):
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            negative = data.get("negative_prompt", "") or ""
            candidates = int(data.get("candidates", 8))
            seed = int(data.get("seed", 0))
            enable_synonyms = bool(data.get("enable_synonyms", True))
            flavor = data.get("flavor", "balanced")
            max_length = int(data.get("max_length", 320))
            style_anchors = data.get(
                "style_anchors", "balanced colors, clean lineart, consistent shading"
            )

            best, cand_json, diag_json = self.optimizer.optimize(
                prompt=prompt,
                candidates=candidates,
                seed=seed,
                enable_synonyms=enable_synonyms,
                flavor=flavor,
                max_length=max_length,
                negative_prompt=negative,
                style_anchors=style_anchors,
            )

            return web.json_response(
                {
                    "optimized_prompt": best,
                    "candidates": json.loads(cand_json),
                    "diagnostics": json.loads(diag_json),
                }
            )
        except Exception as e:
            print(f"[TIPO] optimize error: {e}")
            return web.json_response({"error": str(e)}, status=500)


# Initialize
tipo_server = TIPOServer()
