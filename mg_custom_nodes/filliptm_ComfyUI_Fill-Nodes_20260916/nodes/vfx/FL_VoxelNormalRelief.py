"""Screen-aligned normal-color cuboids with animated depth relief."""
import math

import cv2
import numpy as np
import torch

from comfy.utils import ProgressBar


class FL_VoxelNormalRelief:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "normals": ("IMAGE",), "depth": ("IMAGE",),
            "cube_size": ("INT", {"default": 12, "min": 4, "max": 64}),
            "relief": ("FLOAT", {"default": 0.65, "min": 0, "max": 2, "step": 0.05}),
            "animation": ("FLOAT", {"default": 0.18, "min": 0, "max": 1, "step": 0.01}),
            "speed": ("FLOAT", {"default": 0.7, "min": 0, "max": 5, "step": 0.05}),
            "fps": ("FLOAT", {"default": 24, "min": 1, "max": 120}),
            "seed": ("INT", {"default": 41, "min": 0, "max": 2147483647}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("voxel_normals",)
    FUNCTION = "render"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "Renders normal-colored cuboids at their source-image positions. Depth drives height; a smooth seeded wave adds subtle motion. Connect before the scan compositor's depth projection. This is a 2.5D image effect, not a voxel mesh."

    def render(self, normals, depth, cube_size, relief, animation, speed, fps, seed):
        return self.render_animated(normals, depth, cube_size, relief, animation, speed, fps, seed)

    def render_animated(self, normals, depth, cube_size, relief, animation, speed, fps, seed, frame_values=None):
        if normals.ndim != 4 or normals.shape[-1] != 3 or depth.ndim != 4 or depth.shape[:3] != normals.shape[:3] or depth.shape[-1] < 1:
            raise ValueError("Voxel Normal Relief needs aligned RGB normals and depth frames at the same resolution.")
        if cube_size < 4 or fps <= 0 or not all(math.isfinite(v) and v >= 0 for v in (relief, animation, speed, fps)):
            raise ValueError("Voxel Normal Relief requires cube size >= 4, positive FPS and finite nonnegative controls.")
        count, height, width = normals.shape[:3]
        output = np.empty((count, height, width, 3), np.float32)
        rows, cols = math.ceil(height / cube_size), math.ceil(width / cube_size)
        gap = max(1, round(cube_size * .22))
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, math.tau, (rows, cols))
        yy, xx = np.mgrid[:rows, :cols]
        x, y = xx * cube_size, yy * cube_size
        right, bottom = np.minimum(width, x + cube_size - gap), np.minimum(height, y + cube_size - gap)
        bases = np.stack((np.stack((x, y), -1), np.stack((right, y), -1),
                          np.stack((right, bottom), -1), np.stack((x, bottom), -1)), -2).astype(np.int32)
        progress = ProgressBar(count)
        phase = 0.0
        for frame in range(count):
            normal = normals[frame].cpu().float().numpy()
            d = depth[frame, :, :, 0].cpu().float().numpy()
            normal = cv2.copyMakeBorder(normal, 0, rows * cube_size - height, 0, cols * cube_size - width, cv2.BORDER_REPLICATE)
            d = cv2.copyMakeBorder(d, 0, rows * cube_size - height, 0, cols * cube_size - width, cv2.BORDER_REPLICATE)
            colors = cv2.resize(normal, (cols, rows), interpolation=cv2.INTER_AREA).clip(0, 1)
            levels = cv2.resize(d, (cols, rows), interpolation=cv2.INTER_AREA).clip(0, 1)
            if frame_values is None:
                phase = frame / fps * speed * math.tau
            else:
                relief = frame_values["relief"][frame]
                animation = frame_values["animation"][frame]
            wave = np.sin(phase + xx * .42 + yy * .31 + phases * .25)
            heights = cube_size * relief * np.maximum(.05, .3 + .6 * levels + animation * wave)
            if frame_values is not None:
                phase += frame_values["speed"][frame] / fps * math.tau
            image = np.zeros((height, width, 3), np.float32)
            offsets = np.stack((-heights * .45, -heights * .65), -1)[..., None, :]
            tops = np.rint(bases + offsets).astype(np.int32)
            sides = np.stack((tops[..., 1, :], bases[..., 1, :], bases[..., 2, :], tops[..., 2, :]), -2).reshape(-1, 4, 2)
            fronts = np.stack((tops[..., 3, :], tops[..., 2, :], bases[..., 2, :], bases[..., 3, :]), -2).reshape(-1, 4, 2)
            tops = tops.reshape(-1, 4, 2)
            colors = colors.reshape(-1, 3)
            side_colors, front_colors = (colors * .48).tolist(), (colors * .68).tolist()
            line_colors, highlight_colors = (colors * .38).tolist(), (colors * .75 + .2).tolist()
            colors = colors.tolist()
            edges = tops[:, :2].tolist()
            # Front rows and right-hand cells cover the sides of cubes behind them.
            for top, side, front, edge, color, side_color, front_color, line_color, highlight in zip(
                    tops, sides, fronts, edges, colors, side_colors, front_colors, line_colors, highlight_colors):
                cv2.fillConvexPoly(image, side, side_color, cv2.LINE_8)
                cv2.fillConvexPoly(image, front, front_color, cv2.LINE_8)
                cv2.fillConvexPoly(image, top, color, cv2.LINE_8)
                cv2.polylines(image, [top], True, line_color, 1, cv2.LINE_8)
                cv2.line(image, edge[0], edge[1], highlight, 1, cv2.LINE_8)
            output[frame] = image.clip(0, 1)
            progress.update(1)
        return (torch.from_numpy(output),)
