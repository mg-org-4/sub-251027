# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generate the public DiffusionGemma grounding benchmark fixtures.

This generator is deliberately offline and deterministic.  It uses only the
Python standard library, writes simple RGB PNG/APNG files, and never creates
benchmark result files.  APNG was selected for the synthetic video cases
because Pillow (and therefore normal ComfyUI image tooling) can read every
frame without an external MP4 encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


WIDTH = 256
HEIGHT = 192
FRAME_COUNT = 6
FRAME_SECONDS = 0.25
SEEDS = [17, 101, 809]
SCHEMA_VERSION = "dg-grounding-benchmark/1"

RGB = tuple[int, int, int]
Point = tuple[int, int]


WHITE: RGB = (250, 250, 247)
BLACK: RGB = (20, 22, 25)
GRAY: RGB = (128, 132, 136)
LIGHT_GRAY: RGB = (224, 227, 229)
RED: RGB = (220, 45, 48)
BLUE: RGB = (36, 93, 214)
GREEN: RGB = (38, 156, 82)
YELLOW: RGB = (242, 193, 46)
ORANGE: RGB = (235, 124, 35)
VIOLET: RGB = (139, 67, 191)
CYAN: RGB = (33, 190, 202)
TEAL: RGB = (28, 145, 142)
BROWN: RGB = (143, 86, 45)


FONT_5X7 = {
    " ": ("00000",) * 7,
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
}


class Canvas:
    """Tiny deterministic RGB rasterizer for the benchmark primitives."""

    def __init__(self, color: RGB = WHITE) -> None:
        self.pixels = bytearray(color * (WIDTH * HEIGHT))

    def set(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            offset = (y * WIDTH + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def rectangle(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        left = max(0, min(x0, x1))
        right = min(WIDTH - 1, max(x0, x1))
        top = max(0, min(y0, y1))
        bottom = min(HEIGHT - 1, max(y0, y1))
        if left > right or top > bottom:
            return
        row = bytes(color) * (right - left + 1)
        for y in range(top, bottom + 1):
            offset = (y * WIDTH + left) * 3
            self.pixels[offset : offset + len(row)] = row

    def circle(self, cx: int, cy: int, radius: int, color: RGB) -> None:
        radius_squared = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            dy_squared = (y - cy) * (y - cy)
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) * (x - cx) + dy_squared <= radius_squared:
                    self.set(x, y, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: RGB, thickness: int = 1) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        radius = max(0, thickness // 2)
        while True:
            self.rectangle(x0 - radius, y0 - radius, x0 + radius, y0 + radius, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def polygon(self, points: list[Point], color: RGB) -> None:
        min_y = max(0, min(y for _, y in points))
        max_y = min(HEIGHT - 1, max(y for _, y in points))
        for y in range(min_y, max_y + 1):
            intersections: list[float] = []
            previous = points[-1]
            for current in points:
                x1, y1 = previous
                x2, y2 = current
                if (y1 <= y < y2) or (y2 <= y < y1):
                    intersections.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
                previous = current
            intersections.sort()
            for index in range(0, len(intersections) - 1, 2):
                self.rectangle(round(intersections[index]), y, round(intersections[index + 1]), y, color)

    def text(self, value: str, x: int, y: int, color: RGB, scale: int = 4) -> None:
        cursor = x
        for character in value:
            glyph = FONT_5X7[character]
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "1":
                        left = cursor + column_index * scale
                        top = y + row_index * scale
                        self.rectangle(left, top, left + scale - 1, top + scale - 1, color)
            cursor += 6 * scale

    def bytes(self) -> bytes:
        return bytes(self.pixels)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _compressed_scanlines(rgb: bytes) -> bytes:
    stride = WIDTH * 3
    raw = b"".join(b"\x00" + rgb[offset : offset + stride] for offset in range(0, len(rgb), stride))
    return zlib.compress(raw, level=9)


def png_bytes(rgb: bytes) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", _compressed_scanlines(rgb)) + _chunk(b"IEND", b"")


def apng_bytes(frames: list[bytes]) -> bytes:
    if len(frames) < 2:
        raise ValueError("APNG fixtures require at least two frames.")
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    output = bytearray(signature + _chunk(b"IHDR", ihdr))
    output.extend(_chunk(b"acTL", struct.pack(">II", len(frames), 0)))
    sequence = 0
    for index, frame in enumerate(frames):
        frame_control = struct.pack(
            ">IIIIIHHBB",
            sequence,
            WIDTH,
            HEIGHT,
            0,
            0,
            1,
            4,
            0,
            0,
        )
        output.extend(_chunk(b"fcTL", frame_control))
        sequence += 1
        compressed = _compressed_scanlines(frame)
        if index == 0:
            output.extend(_chunk(b"IDAT", compressed))
        else:
            output.extend(_chunk(b"fdAT", struct.pack(">I", sequence) + compressed))
            sequence += 1
    output.extend(_chunk(b"IEND", b""))
    return bytes(output)


def _star(cx: int, cy: int, outer: int, inner: int) -> list[Point]:
    # Integer points for a visually obvious five-pointed star.
    unit = [
        (0, -1000),
        (224, -309),
        (951, -309),
        (363, 118),
        (588, 809),
        (0, 382),
        (-588, 809),
        (-363, 118),
        (-951, -309),
        (-224, -309),
    ]
    points: list[Point] = []
    for index, (x, y) in enumerate(unit):
        radius = outer if index % 2 == 0 else inner
        points.append((cx + x * radius // 1000, cy + y * radius // 1000))
    return points


def _image_fixtures() -> dict[str, bytes]:
    fixtures: dict[str, bytes] = {}

    canvas = Canvas()
    canvas.rectangle(28, 62, 92, 126, RED)
    canvas.circle(182, 94, 34, BLUE)
    fixtures["fixtures/images/img_01_shape_real.png"] = png_bytes(canvas.bytes())

    canvas = Canvas(GRAY)
    canvas.rectangle(20, 91, 235, 100, (142, 146, 150))
    fixtures["fixtures/images/img_02_shape_neutral.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    canvas.polygon([(128, 24), (82, 91), (174, 91)], GREEN)
    canvas.polygon([(128, 105), (166, 143), (128, 181), (90, 143)], YELLOW)
    fixtures["fixtures/images/img_03_shape_unrelated.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    for x in (62, 128, 194):
        canvas.circle(x, 96, 25, ORANGE)
    fixtures["fixtures/images/img_04_count_three.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    canvas.rectangle(35, 45, 220, 147, VIOLET)
    canvas.rectangle(101, 69, 154, 123, CYAN)
    fixtures["fixtures/images/img_05_color_pair.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    canvas.rectangle(18, 53, 238, 139, LIGHT_GRAY)
    canvas.text("NORTH 7", 42, 78, BLACK, scale=4)
    fixtures["fixtures/images/img_06_text_north7.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    canvas.circle(128, 52, 28, RED)
    canvas.rectangle(78, 118, 178, 158, BLUE)
    fixtures["fixtures/images/img_07_spatial_above.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    canvas.rectangle(55, 48, 155, 149, YELLOW)
    canvas.circle(159, 102, 53, TEAL)
    fixtures["fixtures/images/img_08_occlusion.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    canvas.polygon(_star(78, 97, 60, 28), BLACK)
    canvas.polygon(_star(197, 103, 25, 12), RED)
    fixtures["fixtures/images/img_09_size_holdout.png"] = png_bytes(canvas.bytes())

    canvas = Canvas()
    colors = (RED, GREEN, BLUE, YELLOW)
    positions = ((45, 38), (145, 38), (45, 108), (145, 108))
    for color, (x, y) in zip(colors, positions):
        canvas.rectangle(x, y, x + 65, y + 45, color)
    fixtures["fixtures/images/img_10_grid_holdout.png"] = png_bytes(canvas.bytes())
    return fixtures


def _motion_frames(positions: list[int]) -> list[bytes]:
    frames: list[bytes] = []
    for x in positions:
        canvas = Canvas()
        for grid_x in (32, 80, 128, 176, 224):
            canvas.line(grid_x, 22, grid_x, 170, LIGHT_GRAY, 2)
        canvas.line(20, 96, 236, 96, BLACK, 3)
        canvas.circle(x, 96, 18, RED)
        frames.append(canvas.bytes())
    return frames


def _box_opening_frames() -> list[bytes]:
    frames: list[bytes] = []
    lid_endpoints = [(190, 68), (184, 59), (176, 49), (166, 39), (154, 31), (140, 25)]
    for endpoint in lid_endpoints:
        canvas = Canvas()
        canvas.rectangle(66, 72, 190, 158, BROWN)
        canvas.rectangle(73, 81, 183, 151, (184, 126, 72))
        canvas.line(66, 71, endpoint[0], endpoint[1], BLACK, 7)
        canvas.circle(66, 71, 5, BLACK)
        frames.append(canvas.bytes())
    return frames


def _camera_pan_frames() -> list[bytes]:
    frames: list[bytes] = []
    for frame_index in range(FRAME_COUNT):
        canvas = Canvas((235, 239, 230))
        shift = frame_index * 24
        for world_x, color in ((40, RED), (105, GREEN), (170, BLUE), (235, YELLOW), (300, VIOLET)):
            x = world_x - shift
            canvas.rectangle(x - 12, 50, x + 12, 147, color)
        canvas.line(128, 77, 128, 115, BLACK, 2)
        canvas.line(109, 96, 147, 96, BLACK, 2)
        frames.append(canvas.bytes())
    return frames


def _traffic_light_frames() -> list[bytes]:
    frames: list[bytes] = []
    states = ("red", "red", "yellow", "yellow", "green", "green")
    for state in states:
        canvas = Canvas((232, 235, 240))
        canvas.rectangle(90, 20, 166, 172, BLACK)
        for name, y, active in (("red", 54, RED), ("yellow", 96, YELLOW), ("green", 138, GREEN)):
            canvas.circle(128, y, 23, active if name == state else (62, 65, 68))
        frames.append(canvas.bytes())
    return frames


def _triangle_descent_frames() -> list[bytes]:
    frames: list[bytes] = []
    for y in (28, 53, 78, 103, 128, 153):
        canvas = Canvas()
        canvas.line(128, 18, 128, 174, LIGHT_GRAY, 2)
        canvas.polygon([(128, y + 20), (103, y - 17), (153, y - 17)], BLUE)
        frames.append(canvas.bytes())
    return frames


def _video_fixtures() -> dict[str, bytes]:
    original_positions = [30, 68, 106, 144, 182, 220]
    shuffled_positions = [30, 144, 68, 220, 106, 182]
    return {
        "fixtures/videos/vid_01_motion_original.png": apng_bytes(_motion_frames(original_positions)),
        "fixtures/videos/vid_02_motion_reversed.png": apng_bytes(_motion_frames(list(reversed(original_positions)))),
        "fixtures/videos/vid_03_motion_shuffled.png": apng_bytes(_motion_frames(shuffled_positions)),
        "fixtures/videos/vid_04_motion_frozen.png": apng_bytes(_motion_frames([30] * FRAME_COUNT)),
        "fixtures/videos/vid_05_box_opens.png": apng_bytes(_box_opening_frames()),
        "fixtures/videos/vid_06_camera_pan.png": apng_bytes(_camera_pan_frames()),
        "fixtures/videos/vid_07_light_change_holdout.png": apng_bytes(_traffic_light_frames()),
        "fixtures/videos/vid_08_triangle_down_holdout.png": apng_bytes(_triangle_descent_frames()),
    }


def build_public_fixture_bytes() -> dict[str, bytes]:
    fixtures = _image_fixtures()
    fixtures.update(_video_fixtures())
    return fixtures


def _fact(
    fact_id: str,
    claim: str,
    aliases: list[str],
    frame_indices: list[int] | None = None,
    timecodes: list[float] | None = None,
    metric_tags: list[str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": fact_id,
        "claim": claim,
        "aliases": aliases,
        "frame_indices": frame_indices or [],
        "timecodes": timecodes or [],
    }
    if metric_tags is not None:
        result["metric_tags"] = metric_tags
    return result


def _case(
    case_id: str,
    category: str,
    fixture: str,
    fixture_sha256: str,
    target_profile: str,
    expected: list[dict[str, object]],
    prohibited: list[dict[str, object]],
    *,
    holdout: bool = False,
    private: bool = False,
    ambiguity: list[str] | None = None,
    group: str = "",
    variant: str = "",
    notes: str = "",
) -> dict[str, object]:
    for item in expected:
        item.setdefault("metric_tags", ["supported_fact"])
    for item in prohibited:
        item.setdefault("metric_tags", ["unsupported_fact"])
    return {
        "id": case_id,
        "category": category,
        "fixture": fixture,
        "fixture_sha256": fixture_sha256,
        "result": f"results/{case_id}.json",
        "private": private,
        "holdout": holdout,
        "target_profile": target_profile,
        "seeds": SEEDS,
        "expected_facts": expected,
        "prohibited_facts": prohibited,
        "ambiguity_labels": ambiguity or ["none"],
        "counterfactual_group": group,
        "counterfactual_variant": variant,
        "notes": notes,
    }


def _public_cases(hashes: dict[str, str]) -> list[dict[str, object]]:
    def fact(fid: str, claim: str, *aliases: str) -> dict[str, object]:
        return _fact(fid, claim, list(aliases))

    cases = [
        _case(
            "img_01_shape_real", "synthetic_image", "fixtures/images/img_01_shape_real.png",
            hashes["fixtures/images/img_01_shape_real.png"], "h3_ref2va",
            [fact("red_square", "A red square is on the left.", "crimson square", "red block"),
             fact("blue_circle", "A blue circle is on the right.", "blue disk", "azure circle"),
             fact("square_left_of_circle", "The red square is left of the blue circle.", "square precedes circle")],
            [fact("green_triangle", "A green triangle is present.", "green three-sided shape"),
             fact("circle_left_of_square", "The blue circle is left of the red square.", "reversed spatial order")],
            group="image_shape_swap", variant="real",
        ),
        _case(
            "img_02_shape_neutral", "synthetic_image", "fixtures/images/img_02_shape_neutral.png",
            hashes["fixtures/images/img_02_shape_neutral.png"], "h3_ref2va",
            [fact("neutral_gray_field", "The image is a neutral gray field with one faint horizontal band.", "gray calibration field")],
            [fact("red_square", "A red square is present.", "crimson square"),
             fact("blue_circle", "A blue circle is present.", "blue disk")],
            group="image_shape_swap", variant="neutral",
        ),
        _case(
            "img_03_shape_unrelated", "synthetic_image", "fixtures/images/img_03_shape_unrelated.png",
            hashes["fixtures/images/img_03_shape_unrelated.png"], "h3_ref2va",
            [fact("green_triangle", "A green triangle is above a yellow diamond.", "green three-sided shape"),
             fact("yellow_diamond", "A yellow diamond is below a green triangle.", "yellow rhombus"),
             fact("triangle_above_diamond", "The triangle is above the diamond.", "triangle over diamond")],
            [fact("red_square", "A red square is present.", "crimson square"),
             fact("blue_circle", "A blue circle is present.", "blue disk")],
            group="image_shape_swap", variant="unrelated",
        ),
        _case(
            "img_04_count_three", "synthetic_image", "fixtures/images/img_04_count_three.png",
            hashes["fixtures/images/img_04_count_three.png"], "ideogram",
            [fact("three_circles", "Exactly three circles appear.", "three disks", "count of three"),
             fact("orange_circles", "All three circles are orange.", "orange disks")],
            [fact("four_circles", "Four circles appear.", "count of four"),
             fact("mixed_circle_colors", "The circles have different colors.", "multicolor circles")],
        ),
        _case(
            "img_05_color_pair", "synthetic_image", "fixtures/images/img_05_color_pair.png",
            hashes["fixtures/images/img_05_color_pair.png"], "ltx",
            [fact("violet_rectangle", "A large violet rectangle fills the center.", "purple rectangle"),
             fact("cyan_square_inside", "A cyan square sits inside the violet rectangle.", "turquoise inner square")],
            [fact("cyan_outside", "The cyan square is outside the violet rectangle.", "separate cyan square"),
             fact("orange_background", "The background is orange.", "orange field")],
        ),
        _case(
            "img_06_text_north7", "synthetic_image", "fixtures/images/img_06_text_north7.png",
            hashes["fixtures/images/img_06_text_north7.png"], "ideogram",
            [fact("text_north_7", "The image reads NORTH 7.", "NORTH seven", "NORTH 7")],
            [fact("text_south_1", "The image reads SOUTH 1.", "SOUTH one"),
             fact("no_text", "The image contains no text.", "blank sign")],
            ambiguity=["block_font"],
        ),
        _case(
            "img_07_spatial_above", "synthetic_image", "fixtures/images/img_07_spatial_above.png",
            hashes["fixtures/images/img_07_spatial_above.png"], "h3_t2va",
            [fact("red_circle_above", "A red circle is above a blue rectangle.", "red disk over blue bar"),
             fact("blue_rectangle_below", "A blue rectangle is below the red circle.", "blue bar under red disk")],
            [fact("red_circle_below", "The red circle is below the rectangle.", "reversed vertical order")],
        ),
        _case(
            "img_08_occlusion", "synthetic_image", "fixtures/images/img_08_occlusion.png",
            hashes["fixtures/images/img_08_occlusion.png"], "h3_ref2va",
            [fact("teal_circle_front", "A teal circle overlaps in front of a yellow square.", "teal disk occludes yellow square"),
             fact("yellow_square_partial", "The yellow square is partially visible behind the circle.", "partly occluded square")],
            [fact("separate_shapes", "The shapes do not overlap.", "no occlusion"),
             fact("square_in_front", "The yellow square is in front of the teal circle.", "reversed occlusion")],
        ),
        _case(
            "img_09_size_holdout", "synthetic_image", "fixtures/images/img_09_size_holdout.png",
            hashes["fixtures/images/img_09_size_holdout.png"], "ltx",
            [fact("large_black_star", "A large black star is on the left.", "big dark star"),
             fact("small_red_star", "A small red star is on the right.", "tiny crimson star"),
             fact("black_star_larger", "The black star is larger than the red star.", "left star is bigger")],
            [fact("equal_star_size", "The stars are equal in size.", "same-size stars")],
            holdout=True,
        ),
        _case(
            "img_10_grid_holdout", "synthetic_image", "fixtures/images/img_10_grid_holdout.png",
            hashes["fixtures/images/img_10_grid_holdout.png"], "ideogram",
            [fact("four_grid_blocks", "Four colored blocks form a two-by-two grid.", "2x2 colored grid"),
             fact("grid_color_order", "The grid is red/green on top and blue/yellow below.", "red green blue yellow order")],
            [fact("three_grid_blocks", "Only three blocks appear.", "three-cell grid"),
             fact("single_row", "All blocks are in one row.", "one-row layout")],
            holdout=True,
        ),
    ]

    times = [round(index * FRAME_SECONDS, 2) for index in range(FRAME_COUNT)]

    def video_fact(fid: str, claim: str, aliases: list[str], frames: list[int], selected_times: list[float]) -> dict[str, object]:
        return _fact(
            fid,
            claim,
            aliases,
            frames,
            selected_times,
            ["supported_fact", "temporal_order"],
        )

    cases.extend([
        _case(
            "vid_01_motion_original", "synthetic_video", "fixtures/videos/vid_01_motion_original.png",
            hashes["fixtures/videos/vid_01_motion_original.png"], "ltx",
            [video_fact("red_circle_moves_right", "The red circle moves smoothly from left to right.", ["disk travels rightward"], [0, 2, 5], [times[0], times[2], times[5]]),
             video_fact("starts_left_ends_right", "It starts at the left edge and ends at the right edge.", ["left opening, right ending"], [0, 5], [times[0], times[5]])],
            [fact("red_circle_moves_left", "The red circle moves right to left.", "disk travels leftward"),
             fact("red_circle_static", "The red circle remains still.", "stationary disk")],
            group="video_motion_order", variant="original",
        ),
        _case(
            "vid_02_motion_reversed", "synthetic_video", "fixtures/videos/vid_02_motion_reversed.png",
            hashes["fixtures/videos/vid_02_motion_reversed.png"], "ltx",
            [video_fact("red_circle_moves_left", "The red circle moves smoothly from right to left.", ["disk travels leftward"], [0, 2, 5], [times[0], times[2], times[5]]),
             video_fact("starts_right_ends_left", "It starts at the right edge and ends at the left edge.", ["right opening, left ending"], [0, 5], [times[0], times[5]])],
            [fact("red_circle_moves_right", "The red circle moves left to right.", "disk travels rightward")],
            group="video_motion_order", variant="reversed",
        ),
        _case(
            "vid_03_motion_shuffled", "synthetic_video", "fixtures/videos/vid_03_motion_shuffled.png",
            hashes["fixtures/videos/vid_03_motion_shuffled.png"], "h3_t2va",
            [video_fact("red_circle_jumps_nonmonotonic", "The red circle jumps back and forth in non-monotonic order.", ["shuffled circle positions"], [0, 1, 2, 3, 4, 5], times)],
            [fact("smooth_left_to_right", "The red circle moves smoothly left to right.", "monotonic rightward motion"),
             fact("smooth_right_to_left", "The red circle moves smoothly right to left.", "monotonic leftward motion")],
            group="video_motion_order", variant="shuffled",
        ),
        _case(
            "vid_04_motion_frozen", "synthetic_video", "fixtures/videos/vid_04_motion_frozen.png",
            hashes["fixtures/videos/vid_04_motion_frozen.png"], "h3_t2va",
            [video_fact("red_circle_static_left", "The red circle stays fixed on the left for every frame.", ["frozen first frame", "stationary left disk"], [0, 2, 5], [times[0], times[2], times[5]])],
            [fact("red_circle_moves_right", "The red circle travels to the right.", "rightward motion")],
            group="video_motion_order", variant="frozen_first_frame",
        ),
        _case(
            "vid_05_box_opens", "synthetic_video", "fixtures/videos/vid_05_box_opens.png",
            hashes["fixtures/videos/vid_05_box_opens.png"], "h3_ref2va",
            [video_fact("box_closed_opening", "The box lid is closed at the opening.", ["closed initial box"], [0], [times[0]]),
             video_fact("box_lid_rises", "The hinged lid rises over time.", ["box opens"], [1, 3, 5], [times[1], times[3], times[5]]),
             video_fact("box_open_ending", "The box is open at the ending.", ["open final box"], [5], [times[5]])],
            [fact("box_closes", "The box starts open and closes.", "lid lowers"),
             fact("box_static", "The lid never moves.", "unchanged closed box")],
        ),
        _case(
            "vid_06_camera_pan", "synthetic_video", "fixtures/videos/vid_06_camera_pan.png",
            hashes["fixtures/videos/vid_06_camera_pan.png"], "ltx",
            [video_fact("background_tracks_left", "Colored background posts translate left across the frame.", ["scene slides left"], [0, 2, 5], [times[0], times[2], times[5]]),
             video_fact("center_reticle_fixed", "The black center reticle remains fixed.", ["stationary crosshair"], [0, 5], [times[0], times[5]]),
             video_fact("simulated_pan_right", "The sequence simulates a camera pan to the right.", ["rightward camera pan"], [0, 2, 5], [times[0], times[2], times[5]])],
            [fact("static_camera_scene", "The background remains fixed.", "no camera motion"),
             fact("pan_left", "The sequence simulates a pan left.", "leftward camera pan")],
            ambiguity=["synthetic_camera_motion"],
        ),
        _case(
            "vid_07_light_change_holdout", "synthetic_video", "fixtures/videos/vid_07_light_change_holdout.png",
            hashes["fixtures/videos/vid_07_light_change_holdout.png"], "h3_t2va",
            [video_fact("red_light_opening", "The red traffic light is active at the opening.", ["initial red signal"], [0], [times[0]]),
             video_fact("yellow_light_middle", "The yellow light is active in the middle.", ["middle amber signal"], [2, 3], [times[2], times[3]]),
             video_fact("green_light_ending", "The green light is active at the ending.", ["final green signal"], [5], [times[5]])],
            [fact("green_to_red", "The sequence changes from green to red.", "reverse signal order"),
             fact("all_lights_active", "All three lights are active together.", "simultaneous signals")],
            holdout=True,
        ),
        _case(
            "vid_08_triangle_down_holdout", "synthetic_video", "fixtures/videos/vid_08_triangle_down_holdout.png",
            hashes["fixtures/videos/vid_08_triangle_down_holdout.png"], "h3_ref2va",
            [video_fact("blue_triangle_moves_down", "A blue triangle moves from the top toward the bottom.", ["downward triangle motion"], [0, 2, 5], [times[0], times[2], times[5]]),
             video_fact("triangle_orientation_constant", "The triangle keeps the same downward-pointing orientation.", ["orientation unchanged"], [0, 5], [times[0], times[5]])],
            [fact("blue_triangle_moves_up", "The blue triangle moves upward.", "upward triangle motion"),
             fact("triangle_rotates", "The triangle rotates while moving.", "spinning triangle")],
            holdout=True,
        ),
    ])
    return cases


def _private_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    profiles = ["ltx", "h3_t2va", "h3_ref2va", "ideogram", "ltx", "h3_ref2va"]
    for category, prefix in (("private_clean", "private_clean"), ("private_known_failure", "private_failure")):
        for number in range(1, 7):
            case_id = f"{prefix}_{number:02d}"
            failure = category == "private_known_failure"
            note = (
                "Placeholder only: before running, replace the zero hash and generic fact annotations with "
                "the locally verified private media hash/facts. Never commit the private media or result data."
            )
            cases.append(
                _case(
                    case_id,
                    category,
                    f"private/{category}/{case_id}.media",
                    "0" * 64,
                    profiles[number - 1],
                    [_fact(
                        f"{case_id}_expected_fact",
                        "Local annotator must replace this with a verified source fact before running.",
                        [f"local expected fact slot {number}"],
                    )],
                    [_fact(
                        f"{case_id}_prohibited_fact",
                        "Local annotator must replace this with a known unsupported claim before running.",
                        [f"local prohibited fact slot {number}"],
                    )],
                    holdout=number == 6,
                    private=True,
                    ambiguity=(
                        ["private_media", "requires_local_annotation", "known_silent_refusal"]
                        if failure
                        else ["private_media", "requires_local_annotation", "clean_control"]
                    ),
                    notes=note,
                )
            )
    return cases


def build_manifest(fixtures: dict[str, bytes]) -> dict[str, object]:
    hashes = {path: hashlib.sha256(payload).hexdigest() for path, payload in fixtures.items()}
    cases = _public_cases(hashes) + _private_cases()
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "DiffusionGemma Grounding Guard Benchmark v1",
        "description": (
            "Thirty offline grounding cases: 10 public synthetic images, 8 public synthetic APNG videos, "
            "6 private clean placeholders, and 6 private known-failure placeholders. Holdout allocation is "
            "2/10 images, 2/8 videos (nearest whole case to 20% of eight), 1/6 private clean, and 1/6 "
            "private known-failure. No result files are generated."
        ),
        "seeds": SEEDS,
        "cases": cases,
    }


def _manifest_bytes(manifest: dict[str, object]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def write_benchmark(root: Path) -> dict[str, object]:
    fixtures = build_public_fixture_bytes()
    manifest = build_manifest(fixtures)
    for relative_path, payload in fixtures.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_bytes(_manifest_bytes(manifest))
    return manifest


def check_benchmark(root: Path) -> list[str]:
    fixtures = build_public_fixture_bytes()
    manifest = build_manifest(fixtures)
    mismatches: list[str] = []
    for relative_path, payload in fixtures.items():
        path = root / relative_path
        if not path.is_file() or path.read_bytes() != payload:
            mismatches.append(relative_path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.read_bytes() != _manifest_bytes(manifest):
        mismatches.append("manifest.json")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic public grounding benchmark fixtures.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "grounding_v1",
        help="Benchmark output directory.",
    )
    parser.add_argument("--check", action="store_true", help="Verify checked-in bytes without writing files.")
    args = parser.parse_args(argv)
    if args.check:
        mismatches = check_benchmark(args.root)
        if mismatches:
            print(json.dumps({"ok": False, "mismatches": mismatches}, sort_keys=True))
            return 1
        print(json.dumps({"ok": True, "root": str(args.root.resolve())}, sort_keys=True))
        return 0
    manifest = write_benchmark(args.root)
    print(json.dumps({"case_count": len(manifest["cases"]), "root": str(args.root.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
