import json
import math
import os
import uuid
from fractions import Fraction

import av
import cv2
import numpy as np
import torch

import folder_paths
import comfy.model_management as model_management
from comfy_execution.utils import get_executing_context
from server import PromptServer
from comfy_api.latest import io
from .FL_StreetScan import FL_StreetScanComposite
from .FL_VoxelNormalRelief import FL_VoxelNormalRelief
from .FL_ScanAudioEdit import FL_ScanAudioEdit, colorize_depth
from .scan_modulation import TARGETS, compile_mappings
from ..audio.audio_envelope import FLAudioEnvelope, load_audio_envelope
from ..audio.FL_Audio_Reactive_Brightness import FL_Audio_Reactive_Brightness
from ..audio.FL_Audio_Reactive_Saturation import FL_Audio_Reactive_Saturation
from ..audio.FL_Audio_Reactive_Edge_Glow import FL_Audio_Reactive_Edge_Glow


ScanAnalysis = io.Custom("FL_SCAN_ANALYSIS")


def scan_progress(stage, value=0, total=1, frame_updates=True):
    context = get_executing_context()
    if context is not None:
        PromptServer.instance.send_sync("fl_scan_progress", {"node": context.node_id,
            "stage": stage, "value": value, "max": total, "frame_updates": frame_updates})


DEFAULTS = {"orbit_degrees": 5.5, "depth_relief": 1.0, "scene_scale": .74,
            "stack_count": 4, "stack_spacing": 1.0, "stack_x": 1.0, "stack_y": 1.0,
            "stack_rotation": 0.0, "stack_opacity": 1.0, "stack_palette": "cobalt",
            "window_order": "newest_on_top", "window_blend": "normal",
            "voxel_opacity": 1.0, "edge_opacity": 1.0, "depth_opacity": 1.0,
            "window_fade_in": 0.0, "window_fade_out": 0.0,
            "normal_mix": 0.0, "hud_opacity": .7, "pose_opacity": .5,
            "surface_seed": 41, "min_cut_frames": 10, "max_cut_frames": 20,
            "cursor_scale": 1.2, "reveal_strength": 1.0,
            "base_brightness": 1.0, "brightness_intensity": .16,
            "base_saturation": .9, "saturation_intensity": .3,
            "edge_threshold": .15, "glow_intensity": 0.0, "envelope_intensity": .28,
            "glow_color": "white", "blend_mode": "screen",
            "motion_mode": "current", "parallax_scope": "whole_scene", "depth_style": "grayscale",
            "parallax_strength": 1.0, "offset_x": 0.0, "offset_y": 0.0, "dolly": 0.0, "steady_depth": .5,
            "voxel_weight": 50.0, "edge_weight": 50.0, "depth_weight": 0.0,
            "cursor_activity": 1.0, "reveal_size": 1.0, "audio_mappings": []}
LIMITS = {"orbit_degrees": (0,25), "depth_relief": (0,3), "scene_scale": (.5,1),
          "stack_count": (0,8), "window_fade_in": (0,1), "window_fade_out": (0,1),
          "normal_mix": (0,1), "hud_opacity": (0,1), "pose_opacity": (0,1),
          "surface_seed": (0,2147481947), "min_cut_frames": (2,120), "max_cut_frames": (2,240),
          "cursor_scale": (.5,3), "reveal_strength": (0,1), "base_brightness": (0,3),
          "brightness_intensity": (-1,1), "base_saturation": (0,3), "saturation_intensity": (-1,1),
          "edge_threshold": (0,.99), "glow_intensity": (0,2), "envelope_intensity": (0,2)}
LIMITS.update({key:bounds for key,bounds in TARGETS.items() if key in DEFAULTS})
CHOICES = {"glow_color": ["white","original","cyan","magenta","yellow"],
           "stack_palette": ["cobalt","cyan","magenta","mono"],
           "window_order": ["newest_on_top","voxel_on_top","edge_on_top","depth_on_top","random_on_snare"],
           "window_blend": ["normal","screen","add"],
           "blend_mode": ["screen","add","overlay"], "motion_mode": ["current","depth_parallax"],
           "parallax_scope": ["whole_scene","reveals_only"], "depth_style": ["grayscale","false_color","contours"]}

CONTROL_HELP = {
    "stack_count": "Number of back plates (0 disables them). These are 2D layers behind the projected scene, not new 3D geometry.",
    "stack_spacing": "Multiplier for the original depth-relative panel spacing. Map Kick here for expanding stacks; 0 collapses the plates.",
    "stack_x": "Horizontal spread direction and multiplier. Negative values spread left; 0 removes horizontal separation.",
    "stack_y": "Vertical spread direction and multiplier, including the original subtle sway. Negative values spread upward.",
    "stack_rotation": "Rotation in degrees per plate around the image center. Map an envelope here to fan the stack.",
    "stack_opacity": "Opacity of the back plates and their outlines. Does not fade the foreground video.",
    "stack_palette": "Alternating back-plate colors. Cobalt preserves the original appearance.",
    "window_order": "Back-to-front reveal order. Random on snare reshuffles only on Envelope 2 rising above 0.5, using the node seed; order holds between hits.",
    "window_blend": "Blend the revealed pixels with the scene below. Independent of the Finish tab's glow blend mode.",
    "voxel_opacity": "Opacity multiplier for voxel-normal windows, after Reveal strength and audio accents.",
    "edge_opacity": "Opacity multiplier for digital-edge windows, after Reveal strength and audio accents.",
    "depth_opacity": "Opacity multiplier for depth-map windows, after Reveal strength and audio accents.",
    "window_fade_in": "Reveal fade-in duration in seconds, starting when the drag begins. Borders and cursors remain visible.",
    "window_fade_out": "Reveal fade-out duration in seconds before the gesture ends. Zero preserves the original abrupt end.",
    "relief": "Voxel extrusion height. Start near 0.65; large values overlap neighboring cubes.",
    "animation": "Amplitude of the animated voxel height variation.",
    "speed": "Voxel animation speed. Zero holds the animation phase.",
    "cube_size": "Voxel cell size in pixels. Larger cells produce fewer, chunkier cubes.",
    "parallax_strength": "Depth-camera exaggeration. Try 0.5–1.5; large values expose missing surfaces.",
    "depth_relief": "Separation between near and far depth planes. Larger values exaggerate depth.",
    "dolly": "Forward/backward camera displacement. Positive values move toward the scene.",
    "steady_depth": "Depth plane held steady by the camera. 0 is far, 1 is near.",
    "offset_x": "Horizontal depth-camera displacement; foreground and background shift differently.",
    "offset_y": "Vertical depth-camera displacement; foreground and background shift differently.",
    "motion_mode": "Current preserves the original projection; depth parallax enables the native depth camera.",
    "parallax_scope": "Apply the native camera to the whole scene or only the revealed layers.",
    "scene_scale": "Projected scene size. Smaller values leave room for the digital frame layers.",
    "orbit_degrees": "Maximum orbit angle in degrees. Large angles reveal reprojection gaps.",
    "cursor_activity": "Fraction of candidate gestures retained. Zero disables gestures.",
    "reveal_size": "Scale of the drag rectangle, latched when each gesture starts.",
    "voxel_weight": "Relative chance of choosing voxel normals for a gesture. Zero disables this layer.",
    "edge_weight": "Relative chance of choosing digital edges for a gesture. Zero disables this layer.",
    "depth_weight": "Relative chance of choosing depth for a gesture. At least one layer weight must be positive.",
    "normal_mix": "Blend projected normal colors into the scene.",
    "hud_opacity": "Opacity of detection boxes and labels.",
    "pose_opacity": "Opacity of detected pose lines.",
    "motion_strength": "Strength of audio-triggered screen camera accents.",
    "min_cut_frames": "Minimum audio-edit interval in frames. Raising it past the maximum also raises the maximum.",
    "max_cut_frames": "Maximum audio-edit interval in frames. Lowering it below the minimum also lowers the minimum.",
    "edge_threshold": "Edge sensitivity for glow. Higher values retain fewer edges.",
}
CONTROL_METADATA = {key: {"default": value, "range": LIMITS.get(key), "choices": CHOICES.get(key),
    "help": CONTROL_HELP.get(key, key.replace("_", " ").capitalize() + ". Reset restores the factory value.")}
    for key,value in DEFAULTS.items() if key != "audio_mappings"}
CONTROL_METADATA.update({key: {"help": CONTROL_HELP[key]} for key in ("cube_size","relief","animation","speed","motion_strength")})
CONTROL_METADATA.update(cursor_count={"help": "Number of concurrent cursor lanes. Zero disables cursor reveals."},
    seed={"help": "Deterministic seed for cursor placement and editing. Keep fixed to compare settings."})


def settings_from_json(value):
    settings = json.loads(value)
    if not isinstance(settings, dict) or settings.keys() - DEFAULTS.keys():
        raise ValueError("Interactive Scan FX: unknown advanced settings.")
    settings = DEFAULTS | settings
    for key, (low, high) in LIMITS.items():
        number = settings[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"Interactive Scan FX: {key} must be between {low} and {high}.")
    for key in ("surface_seed", "min_cut_frames", "max_cut_frames", "stack_count"):
        if not isinstance(settings[key], int):
            raise ValueError(f"Interactive Scan FX: {key} must be an integer.")
    if settings["max_cut_frames"] < settings["min_cut_frames"]:
        settings["min_cut_frames"], settings["max_cut_frames"] = settings["max_cut_frames"], settings["min_cut_frames"]
    for name,choices in CHOICES.items():
        if settings[name] not in choices:
            raise ValueError(f"Interactive Scan FX: invalid {name}.")
    return settings


class FL_ScanAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"depth": ("IMAGE",), "normals": ("IMAGE",)}, "optional": {
            "subject_masks": ("MASK",), "detections": ("FL_SCAN_TRACKS",), "pose_keypoints": ("POSE_KEYPOINT",)}}

    RETURN_TYPES = ("FL_SCAN_ANALYSIS",)
    RETURN_NAMES = ("analysis",)
    FUNCTION = "pack"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "Packages one shot's aligned analysis. Connect shots in chronological order to Interactive Scan FX. Does not run or load models."

    def pack(self, depth, normals, subject_masks=None, detections=None, pose_keypoints=None):
        if depth.ndim != 4 or normals.ndim != 4 or depth.shape[:3] != normals.shape[:3] or normals.shape[-1] != 3 or len(depth) == 0:
            raise ValueError("Scan Analysis needs nonempty, aligned depth and RGB normal frames.")
        return ({"depth": depth, "normals": normals, "subject_masks": subject_masks,
                 "detections": detections, "pose_keypoints": pose_keypoints},)


class FL_ScanVideoSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "sections": ("INT", {"default": 4, "min": 1, "max": 100}),
                             "section_index": ("INT", {"default": 0, "min": 0, "max": 99})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "split"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "Extracts one proportional processing chunk. Indices 0 through sections-1 cover every frame exactly once, including remainder frames. This does not detect scene cuts."

    def split(self, images, sections, section_index):
        if not 1 <= sections <= len(images) or not 0 <= section_index < sections:
            raise ValueError("Scan Video Section: use at least one frame per section and an index below the section count.")
        start = len(images) * section_index // sections
        end = len(images) * (section_index + 1) // sections
        return (images[start:end].clone(),)


class FL_ScanVideoShots:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}, "optional": {"prompt_schedule": ("FL_PROMPT_SCHEDULE",)}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "Runs one shared analysis branch for every authored shot. Adjacent sections in one render group remain together. Without a schedule, processes the video as one shot."

    def split(self, images, prompt_schedule=None):
        if prompt_schedule is None:
            return ([images],)
        ranges = []
        cursor = 0
        previous_group = None
        for section in prompt_schedule["sections"]:
            start, end = section["start_frame"], section["end_frame"]
            if start != cursor or end <= start or end > len(images):
                raise ValueError("Scan Video Shots: schedule must cover this video in order without gaps or overlaps.")
            group = section.get("render_group")
            if group is not None and group == previous_group:
                ranges[-1][1] = end
            else:
                ranges.append([start, end])
            previous_group = group
            cursor = end
        if cursor != len(images) or not ranges:
            raise ValueError("Scan Video Shots: schedule length must match the selected video. Use its original schedule or disconnect the schedule for a single shot.")
        return ([images[start:end].clone() for start, end in ranges],)


class FL_ScanAnalysisCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"analysis": ("FL_SCAN_ANALYSIS",)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("FL_SCAN_ANALYSIS",)
    FUNCTION = "collect"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "Collects the shared analysis branch's ordered shots into one Scan FX input."

    def collect(self, analysis):
        return ({"shots": analysis},)


def write_preview(final, surface, mask, report, envelopes, fps, depth=None, curves=None, mappings=None, depth_style="grayscale"):
    filename = f"fl_scan_{uuid.uuid4().hex}.mp4"
    path = os.path.join(folder_paths.get_temp_directory(), filename)
    height, width = final.shape[1:3]
    w = min(320, width)
    w -= w % 2
    h = max(2, round(height * w / width / 2) * 2)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        columns = 3 if depth is not None else 2
        stream.width, stream.height, stream.pix_fmt = w * columns, h * 2, "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        for frame in range(len(final)):
            rgb = cv2.resize(final[frame].cpu().numpy(), (w,h))
            surf = cv2.resize(surface[frame].cpu().numpy(), (w,h))
            matte = cv2.resize(mask[frame].cpu().numpy(), (w,h))
            debug = rgb.copy()
            for event in report["cursor_events"]:
                if event["start"] <= frame < event["end"]:
                    a = tuple(np.rint(np.array(event["anchor"]) * [w,h]).astype(int))
                    b = tuple(np.rint(np.array(event["target"]) * [w,h]).astype(int))
                    cv2.arrowedLine(debug, a, b, (1,.7,.1), 1, tipLength=.15)
            shot = next(s["shot"] for s in report["segments"] if s["start_frame"] <= frame < s["end_frame"])
            cv2.putText(debug, f"FRAME {frame} / SHOT {shot}", (6,16), cv2.FONT_HERSHEY_PLAIN, .8, (1,1,1), 1)
            tiles = [rgb,surf,np.repeat(matte[:,:,None],3,2),debug]
            if depth is not None:
                tiles.extend([colorize_depth(cv2.resize(depth[frame].cpu().numpy(),(w,h)),depth_style),np.zeros_like(rgb)])
            atlas = np.concatenate((np.concatenate(tiles[:columns],1),np.concatenate(tiles[columns:],1)),0)
            video_frame = av.VideoFrame.from_ndarray((atlas.clip(0,1)*255).astype(np.uint8), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            if frame % max(1, len(final)//100) == 0:
                scan_progress("Encoding previews", frame+1, len(final), False)
        for packet in stream.encode():
            container.mux(packet)
    return {"filename": filename, "subfolder": "", "type": "temp", "fps": fps, "frames": len(final),
            "width": w, "height": h, "envelopes": [e["values"] for e in envelopes],
            "cuts": report["cuts"], "events": len(report["cursor_events"]), "columns": columns,
            "views": ["Final","Surface","Mask","Debug"] + (["Depth"] if depth is not None else []),
            "segments": report["segments"], "curves": curves or {}, "mappings": mappings or []}


class FL_InteractiveScanFX(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_InteractiveScanFX", display_name="FL Interactive Scan FX", category="🏵️Fill Nodes/VFX",
            is_output_node=True,
            inputs=[io.Image.Input("images"),
                io.Autogrow.Input("analysis", io.Autogrow.TemplatePrefix(ScanAnalysis.Input("shot"), "shot", min=1, max=100)),
                FLAudioEnvelope.Input("kick_envelope"), FLAudioEnvelope.Input("snare_envelope"), FLAudioEnvelope.Input("hihat_envelope"),
                io.Int.Input("fps", default=24, min=1, max=120),
                io.Int.Input("cube_size", default=12, min=4, max=64),
                io.Float.Input("relief", default=.65, min=0, max=2, step=.05),
                io.Float.Input("animation", default=.18, min=0, max=1, step=.01),
                io.Float.Input("speed", default=.7, min=0, max=5, step=.05),
                io.Int.Input("cursor_count", default=3, min=0, max=3),
                io.Float.Input("motion_strength", default=.6, min=0, max=2, step=.05),
                io.Int.Input("seed", default=73, min=0, max=2147483647, control_after_generate=True),
                io.String.Input("advanced_settings", default=json.dumps(DEFAULTS),extra_dict={"scan_mapping_targets":TARGETS,"scan_choices":CHOICES,"scan_controls":CONTROL_METADATA})],
            outputs=[io.Image.Output(display_name="final"), io.Image.Output(display_name="projected_surface"),
                     io.Mask.Output(display_name="reveal_mask"), io.String.Output(display_name="report")])

    @classmethod
    def execute(cls, images, analysis, kick_envelope, snare_envelope, hihat_envelope, fps, cube_size, relief, animation, speed, cursor_count, motion_strength, seed, advanced_settings):
        settings = settings_from_json(advanced_settings)
        shots = []
        for key in sorted(analysis, key=lambda k: int(k.removeprefix("shot"))):
            value = analysis[key]
            shots.extend(value["shots"] if "shots" in value else [value])
        lengths = [len(s["depth"]) for s in shots]
        if not shots or sum(lengths) != len(images) or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Interactive Scan FX: video has {len(images)} frames, but analysis covers {sum(lengths)} ({lengths}). Connect every ordered section from the same video; do not pad or repeat analysis frames.")
        envelopes = [load_audio_envelope(e) for e in (kick_envelope,snare_envelope,hihat_envelope)]
        if any(e["total_frames"] != len(images) or not math.isclose(e["fps"], fps) for e in envelopes):
            raise ValueError("Interactive Scan FX: all envelopes must match video frame count and FPS.")
        curves,mappings = compile_mappings(settings | {"relief":relief,"animation":animation,"speed":speed,"motion_strength":motion_strength},envelopes,fps,lengths)
        scan = torch.empty_like(images, device="cpu", dtype=torch.float32)
        surface = torch.empty_like(scan)
        projected_depth = torch.empty(images.shape[:3],device="cpu",dtype=torch.float32)
        offset = 0
        for index, (shot, count) in enumerate(zip(shots,lengths)):
            source = images[offset:offset+count]
            if shot["depth"].shape[:3] != source.shape[:3] or shot["normals"].shape[:3] != source.shape[:3]:
                raise ValueError(f"Interactive Scan FX: shot {index+1} analysis resolution does not match the video.")
            shot_seed = settings["surface_seed"] + index * 17
            local_curves = {key:values[offset:offset+count] for key,values in curves.items()}
            animated_voxels = any(row["target"] in ("relief","animation","speed") for row in mappings)
            scan_progress(f"Shot {index+1}/{len(shots)} · Voxel normals", 0, count)
            voxels, = FL_VoxelNormalRelief().render_animated(shot["normals"],shot["depth"],cube_size,relief,animation,speed,fps,shot_seed,
                local_curves if animated_voxels else None)
            masks = shot["subject_masks"]
            if masks is None:
                masks = torch.zeros(source.shape[:3], dtype=torch.float32)
            tracks = shot["detections"]
            if tracks is None:
                tracks = {"frames": [[] for _ in range(count)], "height": source.shape[1], "width": source.shape[2]}
            scan_progress(f"Shot {index+1}/{len(shots)} · Depth projection", 0, count)
            composite, _, projected,depth_frames = FL_StreetScanComposite().render_layers(source,shot["depth"],voxels,masks,tracks,
                fps,shot_seed,settings["orbit_degrees"],settings["depth_relief"],settings["scene_scale"],0,
                settings["normal_mix"],settings["hud_opacity"],0,shot["pose_keypoints"],settings["pose_opacity"],"digital_layers",
                local_curves,settings["motion_mode"],settings["parallax_scope"],
                settings["stack_count"],settings["stack_palette"])
            scan[offset:offset+count] = composite
            surface[offset:offset+count] = projected
            projected_depth[offset:offset+count] = depth_frames
            offset += count
            del voxels, composite, projected,depth_frames
        scan_progress("Cursor reveals and audio edit", 0, len(images))
        final, original, mask, report_json = FL_ScanAudioEdit().render_mapped(images,scan,surface,*envelopes,
            ",".join(map(str,lengths)),fps,seed,settings["min_cut_frames"],settings["max_cut_frames"],1.7,
            cursor_count,settings["cursor_scale"],settings["reveal_strength"],motion_strength,"audio_locked",
            curves,projected_depth,settings["depth_style"],settings["window_order"],settings["window_blend"],
            settings["window_fade_in"],settings["window_fade_out"])
        del original, scan
        quiet = {**envelopes[0],"values":[0],"total_frames":1,"duration":1/fps}
        finish_device = model_management.get_torch_device()
        frame_bytes = final.shape[1] * final.shape[2] * final.element_size()
        if finish_device.type != "cpu" and model_management.get_free_memory(finish_device) < frame_bytes * 64 + 512 * 1024**2:
            finish_device = torch.device("cpu")
        scan_progress("Color and glow", 0, len(final), False)
        for start in range(len(final)):
            end = start+1
            frames, = FL_Audio_Reactive_Brightness().apply_brightness(final[start:end],quiet,
                mask=mask[start:end,:,:,None].expand(-1,-1,-1,3),base_brightness=curves["brightness"][start],brightness_intensity=0)
            frames = frames.to(finish_device)
            frames, = FL_Audio_Reactive_Saturation().apply_saturation(frames,quiet,base_saturation=curves["saturation"][start],saturation_intensity=0)
            frames, = FL_Audio_Reactive_Edge_Glow().apply_edge_glow(frames,quiet,
                glow_intensity=curves["glow"][start],envelope_intensity=0,
                **{k:settings[k] for k in ("edge_threshold","glow_color","blend_mode")})
            final[start:end] = frames.to(final.device)
            if start % max(1, len(final)//100) == 0:
                scan_progress("Color and glow", end, len(final), False)
        report = json.loads(report_json)
        scan_progress("Encoding previews", 0, len(final), False)
        preview = write_preview(final,surface,mask,report,envelopes,fps,projected_depth,curves,mappings,settings["depth_style"])
        if mappings or settings["motion_mode"] != "current" or settings["depth_weight"] > 0:
            report.update(motion_mode=settings["motion_mode"],parallax_scope=settings["parallax_scope"],mappings=mappings)
            report_json = json.dumps(report,indent=2)
        scan_progress("Complete", 1, 1, False)
        return io.NodeOutput(final,surface,mask,report_json,ui={"fl_interactive_scan":[preview]})
