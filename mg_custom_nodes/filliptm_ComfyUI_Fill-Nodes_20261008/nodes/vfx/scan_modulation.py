import math


TARGETS = {
    "stack_spacing": (0,4), "stack_x": (-2,2), "stack_y": (-2,2),
    "stack_rotation": (-15,15), "stack_opacity": (0,1),
    "voxel_opacity": (0,1), "edge_opacity": (0,1), "depth_opacity": (0,1),
    "relief": (0,2), "animation": (0,1), "speed": (0,5),
    "motion_strength": (0,2), "orbit_degrees": (0,25), "depth_relief": (0,3),
    "scene_scale": (.5,1), "normal_mix": (0,1), "hud_opacity": (0,1), "pose_opacity": (0,1),
    "parallax_strength": (0,2), "offset_x": (-.2,.2), "offset_y": (-.2,.2),
    "dolly": (-.3,.3), "steady_depth": (0,1),
    "cursor_activity": (0,1), "reveal_size": (.25,1.75), "cursor_scale": (.5,3), "reveal_strength": (0,1),
    "voxel_weight": (0,100), "edge_weight": (0,100), "depth_weight": (0,100),
    "brightness": (0,3), "saturation": (0,3), "glow": (0,2),
}


def compile_mappings(settings, envelopes, fps, lengths):
    count = sum(lengths)
    curves = {key: [settings[key]] * count for key in TARGETS if key in settings}
    # These are the original three audio reactions, expressed as absolute values.
    curves["brightness"] = [settings["base_brightness"] + v * settings["brightness_intensity"] for v in envelopes[1]["values"]]
    curves["saturation"] = [max(0,settings["base_saturation"] + v * settings["saturation_intensity"]) for v in envelopes[0]["values"]]
    curves["glow"] = [settings["glow_intensity"] + v * settings["envelope_intensity"] for v in envelopes[0]["values"]]
    rows = settings["audio_mappings"]
    if not isinstance(rows,list):
        raise ValueError("Audio mappings must be a list.")
    occupied = {}
    boundaries = set()
    offset = 0
    for length in lengths:
        boundaries.add(offset)
        offset += length
    evaluated = []
    for index,row in enumerate(rows):
        if not isinstance(row,dict) or row.keys() - {"enabled","source","target","minimum","maximum","start_frame","end_frame","invert","smoothing"}:
            raise ValueError(f"Audio mapping {index+1}: unknown fields.")
        if not isinstance(row.get("enabled",True),bool) or not isinstance(row.get("invert",False),bool):
            raise ValueError(f"Audio mapping {index+1}: enabled and invert must be booleans.")
        if not row.get("enabled",True):
            continue
        target,source = row.get("target"),row.get("source")
        if target not in TARGETS or type(source) is not int or not 0 <= source < 3:
            raise ValueError(f"Audio mapping {index+1}: select a supported parameter and envelope.")
        start,end = row.get("start_frame",0),row.get("end_frame")
        end = count if end is None else end
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= count:
            raise ValueError(f"Audio mapping {index+1}: range must fit the video (end is exclusive).")
        low,high = TARGETS[target]
        minimum,maximum,smoothing = row.get("minimum"),row.get("maximum"),row.get("smoothing",0)
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (minimum,maximum,smoothing)) or not low <= minimum <= maximum <= high or not 0 <= smoothing <= 2:
            raise ValueError(f"Audio mapping {index+1}: invalid range or smoothing; {target} supports {low} to {high}.")
        if any(start < b and end > a for a,b in occupied.get(target,[])):
            raise ValueError(f"Audio mappings for {target} overlap. Use non-overlapping ranges.")
        occupied.setdefault(target,[]).append((start,end))
        coefficient = 1 if smoothing == 0 else 1-math.exp(-1/(fps*smoothing))
        previous = None
        for frame in range(start,end):
            value = envelopes[source]["values"][frame]
            if row.get("invert",False):
                value = 1-value
            value = minimum + (maximum-minimum)*value
            previous = value if previous is None or frame in boundaries else previous + coefficient*(value-previous)
            curves[target][frame] = previous
        evaluated.append({**row,"start_frame":start,"end_frame":end})
    if any(sum(curves[k][i] for k in ("voxel_weight","edge_weight","depth_weight")) <= 0 for i in range(count)):
        raise ValueError("Reveal layer weights must have a positive total at every frame.")
    return curves,evaluated
