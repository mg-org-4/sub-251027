const NORMALIZED_INPUT_LABELS = {
    reference_video_1: "Video Guide Frames",
    driving_audio: "Driving Audio",
    audio_vae: "Driving Audio VAE",
    audio_references: "Reference Audios (Optional)",
    reference_audio_1: "Reference Audio (Legacy)",
    reference_audio_vae: "Reference Audio VAE (Legacy)",
};

export function normalizeReferenceAudioLabels(node) {
    if (!node) {
        return false;
    }
    if (node.comfyClass === "H3ContinuumReferenceAudios") {
        const output = node.outputs?.find((item) => item.name === "audio_references");
        if (output) output.label = "Reference Audios";
        return true;
    }
    for (const input of node.inputs || []) {
        const label = input.name === "reference_video_1"
            && node.comfyClass === "H3ContinuumSamplerV38"
            ? "Timeline Video Frames"
            : NORMALIZED_INPUT_LABELS[input.name];
        if (label) {
            input.label = label;
        }
    }
    return true;
}
