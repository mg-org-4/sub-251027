import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-ModelQuantizer.appearance",
    async nodeCreated(node) {
        // Model Quantization nodes styling - Apply styling
        if (node.comfyClass === "ModelToStateDict" ||
            node.comfyClass === "QuantizeFP8Format" ||
            node.comfyClass === "QuantizeModel" ||
            node.comfyClass === "SaveAsSafeTensor" ||
            node.comfyClass === "ControlNetFP8QuantizeNode" ||
            node.comfyClass === "ControlNetMetadataViewerNode" ||
            node.comfyClass === "GGUFQuantizerNode") {
            node.color = "#f9918b";
            node.bgcolor = "#a1cfa9";
        }
    }
});