import { app } from "../../../scripts/app.js";

// ============================================================================
// QWEN3-8B TEXT ENCODER INSPECTOR — Visual Analysis UI
// For Flux 2 Klein 9B
// ============================================================================

app.registerExtension({
    name: "Qwen3_8BTextEncoderInspector.VisualReport",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Qwen3_8BTextEncoderInspector") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(output) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            
            if (output && output.analysis_json && output.analysis_json[0]) {
                try {
                    const jsonStr = output.analysis_json[0];
                    if (jsonStr && jsonStr.length > 2) {
                        this._inspectorData = JSON.parse(jsonStr);
                        this.setDirtyCanvas(true);
                    }
                } catch (e) {
                    console.error("Inspector parse error:", e);
                }
            }
        };

        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);
            
            if (!this._inspectorData) return;
            
            const data = this._inspectorData;
            const layers = data.layers || {};
            const ablation = data.ablation || {};
            
            const startY = this.size[1] + 10;
            const width = Math.max(this.size[0], 400);
            const layerCount = Object.keys(layers).length;
            
            if (layerCount === 0) return;
            
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, startY, width, 180);
            
            ctx.fillStyle = "#fff";
            ctx.font = "bold 11px Arial";
            ctx.fillText("Qwen3-8B Layer Activation Analysis", 10, startY + 15);
            
            const barStartY = startY + 25;
            const barWidth = (width - 80) / 36;
            
            let maxMag = 0;
            for (const [key, layerData] of Object.entries(layers)) {
                if (layerData.magnitude > maxMag) maxMag = layerData.magnitude;
            }
            
            for (let i = 0; i < 36; i++) {
                const layerKey = `L${i}`;
                const layerData = layers[layerKey] || {};
                const mag = layerData.magnitude || 0;
                const attnMag = layerData.attn_mag || 0;
                const mlpMag = layerData.mlp_mag || 0;
                
                const x = 40 + i * barWidth;
                const normalizedMag = maxMag > 0 ? mag / maxMag : 0;
                const normalizedAttn = maxMag > 0 ? attnMag / maxMag : 0;
                const normalizedMlp = maxMag > 0 ? mlpMag / maxMag : 0;
                
                const barLen = normalizedMag * 40;
                
                let color;
                if (i < 12) color = "#4488ff";
                else if (i < 24) color = "#44cc44";
                else color = "#ff8844";
                
                ctx.fillStyle = color;
                ctx.fillRect(x, barStartY + 40 - barLen, barWidth - 1, barLen);
                
                ctx.fillStyle = "#6666ff";
                const attnLen = normalizedAttn * 40;
                ctx.fillRect(x, barStartY + 90 - attnLen, barWidth - 1, attnLen);
                
                ctx.fillStyle = "#ff6644";
                const mlpLen = normalizedMlp * 40;
                ctx.fillRect(x, barStartY + 140 - mlpLen, barWidth - 1, mlpLen);
            }
            
            ctx.fillStyle = "#888";
            ctx.font = "9px Arial";
            ctx.fillText("Total", 5, barStartY + 25);
            ctx.fillText("Attn", 5, barStartY + 75);
            ctx.fillText("MLP", 5, barStartY + 125);
            
            ctx.fillStyle = "#4488ff";
            ctx.fillText("Early (0-11)", 50, barStartY + 55);
            ctx.fillStyle = "#44cc44";
            ctx.fillText("Middle (12-23)", 140, barStartY + 55);
            ctx.fillStyle = "#ff8844";
            ctx.fillText("Late (24-35)", 250, barStartY + 55);
            
            if (Object.keys(ablation).length > 0 && !ablation.error) {
                const ablY = barStartY + 150;
                ctx.fillStyle = "#fff";
                ctx.font = "bold 10px Arial";
                ctx.fillText("Ablation Impact:", 10, ablY);
                
                let offsetX = 100;
                for (const [region, result] of Object.entries(ablation)) {
                    const score = result.impact_score || 0;
                    ctx.fillStyle = score > 30 ? "#ff4444" : score > 15 ? "#ffaa44" : "#44aa44";
                    ctx.fillText(`${region}: ${score.toFixed(1)}`, offsetX, ablY);
                    offsetX += 120;
                }
            }
        };
    }
});
