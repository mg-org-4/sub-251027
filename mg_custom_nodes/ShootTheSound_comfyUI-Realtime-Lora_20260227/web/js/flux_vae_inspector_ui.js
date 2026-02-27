import { app } from "../../../scripts/app.js";

// ============================================================================
// FLUX VAE INSPECTOR — Visual Analysis UI (125 tensor units)
// ============================================================================

app.registerExtension({
    name: "FluxVAEInspector.VisualReport",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FluxVAEInspector") return;

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
                    console.error("VAE Inspector parse error:", e);
                }
            }
        };

        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);
            if (!this._inspectorData) return;

            const data = this._inspectorData;
            const blocks = data.blocks || {};
            const ablation = data.ablation || {};
            const blockKeys = Object.keys(blocks);
            if (blockKeys.length === 0) return;

            const startY = this.size[1] + 10;
            const width = Math.max(this.size[0], 500);

            const decBlocks = blockKeys.filter(k => k.startsWith("d_"));
            const encBlocks = blockKeys.filter(k => k.startsWith("e_"));

            let totalHeight = 30;
            if (decBlocks.length > 0) totalHeight += 16 + decBlocks.length * 11;
            if (encBlocks.length > 0) totalHeight += 16 + encBlocks.length * 11;

            const ablKeys = Object.keys(ablation).filter(k => ablation[k] && typeof ablation[k].impact_score === 'number');
            if (ablKeys.length > 0) totalHeight += 20 + Math.min(ablKeys.length, 15) * 12;

            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, startY, width, totalHeight + 10);

            ctx.fillStyle = "#fff";
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "left";
            ctx.fillText("Flux VAE — 125 Tensor Unit Analysis", 10, startY + 15);

            let yPos = startY + 30;
            let maxNorm = 0;
            for (const bd of Object.values(blocks)) {
                if (bd.weight_norm > maxNorm) maxNorm = bd.weight_norm;
            }

            function drawBlockList(blockList, title, titleColor) {
                ctx.fillStyle = titleColor;
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "left";
                ctx.fillText(title, 10, yPos);
                yPos += 12;

                for (const bid of blockList) {
                    const bd = blocks[bid];
                    if (!bd) continue;
                    const norm = bd.weight_norm || 0;
                    const barMax = width - 200;
                    const barLen = maxNorm > 0 ? (norm / maxNorm) * barMax : 0;

                    let color = "#888";
                    if (bid.includes("conv")) color = bid.startsWith("d_") ? "#ff7744" : "#4488ff";
                    else if (bid.includes("norm")) color = "#99bb55";
                    else if (bid.includes("attn")) color = "#6688ff";
                    else if (bid.includes("nin") || bid.includes("shortcut")) color = "#cc66cc";
                    else if (bid.includes("_up") || bid.includes("_down")) color = "#cc88cc";

                    const label = bd.label || bid;
                    ctx.fillStyle = "#999";
                    ctx.font = "8px Arial";
                    ctx.textAlign = "right";
                    ctx.fillText(label.length > 22 ? label.substring(0, 22) + "…" : label, 130, yPos);

                    ctx.fillStyle = color;
                    ctx.fillRect(135, yPos - 7, Math.max(1, barLen), 8);

                    ctx.fillStyle = "#666";
                    ctx.textAlign = "left";
                    ctx.font = "7px Arial";
                    ctx.fillText(`${norm.toFixed(0)}`, 140 + barLen, yPos);

                    yPos += 11;
                }
            }

            if (decBlocks.length > 0) drawBlockList(decBlocks, `DECODER (${decBlocks.length} units)`, "#ff8844");
            if (encBlocks.length > 0) {
                yPos += 4;
                drawBlockList(encBlocks, `ENCODER (${encBlocks.length} units)`, "#4488ff");
            }

            // Ablation top results
            if (ablKeys.length > 0) {
                yPos += 8;
                ctx.fillStyle = "#fff";
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "left";
                ctx.fillText("ABLATION IMPACT (top 15):", 10, yPos);
                yPos += 12;

                const sorted = ablKeys
                    .map(k => [k, ablation[k]])
                    .sort((a, b) => b[1].impact_score - a[1].impact_score)
                    .slice(0, 15);

                for (const [bid, result] of sorted) {
                    const score = result.impact_score || 0;
                    const label = blocks[bid]?.label || bid;
                    ctx.fillStyle = score > 30 ? "#ff4444" : score > 15 ? "#ffaa44" : "#44aa44";
                    ctx.font = "8px Arial";
                    ctx.textAlign = "left";
                    ctx.fillText(`${label}: ${score.toFixed(1)}`, 20, yPos);

                    // RGB breakdown
                    const r = result.r_impact || 0, g = result.g_impact || 0, b2 = result.b_impact || 0;
                    ctx.fillStyle = "#aa4444";
                    ctx.fillText(`R:${(r*1000).toFixed(1)}`, 220, yPos);
                    ctx.fillStyle = "#44aa44";
                    ctx.fillText(`G:${(g*1000).toFixed(1)}`, 280, yPos);
                    ctx.fillStyle = "#4444aa";
                    ctx.fillText(`B:${(b2*1000).toFixed(1)}`, 340, yPos);

                    yPos += 12;
                }
            }
        };
    }
});
