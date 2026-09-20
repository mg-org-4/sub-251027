import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

app.registerExtension({
  name: "ComfyUI.FL_KsamplerSEG.Progress",
  nodeCreated(node) {
    if (!["FL_KsamplerSEG", "FL_KsamplerSEGAdvanced"].includes(node.comfyClass || node.type)) return;
    const element = document.createElement("div");
    element.style.cssText = "padding:8px;background:#18181b;color:#fafafa;font:12px sans-serif;box-sizing:border-box;";
    const label = document.createElement("div");
    label.textContent = "Run to see the active region and full-canvas preview.";
    const canvas = document.createElement("canvas");
    canvas.width = 300;
    canvas.height = 112;
    canvas.style.cssText = "display:block;width:100%;height:112px;object-fit:contain;";
    const legend = document.createElement("div");
    legend.textContent = "White: edit area · cyan: sampler crop · preview below: full canvas";
    legend.style.cssText = "font-size:10px;color:#a1a1aa;";
    element.append(label, canvas, legend);
    const widget = node.addDOMWidget("seg_progress", "seg_progress", element, {
      serialize: false, hideOnZoom: false,
      getMinHeight: () => 190, getMaxHeight: () => 190,
    });
    widget.serialize = false;
    let mask;
    let region;
    const draw = (detail) => {
      const ctx = canvas.getContext("2d");
      const [width, height] = detail.size;
      const scale = Math.min(284 / width, 104 / height);
      const left = (300 - width * scale) / 2;
      const top = (112 - height * scale) / 2;
      const [y0, x0, y1, x1] = detail.crop;
      ctx.clearRect(0, 0, 300, 112);
      ctx.fillStyle = "#09090b";
      ctx.fillRect(left, top, width * scale, height * scale);
      if (mask) ctx.drawImage(mask, left + x0 * scale, top + y0 * scale, (x1-x0)*scale, (y1-y0)*scale);
      ctx.strokeStyle = "#71717a";
      ctx.strokeRect(left, top, width * scale, height * scale);
      ctx.strokeStyle = "#22d3ee";
      ctx.strokeRect(left + x0 * scale, top + y0 * scale, (x1-x0)*scale, (y1-y0)*scale);
    };
    const onProgress = ({ detail }) => {
      if (String(detail.node) !== String(node.id) || node.graph !== app.graph) return;
      region = detail.region;
      const step = detail.start_step == null ? detail.step : detail.start_step + detail.step;
      const steps = detail.start_step == null ? detail.steps : detail.schedule_steps;
      label.textContent = `Region ${detail.position}/${detail.count} (mask ${detail.region}) · Step ${step}/${steps}`
        + (detail.state === "complete" ? " · Complete" : detail.state === "stopped" ? " · Stopped" : "");
      label.title = detail.start_step == null ? "" : `Window ${detail.start_step}–${detail.end_step}: ${detail.step}/${detail.steps} steps completed`;
      if (detail.mask) {
        mask = undefined;
        const image = new Image();
        image.onload = () => {
          if (region !== detail.region) return;
          mask = image;
          draw(detail);
        };
        image.src = detail.mask;
      }
      draw(detail);
    };
    api.addEventListener("fl_seg_sampling", onProgress);
    widget.onRemove = () => api.removeEventListener("fl_seg_sampling", onProgress);
  },
});
