// Minimal UI hook: adds a "TIPO Optimize" button to TIPOPromptOptimizer node
const app = window.app;
const api = window.api;

// Defer registration until app is ready
(function registerIllustriousTIPO(){
  if (!window.app || !window.app.registerExtension) { setTimeout(registerIllustriousTIPO, 50); return; }
  const app = window.app;
  app.registerExtension({
  name: "Illustrious.TIPO",
  nodeCreated(node) {
    if (node.comfyClass === "TIPOPromptOptimizer") {
      const btn = node.addWidget("button", "TIPO Optimize", null, async () => {
        try {
          const promptW = node.widgets.find(w => w.name === "prompt");
          const candidatesW = node.widgets.find(w => w.name === "candidates");
          const seedW = node.widgets.find(w => w.name === "seed");
          const synW = node.widgets.find(w => w.name === "enable_synonyms");
          const flavorW = node.widgets.find(w => w.name === "flavor");
          const maxLenW = node.widgets.find(w => w.name === "max_length");
          const negW = node.widgets.find(w => w.name === "negative_prompt");
          const anchorsW = node.widgets.find(w => w.name === "style_anchors");

          const body = {
            prompt: promptW?.value || "",
            candidates: candidatesW?.value || 8,
            seed: seedW?.value || 0,
            enable_synonyms: synW?.value ?? true,
            flavor: flavorW?.value || "balanced",
            max_length: maxLenW?.value || 320,
            negative_prompt: negW?.value || "",
            style_anchors: anchorsW?.value || ""
          };

          const res = await api.fetchApi("/illustrious/tipo/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          });
          if (!res.ok) throw new Error(await res.text());
          const data = await res.json();
          // Write results into the node outputs via temp labels
          node.promptPreview = data.optimized_prompt;
          node.diag = data.diagnostics;
          app.ui?.showMessage?.("TIPO selected prompt applied to output.");
        } catch (e) {
          console.error("TIPO optimize failed", e);
          app.ui?.showToast?.("TIPO optimize failed: " + e.message);
        }
      });
      btn.serialize = false;

      // Draw preview line
      const orig = node.onDrawForeground;
      node.onDrawForeground = function(ctx) {
        orig?.call(this, ctx);
        if (this.promptPreview) {
          ctx.save();
          ctx.fillStyle = "#bbb";
          ctx.font = "10px sans-serif";
          const s = (this.promptPreview.length > 60)
            ? this.promptPreview.slice(0, 57) + "..."
            : this.promptPreview;
          ctx.fillText("TIPO preview: " + s, 10, this.size[1] - 10);
          ctx.restore();
        }
      };
    }
  }
  });
})();
console.log("🧠 TIPO Interactive loaded");