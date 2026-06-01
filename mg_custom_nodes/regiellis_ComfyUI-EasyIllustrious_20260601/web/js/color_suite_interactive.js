// ========================================================================
// Illustrious Color Suite - Unified Interactive Frontend
// Combines Auto Color Corrector and Smart Cache preview UX in one controller
// ========================================================================

(function registerIllustriousColorSuite(){
  if (window.__illustrious_color_suite_registered) return;
  const registrar = (window.app?.registerExtension || window.app?.extensionManager?.registerExtension?.bind(window.app.extensionManager));
  if (!registrar) { setTimeout(registerIllustriousColorSuite, 60); return; }
  window.__illustrious_color_suite_registered = true;

  const app = window.app; const api = window.api;

  function getNode(id){ return app?.graph?.getNodeById(id); }

  registrar({
    name: "Illustrious.ColorSuite",
    nodeCreated(node){
      if (node.comfyClass !== "IllustriousColorSuite") return;

      // State
      node.cacheStatus = { isActive:false, previewMode:false, timestamp:null, modelVersion:"auto" };
      node.previewImageB64 = null;

      // Helpers
      node.getCurrentSettings = function(){
        const s = {}; (this.widgets||[]).forEach(w=>{ if(w.name) s[w.name]=w.value; }); return s;
      };
      node.applyPreviewAdjustment = async function(settingName, value){
        try{
          const res = await api.fetchApi("/illustrious/smart_cache/adjust_preview", {
            method:"POST", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({ node_id: this.id, setting_name: settingName, setting_value: value, current_settings: this.getCurrentSettings() })
          });
          const out = await res.json();
          if(out.preview_image){ this.previewImageB64 = out.preview_image; this.setDirtyCanvas(true,true);}
        }catch(e){ console.error("[ColorSuite] preview adjust failed", e); }
      };

      // Widgets: quick actions
      if(!node.widgets.find(w=>w.name==="✅ Apply & Commit")) node.addWidget("button","✅ Apply & Commit",null, async ()=>{
        try{
          await api.fetchApi("/illustrious/smart_cache/commit_preview", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ node_id: node.id, settings: node.getCurrentSettings() }) });
          node.cacheStatus.previewMode=false; node.setDirtyCanvas(true,true);
        }catch(e){ console.error("[ColorSuite] commit failed", e); }
      });
      if(!node.widgets.find(w=>w.name==="🚪 Exit Preview Mode")) node.addWidget("button","🚪 Exit Preview Mode",null, ()=>{ node.cacheStatus.previewMode=false; node.setDirtyCanvas(true,true); });

      // Hook widget callbacks for live preview
      const watch = ["correction_strength","fix_oversaturation","enhance_details","balance_colors","adjust_contrast","custom_preset","model_version","adjustment_mode"];      
      (node.widgets||[]).forEach(w=>{
        if(w && watch.includes(w.name)){
          const orig = w.callback;
          w.callback = (val)=>{ if(orig) orig.call(node,val); if(node.cacheStatus.previewMode) node.applyPreviewAdjustment(w.name,val); };
        }
      });

      // Draw preview stripe if we have a cached preview image
      const origOnDraw = node.onDrawForeground;
      node.onDrawForeground = function(ctx){
        origOnDraw?.call(this, ctx);
        if(this.previewImageB64){
          const img = new Image(); img.src = "data:image/png;base64,"+this.previewImageB64;
          const w = this.size[0]-10, h = Math.min(120, this.size[1]-20);
          ctx.save(); ctx.globalAlpha=0.95; ctx.drawImage(img,5, this.size[1]-h-5, w, h); ctx.restore();
          ctx.fillStyle = this.cacheStatus.previewMode? "#ff6b9d":"#22c55e";
          ctx.font = "10px Arial"; ctx.textAlign = "right"; ctx.fillText(this.cacheStatus.previewMode?"Preview (cached)":"Cached", this.size[0]-6, this.size[1]-h-8);
        }
      };
    },

    setup(){
      // Wire preview/commit events
      api?.addEventListener && api.addEventListener("illustrious.smart_cache_preview_update", ({detail})=>{
        const n = getNode(detail.node_id); if(!n || n.comfyClass!=="IllustriousColorSuite") return;
        n.previewImageB64 = detail.preview_image; n.cacheStatus.timestamp = detail.timestamp; n.cacheStatus.modelVersion = detail.model_version||n.cacheStatus.modelVersion; n.setDirtyCanvas(true,true);
      });
      api?.addEventListener && api.addEventListener("illustrious.smart_cache_commit", ({detail})=>{
        const n = getNode(detail.node_id); if(!n || n.comfyClass!=="IllustriousColorSuite"||!detail.success) return;
        n.cacheStatus.previewMode=false; n.setDirtyCanvas(true,true);
      });
    }
  });
})();

console.log("🎨 Illustrious Color Suite interactive loaded");
