/**
 * Smart Scene Generator Settings Integration
 * Registers settings for the floating UI panel
 */

// Use globals and defer registration for broader compatibility
(function registerSmartSceneSettings(){
    if (window.__illustrious_smart_scene_settings_registered) return;
    if (!window.app || !window.app.registerExtension) { setTimeout(registerSmartSceneSettings, 50); return; }
    window.__illustrious_smart_scene_settings_registered = true;
    const app = window.app;
    app.registerExtension({
    name: "Illustrious.SmartSceneSettings",
    settings: [
        {
            id: "illustrious.smart_scene.auto_show_panel",
            type: "boolean", 
            name: "Auto show Smart Scene panel",
            defaultValue: true,
            onChange: (value) => {
                if (value && window.smartSceneOpenPanel) {
                    window.smartSceneOpenPanel();
                } else if (!value && window.smartSceneClosePanel) {
                    window.smartSceneClosePanel();
                }
            }
        },
        {
            id: "illustrious.smart_scene.remember_position", 
            type: "boolean",
            name: "Remember Smart Scene panel position",
            defaultValue: true
        },
        {
            id: "illustrious.smart_scene.panel_x",
            type: "hidden"
        },
        {
            id: "illustrious.smart_scene.panel_y", 
            type: "hidden"
        },
        {
            id: "illustrious.smart_scene.default_complexity",
            type: "combo",
            name: "Default complexity level",
            defaultValue: "medium",
            options: ["simple", "medium", "detailed"]
        },
        {
            id: "illustrious.smart_scene.default_output_style",
            type: "combo", 
            name: "Default output style",
            defaultValue: "tags",
            options: ["tags", "natural", "mixed"]
        },
        {
            id: "illustrious.smart_scene.quick_access_hotkey",
            type: "combo",
            name: "Quick access hotkey", 
            defaultValue: "none",
            options: ["none", "ctrl+shift+s", "alt+s", "f1"]
        }
    ]
    });
})();