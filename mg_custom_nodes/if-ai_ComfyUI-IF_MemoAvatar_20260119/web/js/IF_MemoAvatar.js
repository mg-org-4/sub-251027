import { app } from "../../../scripts/app.js";

// Create base styles for buttons
const style = document.createElement('style');
style.textContent = `
    .if-button {
        background: var(--comfy-input-bg);
        border: 1px solid var(--border-color);
        color: var(--input-text);
        padding: 4px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-right: 5px;
    }
    
    .if-button:hover {
        background: var(--comfy-input-bg-hover);
    }
`;
document.head.appendChild(style);

app.registerExtension({
    name: "Comfy.IF_MemoAvatar",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IF_MemoAvatar") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            const result = origOnNodeCreated?.apply(this, arguments);

            // Add preview widget
            this.addWidget("preview", "preview", {
                serialize: false,
                size: [256, 256]
            });

            // Ensure proper widget parent references
            if (this.widgets) {
                this.widgets.forEach(w => w.parent = this);
            }

            return result;
        };

        // Add size handling
        nodeType.prototype.onResize = function(size) {
            const minWidth = 400;
            const minHeight = 200;
            size[0] = Math.max(size[0], minWidth);
            size[1] = Math.max(size[1], minHeight);
        };

        // Add execution handling
        nodeType.prototype.onExecuted = function(message) {
            if (message.preview) {
                const previewWidget = this.widgets.find(w => w.name === "preview");
                if (previewWidget) {
                    previewWidget.value = message.preview;
                }
            }
        };
    }
});