import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.AudioRecorder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AudioRecorder") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.setSize([250, 160]);
                this.isRecording = false;
                this.recordingTimeout = null;
                
                const triggerWidget = this.widgets.find(w => w.name === "trigger");
                if (triggerWidget) {
                    triggerWidget.type = "hidden";
                    triggerWidget.hidden = true;
                    triggerWidget.value = 0;
                }
                
                return r;
            };

            nodeType.prototype.onDrawForeground = function(ctx) {
                if (!this.buttonRect) {
                    const [w, h] = this.size;
                    const margin = 10;
                    const buttonHeight = 30;
                    this.buttonRect = [margin, h - buttonHeight - margin, w - margin * 2, buttonHeight];
                }

                const [x, y, w, h] = this.buttonRect;

                ctx.fillStyle = this.isRecording ? "#ff4444" : "#2a2a2a";
                ctx.strokeStyle = "#666";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(x, y, w, h, 4);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = "#fff";
                ctx.font = "14px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    this.isRecording ? "Recording..." : "Start Recording",
                    x + w/2,
                    y + h/2
                );
            };

            nodeType.prototype.onMouseDown = function(event, local_pos) {
                if (!this.buttonRect) return false;

                const [x, y, w, h] = this.buttonRect;
                if (local_pos[0] >= x && local_pos[0] <= x + w &&
                    local_pos[1] >= y && local_pos[1] <= y + h) {
                    
                    if (!this.isRecording) {
                        this.isRecording = true;
                        
                        // Clear any existing timeout
                        if (this.recordingTimeout) {
                            clearTimeout(this.recordingTimeout);
                        }
                        
                        // Set new timeout for 10 seconds
                        this.recordingTimeout = setTimeout(() => {
                            this.isRecording = false;
                            app.graph.setDirtyCanvas(true);
                        }, 10000);  // 10 seconds
                        
                        const triggerWidget = this.widgets.find(w => w.name === "trigger");
                        if (triggerWidget) {
                            triggerWidget.value = (triggerWidget.value || 0) + 1;
                        }
                        
                        app.queuePrompt();
                    }
                    return true;
                }
                return false;
            };
        }
    }
});