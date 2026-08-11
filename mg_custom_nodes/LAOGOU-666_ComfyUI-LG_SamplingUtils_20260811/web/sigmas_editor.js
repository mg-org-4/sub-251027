import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "SigmasEditor.Interactive",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SigmasEditor") {
            
            // 扩展节点的构造函数
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 初始化节点数据
                this.sigmas_data = null;
                this.adjustments = []; // 存储实际调整后的sigma值
                this.dragging_point = -1;
                this.isAdjusting = false;
                
                // 允许调整节点大小
                this.resizable = true;
                
                // 设置初始大小（宽度自动，高度400）
                this.size = [this.size[0] || 600, 400];
                
                // 设置WebSocket监听
                this.setupWebSocket();
                
                return result;
            };
            
            // 添加WebSocket设置方法
            nodeType.prototype.setupWebSocket = function() {
                const messageHandler = (event) => {
                    const data = event.detail;
                    
                    if (!data || !data.node_id || !data.sigmas_data) {
                        return;
                    }
                    
                    // 通过node_id查找对应的节点
                    const targetNode = app.graph.getNodeById(parseInt(data.node_id));
                    
                    // 检查是否是当前节点
                    if (targetNode && targetNode === this) {
                        this.sigmas_data = data.sigmas_data.original;
                        // 从后端接收调整后的值，如果为空则使用原始值
                        this.adjustments = data.sigmas_data.adjusted || data.sigmas_data.original.slice();
                        
                        // 第一次执行时，触发 sigmas_adjustments 组件更新
                        this.updateAdjustmentsWidget();
                        
                        // 立即更新画布（强制重新计算尺寸）
                        if (this.canvas) {
                            this.updateCanvas(true);
                        } else {
                            // 如果画布还没创建，等待一下再尝试
                            setTimeout(() => {
                                if (this.canvas) {
                                    this.updateCanvas(true);
                                }
                            }, 100);
                        }
                        this.setDirtyCanvas(true, true);
                    }
                };
                
                api.addEventListener("sigmas_editor_update", messageHandler);
                
                // 存储handler引用以便后续清理
                this._sigmasEditorMessageHandler = messageHandler;
            };
            
            // 添加节点时的处理
            const onAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function() {
                const result = onAdded?.apply(this, arguments);
                
                if (!this.canvasContainer && this.id !== undefined && this.id !== -1) {
                    // 创建画布容器（高度由节点自适应）
                    const container = document.createElement("div");
                    container.style.position = "relative";
                    container.style.width = "100%";
                    container.style.height = "100%";
                    container.style.minHeight = "300px";
                    container.style.backgroundColor = "#1e1e1e";
                    container.style.borderRadius = "8px";
                    container.style.overflow = "hidden";
                    
                    // 创建画布
                    const canvas = document.createElement("canvas");
                    canvas.style.width = "100%";
                    canvas.style.height = "100%";
                    canvas.style.cursor = "crosshair";
                    
                    container.appendChild(canvas);
                    this.canvas = canvas;
                    this.canvasContainer = container;
                    
                    // 添加鼠标事件监听
                    this.addCanvasEventListeners();
                    
                    // 添加DOM组件
                    this.widgets ||= [];
                    this.widgets_up = true;
                    
                    requestAnimationFrame(() => {
                        if (this.widgets) {
                            this.canvasWidget = this.addDOMWidget("sigmas_canvas", "canvas", container);
                            
                            // 初始化时强制计算画布尺寸
                            this.updateCanvas(true);
                            this.setDirtyCanvas(true, true);
                        }
                    });
                }
                
                return result;
            };
            
            // 添加画布事件监听器
            nodeType.prototype.addCanvasEventListeners = function() {
                const canvas = this.canvas;
                
                // 获取缩放后的鼠标坐标
                const getScaledMousePos = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    // 计算缩放比例
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    
                    // 鼠标相对于canvas显示区域的位置
                    const displayX = e.clientX - rect.left;
                    const displayY = e.clientY - rect.top;
                    
                    // 转换为实际canvas坐标
                    const canvasX = displayX * scaleX;
                    const canvasY = displayY * scaleY;
                    
                    return { x: canvasX, y: canvasY };
                };
                
                // 鼠标按下
                canvas.addEventListener("mousedown", (e) => {
                    const pos = getScaledMousePos(e);
                    const pointIdx = this.findNearestPoint(pos.x, pos.y);
                    
                    if (pointIdx !== -1) {
                        this.dragging_point = pointIdx;
                        this.isAdjusting = true;
                        this.updateCanvas(); // 只重绘，不resize
                    }
                });
                
                // 鼠标移动
                canvas.addEventListener("mousemove", (e) => {
                    if (this.dragging_point !== -1 && this.isAdjusting) {
                        const pos = getScaledMousePos(e);
                        this.updatePointAdjustment(this.dragging_point, pos.y);
                        this.updateAdjustmentsWidget(); // 实时同步更新组件值
                        this.updateCanvas(); // 只重绘，不resize
                    }
                });
                
                // 鼠标释放
                canvas.addEventListener("mouseup", (e) => {
                    if (this.dragging_point !== -1) {
                        this.dragging_point = -1;
                        this.isAdjusting = false;
                        
                        // 更新adjustments widget
                        this.updateAdjustmentsWidget();
                        this.updateCanvas(); // 只重绘，不resize
                    }
                });
                
                // 鼠标离开
                canvas.addEventListener("mouseleave", (e) => {
                    if (this.dragging_point !== -1) {
                        this.dragging_point = -1;
                        this.isAdjusting = false;
                        this.updateAdjustmentsWidget();
                        this.updateCanvas(); // 只重绘，不resize
                    }
                });
            };
            
            // 查找最近的点
            nodeType.prototype.findNearestPoint = function(mouseX, mouseY) {
                if (!this.sigmas_data || this.sigmas_data.length === 0) return -1;
                
                const paddingLeft = 60;
                const paddingTop = 50;
                const paddingRight = 20;
                const paddingBottom = 50;
                
                const canvas = this.canvas;
                const chartWidth = canvas.width - paddingLeft - paddingRight;
                const chartHeight = canvas.height - paddingTop - paddingBottom;
                const chartX = paddingLeft;
                const chartY = paddingTop;
                
                const steps = this.sigmas_data.length;
                
                let closestDist = Infinity;
                let closestIdx = -1;
                
                for (let i = 0; i < steps; i++) {
                    const x = chartX + (chartWidth / (steps - 1)) * i;
                    // adjustments存储的是实际调整后的sigma值
                    const adjustedValue = this.adjustments[i] !== undefined ? this.adjustments[i] : this.sigmas_data[i];
                    const y = chartY + chartHeight - (adjustedValue * chartHeight);
                    
                    const dist = Math.sqrt(Math.pow(mouseX - x, 2) + Math.pow(mouseY - y, 2));
                    if (dist < 15 && dist < closestDist) {
                        closestDist = dist;
                        closestIdx = i;
                    }
                }
                
                return closestIdx;
            };
            
            // 更新点的调整值
            nodeType.prototype.updatePointAdjustment = function(pointIdx, mouseY) {
                if (!this.sigmas_data || pointIdx < 0 || pointIdx >= this.sigmas_data.length) return;
                
                const paddingTop = 50;
                const paddingBottom = 50;
                
                const canvas = this.canvas;
                const chartHeight = canvas.height - paddingTop - paddingBottom;
                const chartY = paddingTop;
                
                // 计算新的sigma值（范围0-1）
                const clampedY = Math.max(chartY, Math.min(chartY + chartHeight, mouseY));
                const newSigmaValue = (chartY + chartHeight - clampedY) / chartHeight;
                
                // 直接存储调整后的sigma值，限制在0-1范围内
                this.adjustments[pointIdx] = Math.max(0.0, Math.min(1.0, newSigmaValue));
            };
            
            // 向上取整到指定小数位
            nodeType.prototype.ceilToFixed = function(value, decimals) {
                const multiplier = Math.pow(10, decimals);
                return Math.ceil(value * multiplier) / multiplier;
            };
            
            // 更新adjustments widget（向上取整到4位小数）
            nodeType.prototype.updateAdjustmentsWidget = function() {
                const widget = this.widgets?.find(w => w.name === "sigmas_adjustments");
                if (widget) {
                    // 对所有调整后的值进行向上取整到4位小数，并格式化为字符串保留末尾的0
                    const formattedValues = this.adjustments.map(v => {
                        const rounded = this.ceilToFixed(v, 4);
                        return rounded.toFixed(4);
                    });
                    // 手动构建 JSON 数组字符串，保留4位小数格式
                    widget.value = '[' + formattedValues.join(', ') + ']';
                }
            };
            
            // 更新画布
            nodeType.prototype.updateCanvas = function(forceResize = false) {
                if (!this.canvas) return;
                
                requestAnimationFrame(() => {
                    const canvas = this.canvas;
                    const ctx = canvas.getContext("2d");
                    
                    // 只在必要时重新设置画布尺寸
                    if (forceResize || !this._canvasInitialized) {
                        const rect = canvas.getBoundingClientRect();
                        
                        // 确保画布有合理的尺寸
                        const width = rect.width > 0 ? rect.width : 600;
                        const height = rect.height > 0 ? rect.height : 300;
                        
                        // 只有尺寸真正改变时才重新设置
                        if (canvas.width !== width || canvas.height !== height) {
                            canvas.width = width;
                            canvas.height = height;
                        }
                        
                        this._canvasInitialized = true;
                    }
                    
                    // 增加padding以容纳坐标轴标签
                    const paddingLeft = 60;
                    const paddingRight = 20;
                    const paddingTop = 50;
                    const paddingBottom = 50;
                    
                    const chartWidth = canvas.width - paddingLeft - paddingRight;
                    const chartHeight = canvas.height - paddingTop - paddingBottom;
                    const chartX = paddingLeft;
                    const chartY = paddingTop;
                    
                    // 清空画布
                    ctx.fillStyle = "#1e1e1e";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // 如果没有数据，显示提示
                    if (!this.sigmas_data || this.sigmas_data.length === 0) {
                        ctx.fillStyle = "#999";
                        ctx.font = "14px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText("Connect Sigmas Input & Execute Workflow", canvas.width / 2, canvas.height / 2);
                        return;
                    }
                    
                    // 绘制图表区域
                    ctx.fillStyle = "#2a2a2a";
                    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
                    
                    // 绘制网格
                    ctx.strokeStyle = "#444";
                    ctx.lineWidth = 1;
                    for (let i = 0; i <= 10; i++) {
                        const y = chartY + (chartHeight / 10) * i;
                        ctx.beginPath();
                        ctx.moveTo(chartX, y);
                        ctx.lineTo(chartX + chartWidth, y);
                        ctx.stroke();
                    }
                    
                    const steps = this.sigmas_data.length;
                    
                    // 确保adjustments数组长度正确，初始化为原始sigma值
                    if (this.adjustments.length !== steps) {
                        this.adjustments = this.sigmas_data.slice();
                    }
                    
                    // 绘制Y轴刻度和标签（Sigma值 0-1，刻度0.1）
                    ctx.fillStyle = "#999";
                    ctx.font = "10px Arial";
                    ctx.textAlign = "right";
                    for (let i = 0; i <= 10; i++) {
                        const value = 1.0 - (i / 10);
                        const y = chartY + (chartHeight / 10) * i;
                        ctx.fillText(value.toFixed(1), chartX - 5, y + 3);
                    }
                    
                    // Y轴标签
                    ctx.save();
                    ctx.translate(15, chartY + chartHeight / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.textAlign = "center";
                    ctx.font = "12px Arial";
                    ctx.fillStyle = "#ccc";
                    ctx.fillText("Sigma Value", 0, 0);
                    ctx.restore();
                    
                    // 绘制X轴刻度和标签
                    ctx.fillStyle = "#999";
                    ctx.font = "10px Arial";
                    ctx.textAlign = "center";
                    
                    // 根据步数决定显示哪些刻度
                    let stepInterval = 1;
                    if (steps > 30) stepInterval = 2;
                    if (steps > 50) stepInterval = 5;
                    if (steps > 100) stepInterval = 10;
                    
                    for (let i = 0; i < steps; i += stepInterval) {
                        const x = chartX + (chartWidth / (steps - 1)) * i;
                        ctx.fillText(i.toString(), x, chartY + chartHeight + 15);
                    }
                    // 确保显示最后一步
                    if ((steps - 1) % stepInterval !== 0) {
                        const x = chartX + chartWidth;
                        ctx.fillText((steps - 1).toString(), x, chartY + chartHeight + 15);
                    }
                    
                    // X轴标签
                    ctx.font = "12px Arial";
                    ctx.fillStyle = "#ccc";
                    ctx.fillText("Steps", chartX + chartWidth / 2, chartY + chartHeight + 30);
                    
                    // 绘制曲线
                    ctx.strokeStyle = "#4a9eff";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    for (let i = 0; i < steps; i++) {
                        const x = chartX + (chartWidth / (steps - 1)) * i;
                        // adjustments存储的是实际调整后的sigma值
                        const adjustedValue = this.adjustments[i] !== undefined ? this.adjustments[i] : this.sigmas_data[i];
                        // 限制在0-1范围内
                        const clampedValue = Math.max(0, Math.min(1, adjustedValue));
                        const y = chartY + chartHeight - (clampedValue * chartHeight);
                        
                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                    
                    // 绘制控制点
                    for (let i = 0; i < steps; i++) {
                        const x = chartX + (chartWidth / (steps - 1)) * i;
                        const adjustedValue = this.adjustments[i] !== undefined ? this.adjustments[i] : this.sigmas_data[i];
                        const clampedValue = Math.max(0, Math.min(1, adjustedValue));
                        const y = chartY + chartHeight - (clampedValue * chartHeight);
                        
                        ctx.fillStyle = this.dragging_point === i ? "#ff6b6b" : "#4a9eff";
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    
                    // 绘制标题
                    ctx.fillStyle = "#ccc";
                    ctx.font = "14px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Sigmas Schedule Editor", canvas.width / 2, 20);
                    
                    // 显示当前拖拽点的信息
                    if (this.dragging_point !== -1) {
                        const orig = this.sigmas_data[this.dragging_point];
                        const adjusted = this.adjustments[this.dragging_point];
                        const multiplier = orig > 0 ? (adjusted / orig) : 1.0;
                        
                        // 向上取整到4位小数显示
                        const origCeil = this.ceilToFixed(orig, 4);
                        const adjustedCeil = this.ceilToFixed(adjusted, 4);
                        
                        ctx.fillStyle = "#ff6b6b";
                        ctx.textAlign = "center";
                        ctx.font = "11px Arial";
                        ctx.fillText(
                            `Step ${this.dragging_point}: Original=${origCeil.toFixed(4)}, Adjusted=${adjustedCeil.toFixed(4)}, Multiplier=${multiplier.toFixed(2)}x`,
                            canvas.width / 2,
                            35
                        );
                    }
                });
            };
            
            // 监听节点尺寸变化
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                const result = onResize?.apply(this, arguments);
                
                // 节点尺寸改变时，强制重新计算画布尺寸
                if (this.canvas) {
                    this._canvasInitialized = false; // 重置标志
                    this.updateCanvas(true);
                }
                
                return result;
            };
            
            // 节点移除时的处理
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                const result = onRemoved?.apply(this, arguments);
                
                // 清理画布
                if (this && this.canvas) {
                    const ctx = this.canvas.getContext("2d");
                    if (ctx) {
                        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    }
                    this.canvas = null;
                }
                if (this) {
                    this.canvasContainer = null;
                }
                
                // 清理事件监听器
                if (this._sigmasEditorMessageHandler) {
                    api.removeEventListener("sigmas_editor_update", this._sigmasEditorMessageHandler);
                    this._sigmasEditorMessageHandler = null;
                }
                
                return result;
            };
        }
    }
});
