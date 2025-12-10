import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
	name: "Comfy.LatentLoader.Advanced.Final", // 使用新名称以避免浏览器缓存问题
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === "LatentLoaderAdvanced") {
			
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				// 上传文件的共享函数
				const uploadFile = async (file) => {
					try {
						const body = new FormData();
						// 将文件上传到临时目录，以避免覆盖问题
						body.append("image", file);
						body.append("overwrite", "true");
						body.append("type", "temp"); // <--- 添加这一行
						const resp = await api.fetchApi("/upload/image", {
							method: "POST",
							body,
						});

						if (resp.status === 200) {
							const data = await resp.json();
							// 确保路径包含了 subfolder 和 type
							const path = `${data.type}/${data.subfolder ? `${data.subfolder}/` : ''}${data.name}`;
							const textWidget = this.widgets.find((w) => w.name === "latent_file");
							if (textWidget) {
								textWidget.value = path;
							}
						} else {
							alert(`Upload Error: ${resp.status} - ${resp.statusText}`);
						}
					} catch (error) {
						console.error("Upload failed:", error);
						alert(`Upload failed: ${error}`);
					}
				};

				// “上传”按钮的逻辑
				this.addWidget("button", "upload_latent", "Upload Latent", () => {
					const inputEl = document.createElement("input");
					inputEl.type = "file";
					inputEl.accept = ".latent";
					document.body.appendChild(inputEl);
					
					const handleFileSelect = () => {
						if (inputEl.files.length > 0) {
							uploadFile(inputEl.files[0]);
						}
						inputEl.remove();
					};

					inputEl.addEventListener("change", handleFileSelect);
					inputEl.style.display = "none";
					inputEl.click();
				});

				// 拖拽事件处理
				this.onDragOver = function(e) {
					// 检查拖拽的是否是文件
					if (e.dataTransfer?.types.includes("Files")) {
						e.preventDefault(); // 关键：阻止浏览器默认行为，以允许拖放
						return true;
					}
					return false;
				};
	
				this.onDragDrop = async function(e) {
					e.preventDefault();  // 关键：阻止浏览器默认行为
					e.stopPropagation(); // 可选：阻止事件冒泡
					
					let handled = false;
					for (const file of e.dataTransfer.files) {
						if (file.name.endsWith(".latent")) {
							await uploadFile(file); // 等待上传完成
							handled = true;
							break; // 只处理第一个有效文件
						}
					}
					return handled;
				};
			};
		}
	},
});
