[English Document](../README.md)

# ComfyUI-Only# ComfyUI-Only

ComfyUI自定义节点集合，最初专注于workflow解析，现已扩展至提供高级文件加载功能，旨在提升您的工作流效率和体验。

## 🌟 主要功能

### 1. 高级Latent加载器 (Load Latent (Advanced)) ⭐核心功能
- 📤 **从任意位置上传**：通过“Upload Latent”按钮，直接从您电脑的任何地方选择 `.latent` 文件，无需再放入 `input` 文件夹。
- ✨ **拖拽上传**：将 `.latent` 文件直接从您的文件管理器拖拽到节点上即可上传。
- 🤖 **智能结构解析**：自动解析多种 latent 内部结构，无论是标准的 `samples` 键，还是非标准的 `latent_tensor` 键，甚至是裸张量。
- 🔢 **智能维度处理**：自动处理3D（图像）、4D（带批次的图像）和5D（视频）的 latent 张量，确保与下游节点（如VAEDecode）的兼容性。

### 2. Workflow图片文件加载器 (WorkflowImageFileLoader)
- 📁 直接从文件系统加载图片文件。
- 🔍 **自动解析元数据**：自动解析图片中的workflow元数据信息。
- 🎯 **智能提取提示词**：智能提取positive和negative提示词。
- 🏷️ **智能提取模型**：提取CheckpointLoaderSimple节点的模型名称。
- ✅ **解决核心痛点**：解决了原生 `IMAGE` 类型不包含元数据导致工作流无法复用的问题。

---

*（为保持简洁，其他旧节点说明已折叠）*

## 🚀 安装与启动

### 推荐方法：通过 ComfyUI Manager 安装
1.  **安装 ComfyUI Manager**：如果您尚未安装，请参照 [ComfyUI Manager的官方说明](https://github.com/ltdrdata/ComfyUI-Manager) 进行安装。
2.  **搜索节点**:
    -   启动 ComfyUI。
    -   点击侧边栏的 "Manager" 按钮。
    -   点击 "Install Custom Nodes"。
    -   在搜索框中输入 `ComfyUI-Only` 或 `eric183`。
3.  **安装节点**：在搜索结果中找到本插件，点击 "Install" 按钮。
4.  **重启 ComfyUI**：安装完成后，请重启 ComfyUI。

### 备选方法：手动安装 (Git Clone)
1.  **进入目录**：打开终端，进入 ComfyUI 的 `custom_nodes` 目录。
    ```bash
    cd path/to/your/ComfyUI/custom_nodes
    ```
2.  **克隆项目**：使用 `git clone` 命令克隆本仓库。
    ```bash
    git clone https://github.com/eric183/ComfyUI-Only.git
    ```
3.  **重启 ComfyUI**：重启 ComfyUI 以加载新节点。

## 📖 使用方法

#### ⭐ 加载Latent文件 (推荐)
1. 在菜单中添加 `Load Latent (Advanced)` 节点 (位于 `ComfyUI-Only/Latent` 分类下)。
2. 点击 `Upload Latent` 按钮从您的电脑中选择一个 `.latent` 文件。
3. 或者，直接将 `.latent` 文件拖拽到节点上。
4. 节点会自动处理上传、解析和格式化，然后将 `LATENT` 输出连接到下一个节点（如 `VAEDecode`）。

#### 解析图片中的Workflow
1. 添加 `Workflow图片文件加载器` 节点 (位于 `ComfyUI-Only/Image` 分类下)。
2. 在 `image_file` 下拉菜单中选择要解析的图片文件。
3. 自动获取输出的 `positive_prompt`, `negative_prompt` 和 `checkpoint_name`。

## 📁 项目结构
```
.
├── __init__.py           <-- 主入口，注册节点和JS扩展
├── js/
│   └── latent_loader.js  <-- 高级Latent加载器的前端UI逻辑
├── nodes/
│   ├── image_processing_nodes.py
│   └── latent_nodes.py
├── LICENSE
└── README.md
```

## ⚙️ 依赖要求
- Python 3.8+
- ComfyUI
- PyTorch
## 📝 更新日志

### v2.1.0 - 集成与简化 (开发代号：Orion)
- 🎉 **集成**：项目已成功添加至 **ComfyUI Manager** 官方列表，实现一键安装。
- 🚀 **简化**：**强烈推荐**通过 ComfyUI Manager 进行安装，简化用户操作。
- 🧹 **重构**：移除了 `requirements.txt` 文件，依赖项由 ComfyUI 或用户按需管理，使项目更轻量。
- 📚 **文档**：全面重写 `README.md`，更新安装说明、项目结构和依赖信息，使其更清晰、准确。

### v2.0.0 - 高级加载器版本 (开发代号：Phoenix)
- 🔥 **全新**：发布 `Load Latent (Advanced)` 节点，支持从任意位置上传和拖拽 latent 文件。
- ✨ **新增**：为新节点编写了独立的前端JS扩展，实现了原生级别的UI体验。
- 🧹 **重构**：项目结构调整，增加了 `js` 目录和 `requirements.txt`。
- 📚 **文档**：全面重写 `README.md`，聚焦新功能，简化旧说明。

### v1.2.0
- ✅ 新增：CheckpointLoaderSimple节点的模型名称提取
- ✅ 所有节点新增checkpoint_name输出

*（更早的日志已省略）*

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件
