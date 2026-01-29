# ComfyUI DreamOmni2 节点

ComfyUI自定义节点，集成DreamOmni2实现多模态理解的图像生成和编辑功能。

## ✨ 功能特性

* 🎨 **图像生成**：从1-3张参考图像生成新图像，VLM引导提示词理解
* ✏️ **图像编辑**：使用参考图像和自然语言指令编辑源图像
* 🧠 **VLM集成**：Qwen2.5-VL模型智能理解提示词
* 💾 **内存优化**：INT8量化 + CPU卸载，高效推理
* ⚡ **基于FLUX**：构建于FLUX.1-Kontext-dev架构

## 🔧 节点列表

### 核心节点

* **RunningHub DreamOmni2 Gen Pipeline**：加载生成管道和LoRA权重
* **RunningHub DreamOmni2 Edit Pipeline**：加载编辑管道和LoRA权重
* **RunningHub DreamOmni2 Generator**：从参考图像和提示词生成图像
* **RunningHub DreamOmni2 Editor**：使用源图像和参考图像编辑图像

## 🚀 快速安装

### 步骤1：安装节点

```bash
# 进入ComfyUI自定义节点目录
cd ComfyUI/custom_nodes/

# 克隆仓库
git clone https://github.com/HM-RunningHub/ComfyUI_RH_DreamOmni2.git
cd ComfyUI_RH_DreamOmni2

# 安装依赖
pip install -r requirements.txt
```

### 步骤2：下载所需模型

按以下结构下载并放置模型：

```
ComfyUI/models/
├── flux/
│   └── FLUX.1-Kontext-dev/
│       ├── transformer/
│       ├── vae/
│       ├── text_encoder/
│       └── text_encoder_2/
└── DreamOmni2/
    ├── gen_lora/
    │   └── pytorch_lora_weights.safetensors
    ├── edit_lora/
    │   └── pytorch_lora_weights.safetensors
    └── vlm-model/
        ├── config.json
        ├── model.safetensors
        └── ...
```

**模型下载地址**：
* FLUX.1-Kontext-dev: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
* DreamOmni2模型: [xiabs/DreamOmni2](https://huggingface.co/xiabs/DreamOmni2)

**快速下载**：
```bash
# 下载DreamOmni2模型（gen_lora、edit_lora、vlm-model）
huggingface-cli download --resume-download --local-dir-use-symlinks False \
    xiabs/DreamOmni2 --local-dir ComfyUI/models/DreamOmni2

# 下载FLUX.1-Kontext-dev
huggingface-cli download --resume-download --local-dir-use-symlinks False \
    black-forest-labs/FLUX.1-Kontext-dev --local-dir ComfyUI/models/flux/FLUX.1-Kontext-dev
```

安装完成后重启ComfyUI。

## 📖 使用方法

### 图像生成工作流

```
[RunningHub DreamOmni2 Gen Pipeline] → [RunningHub DreamOmni2 Generator] → [保存/预览图像]
                                              ↑
                                    [加载图像（参考图1-3）]
```

**生成器参数**：
* `ref_image`：主要参考图像（必需）
* `ref_image_2`, `ref_image_3`：额外参考图像（可选）
* `prompt`：描述期望输出的自然语言指令
* `width`, `height`：输出图像尺寸（默认：1024×1024）
* `num_inference_steps`：去噪步数（默认：30）
* `guidance_scale`：CFG引导强度（默认：3.5）
* `seed`：随机种子，用于结果复现

### 图像编辑工作流

```
[RunningHub DreamOmni2 Edit Pipeline] → [RunningHub DreamOmni2 Editor] → [保存/预览图像]
                                              ↑
                                    [加载图像（源图 + 参考图）]
```

**编辑器参数**：
* `src_image`：待编辑的源图像（必需）
* `ref_image`：风格/内容参考图像（必需）
* `prompt`：自然语言编辑指令
* `num_inference_steps`：去噪步数（默认：30）
* `guidance_scale`：CFG引导强度（默认：3.5）
* `seed`：随机种子，用于结果复现

### 提示词示例

**生成任务**：
* "创建一个蓝发金眼的动漫风格肖像"
* "生成一个赛博朋克风格的夜晚城市景观，带霓虹灯"

**编辑任务**：
* "将头发颜色改为红色，保持面部特征"
* "给人物添加墨镜和皮夹克"

## 🛠️ 技术要求

* **GPU**：18GB+ 显存
* **内存**：推荐64GB+
* **CUDA**：GPU推理必需

## ⚠️ 重要说明

* **模型路径**：模型必须放置在`ComfyUI/models/`目录下
* **CPU卸载**：自动启用以优化内存使用
* **INT8量化**：应用于transformer，支持12GB显存
* **VLM处理**：VLM模型会在生成前自动增强你的提示词
* 首次使用前必须下载所有模型文件

## 📄 许可证

本项目采用Apache License 2.0开源协议。

## 🔗 相关链接

* [DreamOmni2官方仓库](https://github.com/dvlab-research/DreamOmni2)
* [DreamOmni2模型](https://huggingface.co/xiabs/DreamOmni2)
* [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
* [Qwen2.5-VL](https://huggingface.co/Qwen)
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 🙏 致谢

开发者：[@HM-RunningHub](https://github.com/HM-RunningHub)

基于dvlab-research的原始[DreamOmni2](https://github.com/dvlab-research/DreamOmni2)项目开发。

## ⭐ 引用

如果本项目对你有帮助，请考虑引用DreamOmni2原始论文：

```bibtex
@article{dreamomni2,
  title={DreamOmni2: Multimodal Instruction-based Editing and Generation},
  author={Xia, Bin and others},
  journal={arXiv preprint},
  year={2025}
}
```

## 关于

ComfyUI自定义节点，用于DreamOmni2图像生成和编辑

### 许可证

Apache-2.0 license

