# ComfyUI RH QwenImage I2L（图生 LoRA）

这是一个 ComfyUI 自定义节点，用于基于 **DiffSynth-Studio 的 Qwen-Image i2L**，从一张或多张训练图片生成 **Image-to-LoRA (I2L)** 的 LoRA（输出为 `.safetensors`）。

- **节点列表**
  - `RunningHub_ImageQwenI2L_Loader(Style)`
  - `RunningHub_ImageQwenI2L_Loader(CFB)`
  - `RunningHub_ImageQwenI2L_LoraGenerator`（会把 LoRA 写入 `ComfyUI/models/loras/`）

原项目（模型）链接：[`DiffSynth-Studio/Qwen-Image-i2L`（ModelScope）](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L/summary)

---

## 安装

### 1）安装本插件（放到 ComfyUI/custom_nodes）

在 ComfyUI 的 `custom_nodes` 目录执行：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_QwenImageI2L.git
```

安装 Python 依赖：

```bash
cd ComfyUI/custom_nodes/ComfyUI_RH_QwenImageI2L
pip install -r requirements.txt
```

> 说明：`sync.sh` 是本地辅助脚本，已通过 `.gitignore` 排除，不会上传到 GitHub。

### 2）安装 DiffSynth-Studio（必须）

本项目依赖 `DiffSynth-Studio` 提供的 `diffsynth`。请安装最新版：

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

### 3）（建议）安装 git-lfs

部分模型仓库需要 Git LFS：

```bash
git lfs install
```

### 4）重启 ComfyUI

安装完成后重启 ComfyUI，节点才会出现在菜单中。

---

## 模型（必须）

本插件通过 DiffSynth-Studio 的模型配置加载模型。**首次运行**时可能会 **自动下载模型**（需要能访问 DiffSynth/ModelScope/HF 采用的模型源）。

如果你希望 **离线/可控**（强烈推荐），请按下面的目录结构提前把模型放到 `ComfyUI/models/` 下。

### 推荐离线目录结构

把以下内容放在 `ComfyUI/models/`：

```
ComfyUI/models/
├─ DiffSynth-Studio/
│  ├─ General-Image-Encoders/
│  │  ├─ SigLIP2-G384/model.safetensors
│  │  └─ DINOv3-7B/model.safetensors
│  └─ Qwen-Image-i2L/
│     ├─ Qwen-Image-i2L-Style.safetensors
│     ├─ Qwen-Image-i2L-Coarse.safetensors
│     ├─ Qwen-Image-i2L-Fine.safetensors
│     └─ Qwen-Image-i2L-Bias.safetensors
└─ Qwen/
   ├─ Qwen-Image/
   │  └─ text_encoder/model*.safetensors
   └─ Qwen-Image-Edit/
      └─ processor/
```

### 两种 Loader 分别需要哪些模型

- **`Loader(Style)`** 会加载：
  - `DiffSynth-Studio/General-Image-Encoders`
    - `SigLIP2-G384/model.safetensors`
    - `DINOv3-7B/model.safetensors`
  - `DiffSynth-Studio/Qwen-Image-i2L`
    - `Qwen-Image-i2L-Style.safetensors`
  - Processor：
    - `Qwen/Qwen-Image-Edit` → `processor/`

- **`Loader(CFB)`** 会加载：
  - `Qwen/Qwen-Image`
    - `text_encoder/model*.safetensors`
  - `DiffSynth-Studio/General-Image-Encoders`
    - `SigLIP2-G384/model.safetensors`
    - `DINOv3-7B/model.safetensors`
  - `DiffSynth-Studio/Qwen-Image-i2L`
    - `Qwen-Image-i2L-Coarse.safetensors`
    - `Qwen-Image-i2L-Fine.safetensors`
    -（生成时可能用到）`Qwen-Image-i2L-Bias.safetensors`
  - Processor：
    - `Qwen/Qwen-Image-Edit` → `processor/`

---

## 使用方法（ComfyUI）

### 基本工作流（Style）

```
[Load Image(多张)] → [RunningHub_ImageQwenI2L_Loader(Style)] → [RunningHub_ImageQwenI2L_LoraGenerator] →（LoRA 保存到 models/loras）
```

### 输出说明

`RunningHub_ImageQwenI2L_LoraGenerator` 会生成并保存 LoRA：

- 保存位置：`ComfyUI/models/loras/i2l_style_lora_<uuid>.safetensors`
- 节点输出：
  - `lora_name`（用于 ComfyUI 的 LoRA 下拉框）
  - `lora_path`（完整路径）

---

## 注意事项

- **GPU**：代码使用 `cuda` + `bfloat16`。请确认你的 GPU/驱动支持 BF16。
- **磁盘卸载**：Loader 默认开启磁盘 offload 以节省显存，请确保磁盘空间充足。

---

## 许可证

开源发布时请同时遵守上游项目与模型提供方的许可条款。


