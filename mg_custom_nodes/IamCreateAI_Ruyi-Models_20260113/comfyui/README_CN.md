## 节点功能

安装完成后，可在 ComfyUI 的 *Add Node - Ruyi* 菜单找到本仓库提供的 3 个节点，如下图所示：

<div align=center>
  <img src="https://github.com/user-attachments/assets/0760399c-c57f-465d-9685-ef910b60421c"></img>
</div>

下文依次介绍各个节点的功能与参数。

> 注：新版 ComfyUI 节点在界面左侧 NODE LIBRARY 中显示。

### Load Model

用于加载模型，并提供**自动下载模型**（通过 auto_download 选项设置）的功能。

<div align=center>
  <img src="https://github.com/user-attachments/assets/2e9fcbb5-6f7c-451c-b742-5b844e949a01"></img>
</div>

- **model**: 选择使用哪个模型，目前只有 Ruyi-Mini-7B 一个选项。
- **auto_download**: 是否自动下载，默认为 yes，检测到模型不存在（或不完整）时，将自动下载模型到 *ComfyUI/models/Ruyi* 路径。
- **auto_update**: 是否自动检查并更新当前模型，默认为 yes。当启用 auto_download 时，系统将自动检查模型是否有更新，并将更新内容下载到 *ComfyUI/models/Ruyi* 路径。请注意，此功能依赖于 huggingface_hub 的缓存机制，因此请勿删除模型路径中的 *.cache* 文件夹，以确保更新过程顺利进行。

### Load LoRA

用于加载 LoRA 模型，LoRA 模型需要放在 *ComfyUI/models/loras* 路径下。

<div align=center>
  <img src="https://github.com/user-attachments/assets/33f860ca-94c9-4686-afe4-7e00de15b0ce"></img>
</div>

- **lora_name**: 需要加载的 LoRA，将自动搜索并显示 *ComfyUI/models/loras* 路径下所有模型文件。
- **strength_model**: LoRA 的影响程度，根据经验通常设置在 1.0 ~ 1.4 效果较好。

### Sampler for Image to Video

用于根据输入图片生成视频，首帧图片（start_img）为必须输入，尾帧图片（end_img）为可选输入。同时，该节点支持**镜头控制**（camera_direction）与**视频主体的运动幅度控制**（motion）。

<div align=center>
  <img src="https://github.com/user-attachments/assets/e5078e95-7100-4e9b-9baa-0d534b085e7a"></img>
</div>

- **start_img**: 首帧图片。
- **end_img**: 尾帧图片，可选输入。
- **video_length**: 视频长度，必须能**被 8 整除**，**最大支持 120 帧**。
- **base_resolution**: 视频分辨率，**模型将根据输入图片的长宽比自动选择最接近的输出视频长宽**。
  - **512** 表示生成的视频像素数接近 512 * 512。
  - **768** 表示生成的视频像素数接近 768 * 768。
- **seed**: 随机种子，用于控制随机数生成器产生随机数的序列。**不同的随机种子通常能生成不同的视频**，当生成的视频不符合需求时候，可调整此值以尝试其他的生成可能。
- **control_after_generate**: 每次生成后随机种子的变化方式。
  - **Fixed** 表示随机种子固定不变。
  - **Increment** 表示随机种子每次增加一。
  - **Decrement** 表示随机种子每次减少一。
  - **Randomize** 表示随机种子运行后随机设置。
- **steps**: 视频生成的迭代次数，**迭代次数越多，需要的时间越久**，通常 25 次能够得到较好的结果。
- **cfg**: 指令（如输入图片）的遵循程度，数值越大遵循程度越好，取值 7 ～ 10 通常能取得较好的生成效果。
- **motion**: 控制视频主体的运动幅度。
  - **1** 基本不运动，适用于静态场景。
  - **2** 正常运动幅度，适用于谈话、转头等常见场合。
  - **3** 运动幅度较大，可能出现转身、走动等情况。
  - **4** 运动幅度非常大，可能出现视频主体离开画面的情况。
  - **Auto** 表示模型自动判断运动幅度大小。
- **camera_direction**: 镜头运动。
  - **Static** 表示静止镜头。
  - **Left** 表示镜头向左移动。
  - **Right** 表示镜头向右移动。
  - **Up** 表示镜头向上移动。
  - **Down** 表示镜头向下移动。
  - **Auto** 表示模型自动判断镜头运动方向。
- **GPU_memory_mode**: 
  - **normal_mode** 是默认模式，**使用显存较多，生成速度较快**。
  - **low_memory_mode** 是低显存模式，**能大幅降低显存用量，但严重影响生成速度**。
- **GPU_offload_steps**: 用于**优化显存占用**，通过将部分临时变量从显存移动到内存而实现，会**增加内存的占用并降低生成速度**。
  - **0** 表示不优化。
  - **1 - 10**，1 显存占用最小，生成速度最慢；10 显存占用最多（少于不优化情况），生成速度最快。
  - 通常情况下，24G 显存可以使用 7 生成 512 分辨率，120 帧视频。更详细数据请参考下文。

## 工作流样例

本节展示了图生视频的工作流样例，可通过菜单中的 *Load* 按钮导入工作流。新版 ComfyUI 可通过左上方菜单的 *Workflow - Open* 加载工作流。

工作流位于 *[comfyui/workflows/](workflows/)* 目录中，素材位于 *[assets/](../assets/)* 目录中。

导入工作流后，**需要手动重新指定输入节点 LoadImage 的输入图片**，由于工作流文件只能记录输入文件的名字，所以目前只能手动设置。

### 图生视频（首帧）

工作流对应 *[workflow-ruyi-i2v-start-frame.json](workflows/workflow-ruyi-i2v-start-frame.json)* 文件。对于显存较大的用户，也可以使用 *[workflow-ruyi-i2v-start-frame-80g.json](workflows/workflow-ruyi-i2v-start-frame-80g.json)* 以提高生成速度。

<div align=center>
  <img style="width:80%" src="https://github.com/user-attachments/assets/4c0a58b8-ea04-4656-bf1f-b8665a3802a3"></img>
</div>

### 图生视频（首尾帧）

工作流对应 *[workflow-ruyi-i2v-start-end-frames.json](workflows/workflow-ruyi-i2v-start-end-frames.json)* 文件。对于显存较大的用户，也可以使用 *[workflow-ruyi-i2v-start-end-frames-80g.json](workflows/workflow-ruyi-i2v-start-end-frames-80g.json)* 以提高生成速度。

<div align=center>
  <img style="width:80%" src="https://github.com/user-attachments/assets/42b685a4-35ad-4dd8-afa2-79ded3e936cd"></img>
</div>

## 常见问题

### 模型加载错误，LoadModel: ConnectionError (MaxRetryError)

通常是网络问题导致 huggingface_hub 下载失败。网络正常的情况下，再次运行 LoadModel 节点即可解决。

### 视频生成速度慢、远低于预期

- 首先，请检查是否开启 Load Model 节点中 GPU_memory_mode 的 low_memory_mode，此模式会大幅降低视频生成速度。
- 其次，请检查 PyTorch 版本。PyTorch 在 2.2 版本支持了 FlashAttention-2（[链接](https://pytorch.org/blog/pytorch2-2/)），能大幅提升计算效率。安装新版本的 PyTorch 能够有效提升生成速度。
