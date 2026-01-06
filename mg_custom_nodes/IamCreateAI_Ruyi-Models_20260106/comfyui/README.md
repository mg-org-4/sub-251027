## ComfyUI Nodes

After installation, you can find the three nodes provided by this repository in the *Add Node - Ruyi* menu, as shown in the image below:

<div align=center>
  <img src="https://github.com/user-attachments/assets/0760399c-c57f-465d-9685-ef910b60421c"></img>
</div>

The following sections will introduce the functions and parameters of each node.

> Note: The new version of ComfyUI nodes is displayed in the NODE LIBRARY on the left side of the interface.

### Load Model

The Load Model node is used to load the model from disk. It also provides the functionality for **automatic model downloading** (auto_download parameter).

<div align=center>
  <img src="https://github.com/user-attachments/assets/2e9fcbb5-6f7c-451c-b742-5b844e949a01"></img>
</div>

- **model**: Select which model to use. Currently, Ruyi-Mini-7B is the only option.
- **auto_download**: Whether to automatically download. Defaults to yes. If the model is detected as missing or incomplete, it will automatically download the model to the *ComfyUI/models/Ruyi* path.
- **auto_update**: Whether to automatically check for and update the current model. Defaults to yes. When auto_download is enabled, the system will automatically check for updates to the model and download any updates to the *ComfyUI/models/Ruyi* directory. Please note that this feature relies on the caching mechanism of huggingface_hub, so do not delete the *.cache* folder in the model directory to ensure a smooth update process.

### Load LoRA

The Load LoRA node is used to load LoRA models, which need to be placed in the *ComfyUI/models/loras* path.

<div align=center>
  <img src="https://github.com/user-attachments/assets/33f860ca-94c9-4686-afe4-7e00de15b0ce"></img>
</div>

- **lora_name**: The LoRA to be loaded; it will automatically search and display all model files in the *ComfyUI/models/loras* path.
- **strength_model**: The degree of influence of the LoRA, typically set between 1.0 and 1.4 for optimal results based on experience.

### Sampler for Image to Video

The Sampler for Image to Video node is used to generate videos based on input images. The starting frame image (start_img) is a required input, while the ending frame image (end_img) is optional. This node also supports **camera control** (camera_direction parameter) and **motion amplitude control** of the video subject (motion parameter).

<div align=center>
  <img src="https://github.com/user-attachments/assets/e5078e95-7100-4e9b-9baa-0d534b085e7a"></img>
</div>

- **start_img**: The starting frame image.
- **end_img**: The ending frame image, optional input.
- **video_length**: The length of the video, which must be divisible by 8, with a maximum of 120 frames.
- **base_resolution**: The video resolution, such as 512, indicates that the generated video will have pixel dimensions close to 512 x 512. The model will automatically select the closest output video aspect ratio based on the input image.
- **seed**: A random number; different random numbers usually generate different videos. If the generated video does not meet requirements, this value can be adjusted to try other generation possibilities.
- **control_after_generate**: The method of changing the random number after each generation.
  - **Fixed** indicates the seed is fixed.
  - **Increment** indicates the seed is increased by one each time.
  - **Decrement** indicates the seed is decreased by one each time.
  - **Randomize** indicates the seed is randomly set each time.
- **steps**: The number of iterations for video generation. **More iterations require more time**. Typically, 25 iterations yield good results.
- **cfg**: The guidance of instructions (such as input images). A higher value indicates better guidance, with values between 7 and 10 usually achieving better generation results.
- **motion**: Controls the motion amplitude of the video subject.
  - **1** indicates minimal motion, which means the video subject is nearly static.
  - **2** indicates normal motion, which could be used in most case.
  - **3** indicates significant motion. The video subject is trying to move as much as possible.
  - **4** indicates a very large motion. Sometimes the video subject may move out of the camera frame.
  - **Auto** indicates the motion is automatically determined by the model.
- **camera_direction**: Camera movement.
  - **Static** indicates a stationary camera.
  - **Left** indicates the camera moves left.
  - **Right** indicates the camera moves right.
  - **Up** indicates the camera moves up.
  - **Down** indicates the camera moves down.
  - **Auto** indicates automatic determination.
- **GPU_memory_mode**: Determines how GPU memory is utilized.
  - **normal_mode** is the default mode, using more GPU memory and generating faster.
  - **low_memory_mode** is the low memory mode, which significantly reduces GPU memory usage but severely impacts generation speed.
- **GPU_offload_steps**: Used to **optimize GPU memory usage** by moving some temporary variables from GPU memory to RAM, which **increases memory usage and decreases generation speed**.
  - **0** indicates no optimization.
  - **1 - 10**, where 1 has the least GPU memory usage and the slowest generation speed; 10 has the most GPU memory usage (less than the non-optimized case) and the fastest generation speed.
  - Generally, with 24GB of GPU memory, you can use 7 to generate a 512 resolution, 120 frame video. For more detailed data, please refer to the following.

## Workflow Example

This section presents an example workflow for generating videos from images. You can import the workflow using the *Load* button in the (bottom right) menu. Note that the new version of ComfyUI move the menu to the top left, and allows you to load workflows through the *Workflow - Open* option.

The workflows are located in the *[comfyui/workflows/](workflows/)* directory, while the assets can be found in the *[assets/](../assets/)* directory.

After importing the workflow, you **need to manually re-specify the input image for the LoadImage input node**. Since the workflow file can only record the names of input files, it does require manual configuration.

### Image to Video (Starting Frame)

The workflow corresponds to the *[workflow-ruyi-i2v-start-frame.json](workflows/workflow-ruyi-i2v-start-frame.json)* file. For users with larger GPU memory, you can also use *[workflow-ruyi-i2v-start-frame-80g.json](workflows/workflow-ruyi-i2v-start-frame-80g.json)* to enhance the generation speed.

<div align=center>
  <img style="width:80%" src="https://github.com/user-attachments/assets/4c0a58b8-ea04-4656-bf1f-b8665a3802a3"></img>
</div>

### Image to Video (Starting and Ending Frames)

The workflow corresponds to the *[workflow-ruyi-i2v-start-end-frames.json](workflows/workflow-ruyi-i2v-start-end-frames.json)* file. For users with larger GPU memory, you can also use *[workflow-ruyi-i2v-start-end-frames-80g.json](workflows/workflow-ruyi-i2v-start-end-frames-80g.json)* to enhance the generation speed.

<div align=center>
  <img style="width:80%" src="https://github.com/user-attachments/assets/42b685a4-35ad-4dd8-afa2-79ded3e936cd"></img>
</div>

## Frequently Asked Questions

### Model loading error: LoadModel: ConnectionError (MaxRetryError)

This is usually caused by network issues leading to a failure in downloading from huggingface_hub. If the network is functioning properly, simply rerunning the LoadModel node should resolve the issue.

### Video generation speed is slow and far below expectations

- First, check if the low_memory_mode in the Load Model node is enabled. This mode significantly reduces video generation speed.
- Second, verify the version of PyTorch. PyTorch version 2.2 supports FlashAttention-2 ([link](https://pytorch.org/blog/pytorch2-2/)), which can greatly enhance computational efficiency. Installing the latest version of PyTorch can effectively improve generation speed.
