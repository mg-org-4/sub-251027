# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI


# Tips
* 11/22 update 增加帧数pad代码 支持任意帧数输入不被裁切
* vae 现在支持safetensor格式，也就是diffuser模型的vae  
* 修复一个bug（full 1.1），上传一个简易的window环境的分割bat，
* 更新适配1.1版本的新模型和代码,降低闪烁，提高保真度和稳定性
* 新增lightx2v 加速vae decoder支持和Wan2.1-VAE-upscale2x 放大decoder支持，只是在full 模式下有效，light的加速模型目前只支持（lightvaew2_1.pth  #32.2M,taew2_1.pth,lighttaew2_1.pth） 三个文件
*  满足部分网友需要超分单张图片的奇怪要求,默认输出25帧1秒的视频，详见示例，Block-Sparse-Attention 目前不支持5090的sm120架构，需要改一下Block-Sparse-Attention的源码来支持（目前还没适配）； 
*  新增切片视频路径加载节点，输入保存切片视频的路径，开启自动推理，即可推理完路径所有视频，切片的一键bat在插件目录下，window环境使用，点击即可； 
*  local_range=7这个是会最清晰，local_range=11会比较稳定，color fix 推荐用小波（没重影）； 
*  编译Block-Sparse-Attention  window的轮子 可以使用 [ smthemex 强制编译版](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6 要联网](https://github.com/lihaoyun6/Block-Sparse-Attention) 两个fork来，不推荐用官方的  
*  Block-Sparse-Attention 正确安装且能调用才是方法的完全体，当前的函数实现会更容易OOM,但是Block-Sparse-Attention轮子实在不好找，目前只有[CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper)的，我提供的（[cu128，torch2.8，py311单体](https://pan.quark.cn/s/c9ba067c89bc)）或者自己编译  
* tile关闭质量更高，需要VRam更高，corlor fix对于非模糊图片可以试试。
*  Choice vae infer full mode ，encoder infer tiny mode 选择vae跑full模式 效果最好，tiny则是速度，数据集基于4倍训练，所以1 scale是不推荐的；  
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 打星；
*  11/22 update Add frame rate pad code to support any frame rate input without cropping； 
* fix when vae is safetesors model's error
* fix a bug（full 1.1），upload a simple split bat for the window environment.
* update to version v1.1 
* add full mode [lightx2v vae encoder](https://huggingface.co/lightx2v/Autoencoders/tree/main) support（only lightvaew2_1.pth,taew2_1.pth,lighttaew2_1.pth） and [Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) support    
* Satisfying the strange requirement of some netizens to VSR a ‘single image’, the default output is 25 frames per second of video, you can find it in the example. ‘Block-Sparse-Attention’ currently does not support the 5090 sm120 architecture and needs to modify the source code of ‘Block-Sparse-Attention’ to support it;  
* Add a sliced video path loading node, enter the path to save the sliced video, enable automatic inference, and all videos in the path can be inferred. The one click ‘bat、 for slicing can be used in the node directory and window environment. Click to complete the inference;  
* Local_range=7 will be the clearest, while local_range=11 will be more stable. For color fix, it is recommended to use wavelets (without ghosting);  
* The wheel for compiling the Block Spark Attention window can be used[ smthemex ](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6](https://github.com/lihaoyun6/Block-Sparse-Attention)Two forks, not recommended to use official ones
* The correct installation and ability to call Block Spark Attention are the complete set of methods. The current function implementation will be easier to OOM, but the Block Spark Attention wheel is really hard to find. Currently, only [CU128 toch2.7]（ https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper ）Yes, I provided [cu128，torch2.8，py311](https://pan.quark.cn/s/c9ba067c89bc)Or compile it yourself
* Tile shutdown has higher quality and requires higher VRam. For non blurry images, corlor fix can be tried.    
* Choose VAE in full mode, encoder in tiny mode. Choosing VAE to run in full mode yields the best results, while tiny mode is for speed. The dataset is based on 4x training, so 1 scale is not recommended;   
* If you find the project useful, please provide the official project [FlashVSR]（ https://github.com/OpenImagingLab/FlashVSR ）Strike stars;
*  if you Like it ， star the official project： [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)  

  
1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FlashVSR

```

2.requirements  
----

```
pip install -r requirements.txt
```
要复现官方效果，必须安装Block-Sparse-Attention [torch2.8 cu2.8 py311 wheel ](https://pan.quark.cn/s/c9ba067c89bc) or [CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper) origin method need this 
```
git clone https://github.com/mit-han-lab/Block-Sparse-Attention 
# git clone https://github.com/smthemex/Block-Sparse-Attention # 无须梯子强制编译
# git clone https://github.com/lihaoyun6/Block-Sparse-Attention # 须梯子
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

3.checkpoints 
----

* 3.1.2 [FlashVSRv1.0](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1
* 3.1.2 [FlashVSRv1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/tree/main) all checkpoints 所有模型，vae 用常规的wan2.1
* 3.2 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
* 3.3 [lightvaew2_1.pth](https://huggingface.co/lightx2v/Autoencoders/tree/main) and [diffusion_pytorch_model.safetensors](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/tree/main/diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1)
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt # v1.1 or v1.0
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors #v1.1 or v1.0
|     ├── posi_prompt.pth
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth # or safetensors
|        ├──lightvaew2_1.pth  #32.2M  or taew2_1.pth,lighttaew2_1.pth
|        ├──Wan2.1_VAE_upscale2x_imageonly_real_v1_diff.safetensors  # rename from diffusion_pytorch_model.safetensors
```
  

# Example
* upscale2x and ligth lightvaew2_1.pth
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example_decoder.png)
* single image VSR 
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example_s.png)
* full old node 
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example18.png)
* tiny new
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example1022.png)
* video files loop
![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example.png)

# Acknowledgements
[DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio)  
[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)  
[taehv](https://github.com/madebyollin/taehv)  

# Citation
```
@misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
      title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
      author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
      year={2025},
      eprint={2510.12747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.12747}, 
}

```
lightx2v
```
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```
