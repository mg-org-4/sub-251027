# ComfyUI_UniBlockSwap
A universal swap node that supports ComfyUI native workflow, allowing 4_6G users to experience Minimax  or Klein9B or Bernini or other large models

# Update
* Fix gguf loader cause high ram error,修复gguf加载时内存占用过大的bug，使用时注意避免推理过大分辨率或者时长过长，导致调用共享显存（如果调用了，就变慢了，不划算）
* Make it for ' low Vram and normal Ram' users to esay running ComfyUI origin workflows.(Support allmot all of comfyUI origin workflows)
* Support text encoder or diffusion models, is enable text encoder will need more Ram 

# Installation  
----

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_UniBlockSwap
```

# Example
* run minimax H3 5min 0.4 just need 4.5G Vram (要降低Te的占用需要加te模块,或者用comfyUI自带的Vbar,OOM再加TE swap,避免内存占用)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/example_minimax.png)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/minimax.png)
* run bernini int4 +loras ,512x384x120frames just need 9-10G Vram (if unpack node,notice batch size is wrong 注意官方模板解开后，batch size指向是错的，须改成1)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/bernini.png)
* run klein9B Q8 just need 4.8G Vram
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/klein9B.png)
* run boogu edit bf16 (Ram is not really used)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/boogu.png)
* run krea2  bf16 (Ram is not really used)
![](https://github.com/smthemex/ComfyUI_UniBlockSwap/blob/main/example_workflows/krea2.png)
