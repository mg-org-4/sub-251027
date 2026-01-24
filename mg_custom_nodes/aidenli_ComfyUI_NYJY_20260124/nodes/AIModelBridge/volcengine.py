from .api import AIModelBridgeFactory, ModelOptionBase
from .definition import *
from PIL import Image
import numpy as np
import base64
import io as sys_io
import torch
import traceback
import time
import requests
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy_api_nodes.util import (
    download_url_to_image_tensor,
    download_url_to_video_output,
    tensor_to_base64_string,
)
import json

def extract_image_size(image_datas, default_width=512, default_height=512):
    """
    从图像数据中提取尺寸信息
    
    Args:
        image_datas: 图像数据列表
        default_width: 默认宽度
        default_height: 默认高度
        
    Returns:
        tuple: (width, height) 图像尺寸
    """
    width, height = default_width, default_height
    if image_datas and hasattr(image_datas[0], 'size'):
        width, height = image_datas[0].size.split("x")
        width, height = int(width), int(height)
    return width, height


class ImageConverter:
    """图像格式转换工具类"""
    
    @staticmethod
    def base64_to_comfyui_image(base64_string):
        """
        将base64图像数据转换为ComfyUI的image格式
        
        Args:
            base64_string: base64编码的图像字符串
            
        Returns:
            torch.Tensor: ComfyUI格式的图像tensor
        """
        image = Image.open(sys_io.BytesIO(base64.b64decode(base64_string)))
        image = np.array(image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(image)
        return tensor_image
    
    @staticmethod
    def comfyui_image_to_base64(image, format="JPEG"):
        """
        将ComfyUI的image格式转换为base64字符串
        
        Args:
            image: ComfyUI的torch.Tensor图像
            format: 图像格式，默认为JPEG
            
        Returns:
            str: base64编码的图像字符串，格式为 data:image/<format>;base64,<base64_string>
        """
        image = image.numpy()
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        buffered = sys_io.BytesIO()
        pil_image.save(buffered, format=format)
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{base64_string}"
    
    @staticmethod
    def batch_base64_to_comfyui_images(base64_strings):
        """
        批量将base64图像数据转换为ComfyUI的image格式
        
        Args:
            base64_strings: base64编码的图像字符串列表
            
        Returns:
            torch.Tensor: 堆叠的图像tensor
        """
        images = []
        
        for base64_string in base64_strings:
            tensor_image = ImageConverter.base64_to_comfyui_image(base64_string)
            images.append(tensor_image)
        
        return torch.stack(images, dim=0)
    
    @staticmethod
    def batch_comfyui_images_to_base64(images, format="JPEG"):
        """
        批量将ComfyUI的image格式转换为base64字符串列表
        
        Args:
            images: ComfyUI的torch.Tensor图像批次
            format: 图像格式，默认为JPEG
            
        Returns:
            list: base64编码的图像字符串列表
        """
        images_base64 = []
        for image in images:
            base64_string = ImageConverter.comfyui_image_to_base64(image, format)
            images_base64.append(base64_string)
        return images_base64

class VolcengineChatOption(ModelOptionBase):
    MODEL_LIST = volcengine_chat_models


class VolcengineImageOption(ModelOptionBase):
    MODEL_LIST = volcengine_image_models

class VolcengineChatNode:
    platform = "volcengine"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "doubao-seed-1-6-251015"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "images": ("IMAGE", {"default": None}),
                "thinking": (["disabled", "enabled", "auto"], {"default": "disabled"}),
                "api_key": ("STRING", {"default": ""}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Answer",)
    FUNCTION = "chat"
    CATEGORY = "NYJY/Volcengine"

    def chat(self, model, prompt, api_key, images=[], thinking="disabled", max_tokens=4096, seed=0):
        try:
            factory = AIModelBridgeFactory()
            model_instance = factory.get_model(self.platform)
            if api_key != "":
                model_instance.set_config({"api_key": api_key})

            format_prompt = {"role": "user", "content": []}
            if images is not None and len(images) > 0:
                for image in images:
                    format_prompt["content"].append({"image_url": {"url": ImageConverter.comfyui_image_to_base64(image)},"type": "image_url"})
            format_prompt["content"].append({"text": prompt, "type": "text"})

            response = model_instance.chat_completion(
                model=model,
                messages=[format_prompt],
                max_tokens=int(max_tokens),
                thinking={"type":thinking},
            )
        
            return (response,)
        except Exception as e:
            print(f"调用模型[{self.platform} -- {model}]失败：", str(e))
            return (str(e), )


class VolcengineTxt2ImgNode:
    platform = "volcengine"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (volcengine_image_models, {"default": "doubao-seedream-4-0-250828"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "ratio_or_size": (seedream4_image_ratio, {"default": "1:1 2048x2048"}),
                "override_with": ("INT", {"default": 0}),
                "override_height":("INT", {"default": 0}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE","INT", "INT",)
    RETURN_NAMES = ("Image","Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "NYJY/Volcengine"

    def generate(self, model, prompt, ratio_or_size, override_with, override_height, max_images, api_key, watermark, seed):
        try:
            factory = AIModelBridgeFactory()
            model_instance = factory.get_model(self.platform)
            if api_key != "":
                model_instance.set_config({"api_key": api_key})
            
            if override_with > 0 and override_height > 0:
                ratio_or_size = f"{override_with}x{override_height}"
            elif "x" in ratio_or_size:
                ratio_or_size = ratio_or_size.split(" ")[1]

            input = {
                "prompt": prompt,
                "size": ratio_or_size, 
                "max_images": max_images,
                "watermark": watermark,
            } 

            image_datas = model_instance.i2i(model, input)
            
            # 提取base64字符串列表
            base64_strings = [image_data.b64_json  for image_data in image_datas]
            
            # 获取第一张图片的尺寸信息
            width, height = extract_image_size(image_datas)
            
            # 转换图片
            images_tensor = ImageConverter.batch_base64_to_comfyui_images(base64_strings)
            return (images_tensor, width, height)
        except Exception as e:
            # 打印堆栈信息
            traceback.print_exc()
            print(f"调用模型[{self.platform} -- {model}]失败：", str(e))
            # 返回一个空的图像tensor作为错误处理
            return (torch.zeros((1, 512, 512, 3)),)

class Seedream4Txt2ImgNode(VolcengineTxt2ImgNode):
    pass

class VolcengineImg2ImgNode:
    platform = "volcengine"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (volcengine_image_models, {"default": "doubao-seedream-4-0-250828"}),
                "images": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "ratio_or_size": (seedream4_image_ratio, {"default": "1:1 2048x2048"}),
                "override_with": ("INT", {"default": 0}),
                "override_height":("INT", {"default": 0}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE","INT", "INT",)
    RETURN_NAMES = ("Image","Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "NYJY/Volcengine"

    def generate(self, model, images, prompt, ratio_or_size, override_with, override_height, max_images, api_key, watermark, seed):
        try:
            factory = AIModelBridgeFactory()
            model_instance = factory.get_model(self.platform)
            if api_key != "":
                model_instance.set_config({"api_key": api_key})

            # 使用ImageConverter批量转换comfyui的image类型到base64
            images_base64 = ImageConverter.batch_comfyui_images_to_base64(images)
            
            if override_with > 0 and override_height > 0:
                ratio_or_size = f"{override_with}x{override_height}"
            elif "x" in ratio_or_size:
                ratio_or_size = ratio_or_size.split(" ")[1]

            input = {
                "prompt": prompt,
                "image": images_base64,
                "size": ratio_or_size, 
                "max_images": max_images,
                "watermark": watermark,
            } 

            image_datas = model_instance.i2i(model, input)
            
            # 提取base64字符串列表
            base64_strings = [image_data.b64_json  for image_data in image_datas]
            
            # 获取第一张图片的尺寸信息
            width, height = extract_image_size(image_datas)
            
            # 转换图片
            images_tensor = ImageConverter.batch_base64_to_comfyui_images(base64_strings)
            return (images_tensor, width, height)
        except Exception as e:
            # 打印堆栈信息
            traceback.print_exc()
            print(f"调用模型[{self.platform} -- {model}]失败：", str(e))
            # 返回一个空的图像tensor作为错误处理
            return (torch.zeros((1, 512, 512, 3)),)

class Seedream4Img2ImgNode(VolcengineImg2ImgNode):
    pass


class Seedream3Txt2ImgNode:
    platform = "volcengine"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "doubao-seedream-3-0-t2i-250415"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "ratio": (seedream3_image_ratio, {"default": "1:1 1328x1328"}),
                "override_with": ("INT", {"default": 0}),
                "override_height":("INT", {"default": 0}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 2.5, "min":1, "max": 10, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
                "watermark": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE","INT", "INT",)
    RETURN_NAMES = ("Image","Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "NYJY/Volcengine"

    def generate(self, model, prompt, ratio, override_with, override_height, guidance_scale, seed, api_key, watermark):
        try:
            factory = AIModelBridgeFactory()
            model_instance = factory.get_model(self.platform)
            if api_key != "":
                model_instance.set_config({"api_key": api_key})
            
            if override_with > 0 and override_height > 0:
                size = f"{override_with}x{override_height}"
            elif "x" in ratio:
                size = ratio.split(" ")[1]
            else:
                size = ratio

            input = {
                "prompt": prompt,
                "size": size, 
                "guidance_scale": guidance_scale,
                "seed": seed,
                "watermark": watermark,
            } 

            image_datas = model_instance.i2i(model, input)
            
            # 提取base64字符串列表
            base64_strings = [image_data.b64_json  for image_data in image_datas]
            
            # 获取第一张图片的尺寸信息
            width, height = extract_image_size(image_datas)
            
            # 转换图片
            images_tensor = ImageConverter.batch_base64_to_comfyui_images(base64_strings)
            return (images_tensor, width, height)
        except Exception as e:
            # 打印堆栈信息
            traceback.print_exc()
            print(f"调用模型[{self.platform} -- {model}]失败：", str(e))
            # 返回一个空的图像tensor作为错误处理
            return (torch.zeros((1, 512, 512, 3)),)

class Seededit3Node:
    platform = "volcengine"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "doubao-seededit-3-0-i2i-250628"}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 5.5, "min":1, "max": 10, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
                "watermark": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE","INT", "INT",)
    RETURN_NAMES = ("Image","Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "NYJY/Volcengine"

    def generate(self, model, image, prompt, guidance_scale, seed, api_key, watermark):
        try:
            factory = AIModelBridgeFactory()
            model_instance = factory.get_model(self.platform)
            if api_key != "":
                model_instance.set_config({"api_key": api_key})

            # 使用ImageConverter转换comfyui的image类型到base64（只支持单张图片）
            images_base64 = []
            base64_string = ImageConverter.comfyui_image_to_base64(image[0])
            images_base64.append(base64_string)
            
            input = {
                "prompt": prompt,
                "image": images_base64,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "watermark": watermark,
            } 

            image_datas = model_instance.i2i(model, input)
            
            # 提取base64字符串列表
            base64_strings = [image_data.b64_json  for image_data in image_datas]
            
            # 获取第一张图片的尺寸信息
            width, height = extract_image_size(image_datas)
            
            # 转换图片
            images_tensor = ImageConverter.batch_base64_to_comfyui_images(base64_strings)
            return (images_tensor, width, height)
        except Exception as e:
            # 打印堆栈信息
            traceback.print_exc()
            print(f"调用模型[{self.platform} -- {model}]失败：", str(e))
            # 返回一个空的图像tensor作为错误处理
            return (torch.zeros((1, 512, 512, 3)),)


class CreateSeedanceVideo(io.ComfyNode):
    platform = "volcengine"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateSeedanceVideo",
            display_name="Create Seedance Video",
            category="NYJY/Volcengine",
            description="生成 Seedance 视频",
            inputs=[
                io.Combo.Input("model", ["doubao-seedance-1-5-pro-251215"], {
                    "default": "doubao-seedance-1-5-pro-251215"
                }),
                io.String.Input("prompt", tooltip="生成视频提示词", optional=True, multiline=True),
                io.Image.Input("first_frame", tooltip="The first frame of the video.", optional=True),
                io.Image.Input("last_frame", tooltip="The last frame of the video.", optional=True),
                io.String.Input("draft_task_id", tooltip="基于样片任务 ID，生成正式视频", default="", optional=True),
                io.Boolean.Input("return_last_frame", tooltip="是否返回生成视频的尾帧图像", default=False),
                io.Boolean.Input("generate_audio", tooltip="是否生成音频", default=True),
                io.Boolean.Input("draft", tooltip="是否开启样片模式，开启后，draft_task_id无效", default=False),
                io.Combo.Input("resolution", options=["480p", "720p", "1080p"],default= "720p", tooltip="视频分辨率，参考图场景不支持1080p"),
                io.Combo.Input("ratio", options=["adaptive", "1:1", "3:4", "4:3", "16:9", "9:16", "21:9"], 
                    default= "adaptive", tooltip="视频的宽高比例"),
                io.Int.Input("duration", tooltip="生成视频时长，单位：秒。支持 2~12 秒。", default=5, min=2, max=12, step=1),
                io.Int.Input("seed", default=-1, min=-1, max=2**32-1, control_after_generate=True),
                io.Boolean.Input("camera_fixed", tooltip="是否固定摄像头。参考图场景不支持", default=False),
                io.Boolean.Input("watermark", tooltip="是否包含水印", default=False),
                io.String.Input("api_key", tooltip="火山方舟api key", optional=True),
            ],
            outputs=[
                io.Video.Output(),
                io.Image.Output(display_name="last_frame", tooltip="视频尾帧图"),
                io.String.Output(display_name="video_url", tooltip="视频下载链接"),
                io.Int.Output(display_name="frame_rate", tooltip="视频帧率"),
                io.String.Output(display_name="video_id", tooltip="视频任务 ID"),
                io.String.Output(display_name="video_info", tooltip="视频任务信息"),
            ],
        )

    @classmethod
    async def execute(cls, model, return_last_frame, generate_audio, draft, resolution, ratio, duration, seed, camera_fixed, watermark, prompt=None, first_frame=None, last_frame=None, draft_task_id=None, api_key=None) -> io.NodeOutput:
        print("生成视频，时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))   

        (task_id, errmsg) = cls._create_task(model, return_last_frame, generate_audio, draft, resolution, ratio, duration, seed, camera_fixed, watermark, prompt, first_frame, last_frame, draft_task_id, api_key)
        if not task_id:
            print(f"创建视频任务失败：{errmsg}")
            return io.NodeOutput()

        # 循环查询视频任务结果
        while True:
            time.sleep(2)
            (return_data, errmsg) = cls._query_task_status(task_id, api_key)
            if not return_data:
                print(f"查询视频任务状态失败：{errmsg}")
                return io.NodeOutput()

            status = return_data.get("status", None)
            print(f"查询视频任务[{return_data.get("id", "")}]状态：{status}")
            if status == "succeeded":
                print(f"视频任务已完成：{return_data}")
                break
            elif status == "queued" or status == "running":
                continue
            elif status == "cancelled":
                print(f"视频任务已取消：{errmsg}")
                return io.NodeOutput()
            elif status == "failed":
                print(f"视频任务失败：{errmsg}")
                return io.NodeOutput()
            elif status == "expired":
                print(f"视频任务已过期：{errmsg}")
                return io.NodeOutput()
        

        video_url = return_data.get("content", {}).get("video_url", "")
        
        video_url = video_url.strip()
        video = await download_url_to_video_output(video_url)
       
        # last_frame_image 默认是个空图片 torch.Tensor 类型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        last_frame_image = torch.zeros(3, 64, 64, device=device)
        if return_data.get("content", {}).get("last_frame_url"):
            last_frame_image = await download_url_to_image_tensor(return_data.get("content", {}).get("last_frame_url", ""))

        # 读取video_path内容
        return io.NodeOutput(
            video,
            last_frame_image,
            video_url,
            return_data.get("framespersecond", 0),
            return_data.get("id", ""),
            json.dumps(return_data, ensure_ascii=False),
        )

    @classmethod
    def _create_task(cls, model, return_last_frame, generate_audio, draft, resolution, ratio, duration, seed, camera_fixed, watermark, prompt=None, first_frame=None, last_frame=None, draft_task_id=None, api_key=None):

        factory = AIModelBridgeFactory()
        model_instance = factory.get_model(cls.platform)
        if api_key != "":
            model_instance.set_config({"api_key": api_key})

        """创建视频生成任务"""
        
        # API 端点
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        create_url = f"{base_url}/contents/generations/tasks"
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_instance.get_config().get("api_key", "")}"
        }

        content = []
        
        # 添加提示词
        content.append({
            "type": "text",
            "text": prompt
        })
        if first_frame is not None:
            # 参考图场景添加首帧图片
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + tensor_to_base64_string(first_frame)
                },
                "role": "first_frame"
            })
        if last_frame is not None:
            # 参考图场景添加尾帧图片
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + tensor_to_base64_string(last_frame)
                },
                "role": "last_frame"
            })

        # 处理特殊情况的分辨率
        if draft:
            # 样片只支持480p
            resolution = "480p"
            return_last_frame = False
        elif (first_frame is not None or last_frame is not None) and resolution == "1080p":
            # 参考图场景不支持1080p，降级为720p
            resolution = "720p"
        elif draft_task_id is not None and draft_task_id != "":
            # 草稿任务场景添加草稿任务 ID
            content = [{
                "draft_task":{
                    "id": draft_task_id,
                },
                "type": "draft_task",
            }]
            generate_audio = False
            duration = None
            seed = None
            camera_fixed = None
            ratio = None

        request_body = {
            "model": model,
            "content": content,
            "return_last_frame": return_last_frame,
            "generate_audio": generate_audio,
            "draft": draft,
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "seed": seed,
            "camera_fixed": camera_fixed,
            "watermark": watermark,
        }

        try:
            # 发送创建任务请求
            print(f"正在创建视频生成任务...")
            if draft:
                print(f"草稿模式: 已启用")
            
            response = requests.post(
                create_url,
                headers=headers,
                json=request_body,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            # 获取任务 ID
            task_id = result.get("id")
            print(f"任务创建成功，任务ID: {task_id}")
            return (task_id, None)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\n详细错误: {json.dumps(error_detail, ensure_ascii=False)}"
                except:
                    error_msg += f"\n响应内容: {e.response.text}"
            return (None, error_msg)
    
    @classmethod
    def _query_task_status(cls, task_id, api_key):
        factory = AIModelBridgeFactory()
        model_instance = factory.get_model(cls.platform)
        if api_key != "":
            model_instance.set_config({"api_key": api_key})
        
        # API 端点
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        query_url = f"{base_url}/contents/generations/tasks/{task_id}"
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {model_instance.get_config().get("api_key", "")}"
        }
        
        while True:
            try:
                # 查询任务状态
                response = requests.get(
                    query_url,
                    headers=headers,
                    timeout=30
                )
                
                response.raise_for_status()
                result = response.json()

                return (result, None)
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"查询请求失败: {str(e)}"
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        error_msg += f"\n详细错误: {json.dumps(error_detail, ensure_ascii=False)}"
                    except:
                        error_msg += f"\n响应内容: {e.response.text}" 
                return (None, error_msg)
