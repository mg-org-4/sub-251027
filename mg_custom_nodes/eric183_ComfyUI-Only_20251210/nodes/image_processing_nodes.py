"""
图像处理相关节点
包含图片加载和workflow解析功能
"""

import json
import os
import re
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch
import folder_paths


class WorkflowParser:
    """
    一个可重用的工作流解析器，用于提取提示词和模型信息。
    """
    # def is_negative_prompt(self, text):
    #     """
    #     判断文本是否为negative prompt
    #     """
    #     text_lower = text.lower()
        
    #     # 首先检查明确的positive关键词
    #     positive_keywords = ["masterpiece", "best quality", "best"]
    #     for keyword in positive_keywords:
    #         if keyword in text_lower:
    #             return False  # 明确是positive
        
    #     # 然后检查明确的negative关键词
    #     negative_keywords = ["worst", "bad"]
    #     for keyword in negative_keywords:
    #         if keyword in text_lower:
    #             return True   # 明确是negative
        
    #     # 如果没有明确关键词，使用其他特征判断
    #     # 以lora标签开头通常是positive
    #     if text.strip().startswith("<lora:"):
    #         return False
        
    #     # 包含更多negative特征词汇
    #     extended_negative_keywords = [
    #         "low quality", "normal quality", "bad anatomy", "bad hands", 
    #         "watermark", "signature", "simple background", "transparent"
    #     ]
    #     for keyword in extended_negative_keywords:
    #         if keyword in text_lower:
    #             return True
        
    #     # 默认判断为positive（保守策略）
    #     return False

    def parse_workflow_data(self, workflow_data):
        """
        解析workflow JSON，提取提示词和检查点名称
        """
        positive_prompt = ""
        filtered_positive_prompt = ""
        negative_prompt = ""
        checkpoint_name = ""
        
        try:
            nodes = workflow_data.get("nodes", [])

            # 1. 新逻辑: 根据节点标题 "title" 查找正向提示词
            for node in nodes:
                title = node.get("title")
                widgets_values = node.get("widgets_values")
                properties = node.get("properties")

                # 检查是否为 alekpet 节点
                is_alekpet_node = False
                if properties and isinstance(properties, dict):
                    if properties.get("cnr_id") == "rgthree-comfy":
                        is_alekpet_node = True

                if is_alekpet_node:
                    # rgthree-comfy 节点从 widgets_values[0] 获取提示词
                    if widgets_values and isinstance(widgets_values, list) and len(widgets_values) > 0 and isinstance(widgets_values[0], str):
                        if title == "positive_prompt":
                            positive_prompt = widgets_values[0].strip()
                        elif title == "filtered_positive_prompt":
                            filtered_positive_prompt = widgets_values[0].strip()
                        elif title == "negative_prompt":
                            negative_prompt = widgets_values[0].strip()
                else:
                    # 原有逻辑，先尝试从 index 1 获取
                    if widgets_values and isinstance(widgets_values, list) and len(widgets_values) > 1 and isinstance(widgets_values[1], str):
                        if title == "positive_prompt":
                            positive_prompt = widgets_values[1].strip()
                        elif title == "filtered_positive_prompt":
                            filtered_positive_prompt = widgets_values[1].strip()
                        elif title == "negative_prompt":
                            negative_prompt = widgets_values[1].strip()
                    # 兼容逻辑：如果 index 1 失败，再尝试从 index 0 获取
                    elif widgets_values and isinstance(widgets_values, list) and len(widgets_values) > 0 and isinstance(widgets_values[0], str):
                        if title == "positive_prompt":
                            positive_prompt = widgets_values[0].strip()
                        elif title == "filtered_positive_prompt":
                            filtered_positive_prompt = widgets_values[0].strip()
                        elif title == "negative_prompt":
                            negative_prompt = widgets_values[0].strip()
            # # 2. 负向提示词提取逻辑 (基本不变)
            # for node in nodes:
            #     title = node.get("title")
            #     widgets_values = node.get("widgets_values")
            #     properties = node.get("properties")

            #     # 检查是否为 alekpet 节点
            #     is_alekpet_node = False
            #     if properties and isinstance(properties, dict):
            #         if properties.get("cnr_id") == "rgthree-comfy":
            #             is_alekpet_node = True

            #     if is_alekpet_node:
            #         if title == "negative_prompt":
            #             negative_prompt = widgets_values[0].strip()

            
            # negative_prompts_list = []
            # for node in nodes:
            #     # 负向提示词通常在没有特殊标题的 CLIPTextEncode 节点中
            #     if node.get("type") == "CLIPTextEncode" and not node.get("title"):
            #         widgets_values = node.get("widgets_values")
            #         if widgets_values and isinstance(widgets_values, list) and len(widgets_values) > 0:
            #             text_content = str(widgets_values[0])
            #             if self.is_negative_prompt(text_content):
            #                 negative_prompts_list.append(text_content.strip())
            
            # if negative_prompts_list:
            #     negative_prompt = ", ".join(list(dict.fromkeys(negative_prompts_list)))

            # 3. 寻找CheckpointLoaderSimple节点 (逻辑不变)
            for node in nodes:
                # 优先使用更通用的 'type' 属性判断
                if node.get("type") == "CheckpointLoaderSimple":
                    widgets_values = node.get("widgets_values", [])
                    if widgets_values and len(widgets_values) > 0 and isinstance(widgets_values[0], str):
                        checkpoint_name = widgets_values[0]
                        break # 找到第一个就停止

                # 兼容旧的 'Node name for S&R' 属性
                properties = node.get("properties", {})
                node_name = properties.get("Node name for S&R", "")
                if node_name == "CheckpointLoaderSimple":
                    widgets_values = node.get("widgets_values", [])
                    if widgets_values and len(widgets_values) > 0 and isinstance(widgets_values[0], str):
                        checkpoint_name = widgets_values[0]
                        break  # 找到第一个就停止
            
        except Exception as e:
            print(f"解析workflow时出错: {e}")
        
        return positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name


class WorkflowImageFileLoader:
    """
    图片文件加载节点，直接读取图片文件并解析workflow信息
    """
    
    def __init__(self):
        self.parser = WorkflowParser()
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取输入目录中的图片文件
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    files.append(f)
        
        return {
            "required": {
                "image_file": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "workflow_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional: Manually input workflow JSON if the image lacks workflow information."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "positive_prompt", "filtered_positive_prompt", "negative_prompt", "checkpoint_name", "workflow_info", "workflow_json_out")
    FUNCTION = "load_and_parse"
    CATEGORY = "only/Image"
    
    def load_and_parse(self, image_file, workflow_json=""):
        """
        加载图片文件并解析workflow信息
        """
        # 获取完整文件路径
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image_file)
        
        # 加载图片并转换为tensor
        try:
            with Image.open(image_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 转换为numpy数组并归一化
                img_array = np.array(img).astype(np.float32) / 255.0
                
                # 转换为ComfyUI的IMAGE格式 [H, W, C]
                output_image = torch.from_numpy(img_array).unsqueeze(0)
        
        except Exception as e:
            # 如果图片加载失败，创建一个黑色图片
            output_image = torch.zeros(1, 512, 512, 3)
            return (output_image, "", "", "", "", "", "")
        
        # 初始化输出
        positive_prompt = ""
        negative_prompt = ""
        checkpoint_name = ""
        workflow_info = "无workflow信息"
        raw_workflow_text = ""
        filtered_positive_prompt = ""
        
        try:
            workflow_data = None
            
            # 如果手动提供了workflow JSON，优先使用
            if workflow_json.strip():
                raw_workflow_text = workflow_json
                workflow_info = "使用手动输入的workflow"
            else:
                # 否则，尝试从图片中提取
                raw_workflow_text = self.extract_workflow_from_image(image_path)
                if raw_workflow_text:
                    workflow_info = "从图片中提取的workflow"
                else:
                    workflow_info = "图片中未找到workflow信息"
            
            # 如果获取到了任何形式的workflow文本，尝试解析
            if raw_workflow_text:
                try:
                    workflow_data = json.loads(raw_workflow_text)
                except json.JSONDecodeError as e:
                    workflow_info += f" - JSON格式错误: {str(e)}"
                    workflow_data = None #确保数据无效
            
            # 解析workflow数据
            if workflow_data:
                positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name = self.parser.parse_workflow_data(workflow_data)
                if positive_prompt or negative_prompt or checkpoint_name:
                    workflow_info += f" - 解析成功: Positive({len(positive_prompt)}字符), Negative({len(negative_prompt)}字符), Checkpoint({checkpoint_name})"
                else:
                    workflow_info += " - 未找到相关节点信息"
            
        except Exception as e:
            workflow_info = f"解析错误: {str(e)}"
        
        return (output_image, positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name, workflow_info, raw_workflow_text)
    
    def extract_workflow_from_image(self, image_path):
        """
        从图片文件中提取原始的workflow JSON字符串
        """
        try:
            with Image.open(image_path) as img:
                # 检查PNG文本信息
                if hasattr(img, 'text') and img.text:
                    # 查找workflow相关的键
                    workflow_keys = ['workflow', 'Workflow', 'ComfyUI_workflow', 'prompt', 'parameters']
                    for key in workflow_keys:
                        if key in img.text:
                            workflow_text = img.text[key]
                            try:
                                # 验证它是否是有效的JSON，但返回原始字符串
                                json.loads(workflow_text)
                                return workflow_text
                            except json.JSONDecodeError:
                                continue
                
                # 检查PNG info参数
                if hasattr(img, 'info') and img.info:
                    for key, value in img.info.items():
                        if 'workflow' in key.lower() or 'comfy' in key.lower():
                            if isinstance(value, str):
                                try:
                                    # 验证它是否是有效的JSON，但返回原始字符串
                                    json.loads(value)
                                    return value
                                except (json.JSONDecodeError, TypeError):
                                    continue
                
                # 检查EXIF数据（主要用于JPEG）
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if isinstance(value, str) and ('workflow' in value.lower() or 'comfy' in value.lower()):
                            try:
                                # 验证它是否是有效的JSON，但返回原始字符串
                                json.loads(value)
                                return value
                            except (json.JSONDecodeError, TypeError):
                                continue
        
        except Exception as e:
            print(f"提取workflow时出错: {e}")
        
        return None
    

class WorkflowJSONParser:
    """
    独立的Workflow JSON解析器节点
    """
    
    def __init__(self):
        self.parser = WorkflowParser()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste ComfyUI workflow JSON here"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "filtered_positive_prompt", "negative_prompt", "checkpoint_name", "parse_info")
    FUNCTION = "parse_workflow"
    CATEGORY = "only/Text"
    
    def parse_workflow(self, workflow_json):
        """
        解析workflow JSON并提取提示词和检查点名称
        """
        positive_prompt = ""
        negative_prompt = ""
        checkpoint_name = ""
        parse_info = ""
        filtered_positive_prompt = ""
        
        try:
            if not workflow_json.strip():
                parse_info = "请输入workflow JSON"
                return (positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name, parse_info)
            
            # 解析JSON
            workflow_data = json.loads(workflow_json)
            
            # 使用共享的解析器
            positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name = self.parser.parse_workflow_data(workflow_data)
            
            # 生成解析信息
            prompt_count = len(positive_prompt.split(',')) if positive_prompt else 0
            checkpoint_count = 1 if checkpoint_name else 0
            
            parse_info = f"找到 {prompt_count} 个提示词, {checkpoint_count} 个 checkpoint。"
            if positive_prompt:
                parse_info += f" Positive: {len(positive_prompt)} 字符。"
            if negative_prompt:
                parse_info += f" Negative: {len(negative_prompt)} 字符。"
            if checkpoint_name:
                parse_info += f" Checkpoint: {checkpoint_name}。"
            
        except json.JSONDecodeError as e:
            parse_info = f"JSON 格式错误: {str(e)}"
            return (positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name, parse_info)
        except Exception as e:
            parse_info = f"解析错误: {str(e)}"
            return (positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name, parse_info)
            
        return (positive_prompt, filtered_positive_prompt, negative_prompt, checkpoint_name, parse_info)
    

# 导出节点类
NODE_CLASS_MAPPINGS = {
    "WorkflowImageFileLoader": WorkflowImageFileLoader,
    "WorkflowJSONParser": WorkflowJSONParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowImageFileLoader": "Workflow Image Loader (File)",
    "WorkflowJSONParser": "Workflow JSON Parser",
} 