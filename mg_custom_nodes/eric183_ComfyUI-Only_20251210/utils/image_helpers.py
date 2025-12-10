"""
图像处理辅助函数
提供图片加载、workflow提取等功能
"""

import json
import base64
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngInfo
import io


def extract_workflow_from_image(image_path):
    """
    从PNG图片中提取workflow信息
    ComfyUI生成的图片通常在PNG的元数据中包含workflow信息
    """
    try:
        with Image.open(image_path) as img:
            # 检查PNG信息
            if hasattr(img, 'text'):
                # 查找workflow相关的键
                workflow_keys = ['workflow', 'Workflow', 'ComfyUI_workflow']
                for key in workflow_keys:
                    if key in img.text:
                        workflow_text = img.text[key]
                        try:
                            return json.loads(workflow_text)
                        except json.JSONDecodeError:
                            continue
            
            # 检查EXIF数据
            if hasattr(img, '_getexif'):
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if 'workflow' in str(tag).lower() or 'comfy' in str(tag).lower():
                            try:
                                return json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                continue
    
    except Exception as e:
        print(f"提取workflow时出错: {e}")
    
    return None


def extract_prompt_from_image(image_path):
    """
    从图片中提取prompt信息
    """
    try:
        with Image.open(image_path) as img:
            if hasattr(img, 'text'):
                # 查找prompt相关的键
                prompt_keys = ['prompt', 'Prompt', 'positive', 'negative']
                prompts = {}
                for key in prompt_keys:
                    if key in img.text:
                        prompts[key] = img.text[key]
                return prompts
    
    except Exception as e:
        print(f"提取prompt时出错: {e}")
    
    return {}


def image_to_tensor(image_path):
    """
    将图片转换为ComfyUI兼容的tensor格式
    """
    import torch
    import numpy as np
    
    try:
        with Image.open(image_path) as img:
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 归一化到0-1
            img_array = img_array.astype(np.float32) / 255.0
            
            # 转换为torch tensor，并添加batch维度
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            return img_tensor
    
    except Exception as e:
        print(f"图片转tensor时出错: {e}")
        return None


def validate_workflow_json(workflow_text):
    """
    验证workflow JSON格式是否正确
    """
    try:
        workflow_data = json.loads(workflow_text)
        
        # 检查基本结构
        if not isinstance(workflow_data, dict):
            return False, "Workflow必须是JSON对象"
        
        if "nodes" not in workflow_data:
            return False, "Workflow缺少nodes字段"
        
        if not isinstance(workflow_data["nodes"], list):
            return False, "nodes字段必须是数组"
        
        return True, "Workflow格式正确"
    
    except json.JSONDecodeError as e:
        return False, f"JSON格式错误: {str(e)}"
    except Exception as e:
        return False, f"验证错误: {str(e)}"


def find_alekpet_nodes(workflow_data):
    """
    在workflow中查找alekpet节点
    """
    alekpet_nodes = []
    
    try:
        nodes = workflow_data.get("nodes", [])
        
        for node in nodes:
            properties = node.get("properties", {})
            cnr_id = properties.get("cnr_id", "")
            
            if cnr_id == "comfyui_custom_nodes_alekpet":
                node_info = {
                    "id": node.get("id", "unknown"),
                    "type": node.get("type", "Unknown"),
                    "widgets_values": node.get("widgets_values", []),
                    "properties": properties
                }
                alekpet_nodes.append(node_info)
    
    except Exception as e:
        print(f"查找alekpet节点时出错: {e}")
    
    return alekpet_nodes


def extract_prompts_from_alekpet_nodes(alekpet_nodes):
    """
    从alekpet节点中提取prompt
    """
    positive_prompts = []
    negative_prompts = []
    
    def is_negative_prompt(text):
        """
        判断文本是否为negative prompt
        """
        text_lower = text.lower()
        
        # 首先检查明确的positive关键词
        positive_keywords = ["masterpiece", "best quality", "best"]
        for keyword in positive_keywords:
            if keyword in text_lower:
                return False  # 明确是positive
        
        # 然后检查明确的negative关键词
        negative_keywords = ["worst", "bad"]
        for keyword in negative_keywords:
            if keyword in text_lower:
                return True   # 明确是negative
        
        # 如果没有明确关键词，使用其他特征判断
        # 以lora标签开头通常是positive
        if text.strip().startswith("<lora:"):
            return False
        
        # 包含更多negative特征词汇
        extended_negative_keywords = [
            "low quality", "normal quality", "bad anatomy", "bad hands", 
            "watermark", "signature", "simple background", "transparent"
        ]
        for keyword in extended_negative_keywords:
            if keyword in text_lower:
                return True
        
        # 默认判断为positive（保守策略）
        return False
    
    for node_info in alekpet_nodes:
        widgets_values = node_info.get("widgets_values", [])
        
        if widgets_values and len(widgets_values) > 0:
            text_content = widgets_values[0]
            
            # 判断是positive还是negative
            if is_negative_prompt(text_content):
                negative_prompts.append(text_content)
            else:
                positive_prompts.append(text_content)
    
    return positive_prompts, negative_prompts


def create_workflow_summary(workflow_data):
    """
    创建workflow摘要信息
    """
    try:
        nodes = workflow_data.get("nodes", [])
        total_nodes = len(nodes)
        
        # 统计节点类型
        node_types = {}
        alekpet_count = 0
        
        for node in nodes:
            node_type = node.get("type", "Unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            properties = node.get("properties", {})
            cnr_id = properties.get("cnr_id", "")
            if cnr_id == "comfyui_custom_nodes_alekpet":
                alekpet_count += 1
        
        summary = f"Workflow包含 {total_nodes} 个节点"
        if alekpet_count > 0:
            summary += f"，其中 {alekpet_count} 个alekpet节点"
        
        # 添加常见节点类型信息
        common_types = ["KSampler", "CheckpointLoaderSimple", "CLIPTextEncode", "VAEDecode"]
        found_types = [t for t in common_types if t in node_types]
        
        if found_types:
            summary += f"。包含: {', '.join(found_types)}"
        
        return summary
    
    except Exception as e:
        return f"生成摘要时出错: {str(e)}" 