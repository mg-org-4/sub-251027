import asyncio
import random
import time
import threading
import hashlib
import base64
import os
import tempfile
import shutil
from io import BytesIO
from typing import List, Optional, Union, Tuple

import torch
import numpy as np
from PIL import Image
import imageio

from comfy.model_management import InterruptProcessingException
from ..services.vlm import VisionService
from ..utils.common import format_api_error, format_model_with_thinking, generate_request_id, log_prepare, log_error, TASK_VIDEO_CAPTION, SOURCE_NODE
from ..services.thinking_control import build_thinking_suppression
from .base import VLMNodeBase


class VideoCaptionNode(VLMNodeBase):
    """
    视频反推提示词节点
    分析输入视频或图像序列并生成描述性提示词
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 从config_manager获取系统提示词配置
        from ..config_manager import config_manager
        system_prompts = config_manager.get_system_prompts()

        # ---获取所有 video_prompts 作为选项---
        video_prompts = {}
        if system_prompts and 'video_prompts' in system_prompts:
            video_prompts = system_prompts['video_prompts']

        # 构建提示词模板选项（支持分类格式：类别/规则名称）
        prompt_template_options = []
        for key, value in video_prompts.items():
            # 过滤掉不在后端显示的规则
            show_in = value.get('showIn', ["frontend", "node"])
            if 'node' not in show_in:
                continue

            name = value.get('name', key)
            category = value.get('category', '')
            # 如果有分类，显示为 "类别/规则名称"，否则直接显示规则名称
            display_name = f"{category}/{name}" if category else name
            prompt_template_options.append(display_name)

        # 如果没有选项,添加一个默认选项
        if not prompt_template_options:
            prompt_template_options = ["默认视频反推提示词"]
        
        # ---动态获取VLM服务/模型列表---
        service_options = cls.get_vlm_service_options()
        default_service = service_options[0] if service_options else "智谱"

        return {
            "required": {
                "rule": (prompt_template_options, {"default": prompt_template_options[0] if prompt_template_options else "默认视频反推提示词", "tooltip": "💡Template Config: Settings -> ✨Prompt Assistant -> Rule Editor"}),
                "custom_rule": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable", "tooltip": "⚠️ Enable to use custom rule content below instead of preset"}),
                "custom_rule_content": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入临时规则内容,仅在启用'临时规则'时生效", "tooltip": "在此输入您的自定义规则内容"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "输入额外的具体要求，将与规则一起发送给模型", "tooltip": "输入额外的具体要求，将与规则一起发送给模型"}),
                "vlm_service": (service_options, {"default": default_service, "tooltip": "Select VLM service and model"}),
                "sampling_mode": (["Auto (Uniform)", "Manual (Indices)"], {"default": "Auto (Uniform)"}),
                "frame_count": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1, "tooltip": "💡Only for 'Auto' mode. Frame limits: GLM-4V≤5, GLM-4.6V≤100, Qwen-VL≤100, Gemini≤3000, Grok≤10"}),
                "manual_indices": ("STRING", {"default": "", "placeholder": "Input indices (e.g. 0,10,20) or range (e.g. 0-10)", "tooltip": "💡Only for 'Manual' mode. Supports comma-separated or range. Negative indices allowed."}),
                # Ollama Automatic VRAM Unload
                "ollama_auto_unload": ("BOOLEAN", {"default": True, "label_on": "Enable", "label_off": "Disable", "tooltip": "Auto unload Ollama model after generation"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "video": ("VIDEO",),
                "image_sequence": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("prompt_output", "preview_frames")
    FUNCTION = "analyze_video_content"
    CATEGORY = "✨Prompt Assistant"
    OUTPUT_NODE = False
    
    @classmethod
    def IS_CHANGED(cls, video=None, image_sequence=None, rule=None, custom_rule=None, custom_rule_content=None, user_prompt=None, vlm_service=None, sampling_mode=None, frame_count=None, manual_indices=None, ollama_auto_unload=None, seed=None, unique_id=None):
        """
        只在输入内容真正变化时才触发重新执行
        """
        # 提取实际的tensor数据
        target_input = None
        if video is not None:
            if isinstance(video, dict):
                target_input = video.get('frames') or video.get('video')
            elif hasattr(video, 'video'):
                target_input = video.video
            elif hasattr(video, 'frames'):
                target_input = video.frames
            elif isinstance(video, torch.Tensor):
                target_input = video
        elif image_sequence is not None:
            target_input = image_sequence
        
        # 计算哈希值
        input_data_hash = ""
        if target_input is not None and isinstance(target_input, torch.Tensor):
            try:
                # 取部分帧和中心区域计算哈希,避免全量计算
                if len(target_input.shape) == 4:
                    frames = target_input.shape[0]
                    # 取首尾和中间帧
                    indices = [0, frames // 2, frames - 1] if frames > 2 else range(frames)
                    
                    hash_data = b""
                    for idx in indices:
                        h, w = target_input.shape[1:3]
                        center_h, center_w = h // 2, w // 2
                        size = min(50, h // 4, w // 4)
                        frame_data = target_input[idx,
                                          max(0, center_h - size):min(h, center_h + size),
                                          max(0, center_w - size):min(w, center_w + size),
                                          0].cpu().numpy().tobytes()
                        hash_data += frame_data
                    
                    input_data_hash = hashlib.md5(hash_data).hexdigest()
                else:
                    input_data_hash = "invalid_shape"
            except Exception:
                input_data_hash = "hash_error"

        # 组合所有输入的哈希值
        input_hash = hash((
            input_data_hash,
            rule,
            bool(custom_rule),
            custom_rule_content,
            user_prompt,
            vlm_service,
            sampling_mode,
            frame_count,
            manual_indices,
            bool(ollama_auto_unload),
            seed
        ))

        return input_hash
    
    def analyze_video_content(self, rule, custom_rule, custom_rule_content, user_prompt, vlm_service, sampling_mode, frame_count, manual_indices, ollama_auto_unload, seed=None, video=None, image_sequence=None, unique_id=None):
        """
        分析视频或图像序列并生成提示词(使用抽帧模式)
        """
        temp_video_path = None
        try:
            # 1. 验证输入
            if video is None and image_sequence is None:
                raise ValueError("Video Input or Image Sequence Input is required")
            
            # 2. 提取tensor数据
            input_tensor = None
            is_pre_sampled = False
            
            if video is not None:
                # 处理VideoFromFile对象(需要先保存再加载)
                if hasattr(video, 'save_to') and callable(getattr(video, 'save_to')):
                    try:
                        fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
                        os.close(fd)
                        video.save_to(temp_video_path)
                        print(f"{self.PROCESS_PREFIX} VideoFromFile saved to temp")
                        #从文件加载为tensor,传入抽帧参数进行预采样优化
                        input_tensor = self._load_video_as_tensor(
                            temp_video_path, 
                            target_count=frame_count if sampling_mode == "Auto (Uniform)" else None,
                            target_indices_str=manual_indices if sampling_mode == "Manual (Indices)" else None
                        )
                        is_pre_sampled = True
                    except Exception as e:
                        if temp_video_path and os.path.exists(temp_video_path):
                            os.unlink(temp_video_path)
                        raise RuntimeError(f"VideoFromFile processing failed: {str(e)}")
                
                # 处理字典格式
                elif isinstance(video, dict):
                    input_tensor = video.get('frames') or video.get('video')
                    if input_tensor is None:
                        for v in video.values():
                            if isinstance(v, torch.Tensor):
                                input_tensor = v
                                break
                
                # 处理tensor
                elif isinstance(video, torch.Tensor):
                    input_tensor = video
                
                # 尝试索引访问
                elif hasattr(video, '__getitem__'):
                    try:
                        first_item = video[0]
                        if isinstance(first_item, torch.Tensor):
                            input_tensor = first_item
                        elif isinstance(first_item, dict):
                            input_tensor = first_item.get('frames') or first_item.get('video')
                    except Exception:
                        pass
                
                if input_tensor is None:
                    raise ValueError(f"Failed to extract tensor from VIDEO input")
            else:
                input_tensor = image_sequence
            
            # 3. 准备提示词
            prompt_template = None
            rule_name = "Custom Rule" if (custom_rule and custom_rule_content) else rule
            
            if custom_rule and custom_rule_content:
                prompt_template = custom_rule_content
            else:
                from ..config_manager import config_manager
                system_prompts = config_manager.get_system_prompts()
                video_prompts = {}
                if system_prompts and 'video_prompts' in system_prompts:
                    video_prompts = system_prompts['video_prompts']

                # 查找模板（按显示名称匹配）
                # 显示名称格式：有分类时为 "类别/规则名称"，无分类时为 "规则名称"
                for key, value in video_prompts.items():
                    name = value.get('name', key)
                    category = value.get('category', '')
                    # 构建与下拉列表一致的显示名称
                    display_name = f"{category}/{name}" if category else name
                    if display_name == rule:
                        prompt_template = value.get('content')
                        break
                
                # 允许用规则名称或键名直接匹配（兼容旧格式）
                if not prompt_template:
                    for key, value in video_prompts.items():
                        if value.get('name') == rule or key == rule:
                            prompt_template = value.get('content')
                            break
                
                if not prompt_template:
                    prompt_template = "请详细描述这段视频的内容,包括主要事件、场景变化、人物动作和视觉风格。"
                    rule_name = "Default Rule"

            # 拼接用户提示词
            if user_prompt and user_prompt.strip():
                prompt_template = f"{prompt_template}\n\n用户补充要求：\n{user_prompt}"

            # ---解析服务/模型字符串---
            service_id, model_name = self.parse_service_model(vlm_service)
            if not service_id:
                raise ValueError(f"Invalid service selection: {vlm_service}")
            
            # ---获取服务配置---
            from ..config_manager import config_manager
            service = config_manager.get_service(service_id)
            if not service:
                raise ValueError(f"Service config not found: {vlm_service}")
            
            # ---构建provider_config---
            # 查找指定的模型或默认模型
            vlm_models = service.get('vlm_models', [])
            target_model = None
            
            if model_name:
                # 查找指定的模型
                target_model = next((m for m in vlm_models if m.get('name') == model_name), None)
            
            if not target_model:
                # 使用默认模型或第一个模型
                target_model = next((m for m in vlm_models if m.get('is_default')), 
                                    vlm_models[0] if vlm_models else None)
            
            if not target_model:
                raise ValueError(f"Service {vlm_service} has no available models")
            
            # 构建配置对象
            provider_config = {
                'provider': service_id,
                'model': target_model.get('name', ''),
                'base_url': service.get('base_url', ''),
                'api_key': service.get('api_key', ''),
                'temperature': target_model.get('temperature', 0.7),
                'max_tokens': target_model.get('max_tokens', 500),
                'top_p': target_model.get('top_p', 0.9),
            }
            
            # Ollama特殊处理:添加auto_unload配置
            if service.get('type') == 'ollama':
                provider_config['auto_unload'] = ollama_auto_unload
            
            model = provider_config.get('model', '')
            
            # 5. 抽帧处理
            request_id = generate_request_id("vcap", None, unique_id)
            # 检查是否关闭思维链
            disable_thinking_enabled = service.get('disable_thinking', True)
            thinking_extra = build_thinking_suppression(service_id, model) if disable_thinking_enabled else None
            model_display = format_model_with_thinking(model, bool(thinking_extra))
            
            # 获取服务显示名称
            service_display_name = service.get('name', service_id)
            
            # 准备阶段日志
            log_prepare(TASK_VIDEO_CAPTION, request_id, SOURCE_NODE, service_display_name, model_display, rule_name, {"模式": sampling_mode})
            
            # [Debug] 输出抽帧参数详情
            # print(f"{self.PROCESS_PREFIX} [video-caption-debug] 输入tensor形状:{input_tensor.shape} | is_pre_sampled:{is_pre_sampled}")
            # print(f"{self.PROCESS_PREFIX} [video-caption-debug] sampling_mode:{sampling_mode} | frame_count:{frame_count} | manual_indices:{manual_indices}")
            
            # 准备抽帧参数
            sampling_kwargs = {}    
            if not is_pre_sampled:
                if sampling_mode == "Auto (Uniform)":
                    sampling_kwargs['target_count'] = frame_count
                elif sampling_mode == "Manual (Indices)":
                    sampling_kwargs['target_indices_str'] = manual_indices
            # 从tensor中提取帧并转为base64,同时获取预览tensor
            frames_data, preview_tensor = self._extract_frames_and_tensor(
                input_tensor, 
                **sampling_kwargs
            )
            
            # [Debug] 输出抽帧结果
            # print(f"{self.PROCESS_PREFIX} [video-caption] 抽帧完成 | 帧数量:{len(frames_data)} | 预览tensor:{preview_tensor.shape}")
            
            # ---注入帧数元信息到提示词---
            # 解决模型识别帧数与实际帧数不一致的问题
            actual_frame_count = len(frames_data)
            frame_info_prefix = f"[重要提示：本次共提供了 {actual_frame_count} 帧图像，请务必逐帧分析，确保输出的描述数量与帧数一致。]\n\n"
            prompt_template = frame_info_prefix + prompt_template
            
            # 调用多图像分析 - 使用基类方法
            result = self._run_vision_task(
                VisionService.analyze_images,
                service_id,
                images_data=frames_data,
                request_id=request_id,
                prompt_content=prompt_template,
                custom_provider=service_id,
                custom_provider_config=provider_config,
                task_type=TASK_VIDEO_CAPTION,
                source=SOURCE_NODE
            )

            # 6. 处理结果
            if result and result.get('success'):
                description = result.get('data', {}).get('description', '').strip()
                if not description:
                    raise RuntimeError("API returned empty result")
                return (description, preview_tensor)
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                # 如果是中断错误,直接抛出InterruptProcessingException,不打印日志(由基类打印)
                if error_msg == "任务被中断":
                    raise InterruptProcessingException()
                raise RuntimeError(f"Analysis failed: {error_msg}")

        except InterruptProcessingException:
            # 不打印日志,由基类统一打印
            raise
        except Exception as e:
            error_msg = format_api_error(e, vlm_service)
            log_error(TASK_VIDEO_CAPTION, request_id, error_msg, source=SOURCE_NODE)
            raise RuntimeError(f"Analysis error: {error_msg}")
        finally:
            # 清理临时视频文件
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except Exception:
                    pass

    def _uniform_sample(self, l, n):
        """
        从列表中均匀采样 n 个元素 (参考 video_sampling_guide.md)
        算法: 将列表分成 n 个等长区间,从每个区间的中心位置取样
        """
        if n >= len(l):
            return l
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        # 确保索引不越界
        idxs = [min(i, len(l) - 1) for i in idxs]
        return [l[i] for i in idxs]

    def _parse_frame_indices(self, indices_str, total_frames):
        """解析手动输入的帧索引字符串"""
        indices = set()
        if not indices_str:
            return []
            
        parts = indices_str.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                if '-' in part:
                    # Range: 0-10
                    start_str, end_str = part.split('-')
                    start = int(start_str)
                    end = int(end_str)
                    # Handle negative indices
                    if start < 0: start += total_frames
                    if end < 0: end += total_frames
                    
                    start = max(0, min(start, total_frames - 1))
                    end = max(0, min(end, total_frames - 1))
                    
                    if start <= end:
                        indices.update(range(start, end + 1))
                else:
                    # Single index
                    idx = int(part)
                    if idx < 0: idx += total_frames
                    idx = max(0, min(idx, total_frames - 1))
                    indices.add(idx)
            except ValueError:
                print(f"{self.LOG_PREFIX} 忽略无效的帧索引格式: {part}")
                
        return sorted(list(indices))

    def _extract_frames_and_tensor(self, tensor, target_count=None, target_indices_str=None):
        """从tensor中提取指定数量或指定索引的帧,返回(base64列表, 预览tensor)"""
        total_frames = tensor.shape[0]
        
        # 生成所有帧的索引列表
        all_indices = list(range(total_frames))
        
        selected_indices = []
        if target_indices_str is not None:
            # 手动模式
            selected_indices = self._parse_frame_indices(target_indices_str, total_frames)
            if not selected_indices:
                print(f"{self.PROCESS_PREFIX} ⚠️ 手动帧索引无效或为空,回退到自动采样")
                selected_indices = self._uniform_sample(all_indices, 8) # Default fallback
        elif target_count is not None:
            # 自动模式
            selected_indices = self._uniform_sample(all_indices, target_count)
        else:
            # 默认全量
            selected_indices = all_indices
        
        # 提取选中的帧 [N, H, W, C]
        selected_tensor = tensor[selected_indices]
        
        frames_base64 = []
        # 遍历选中的tensor进行转换
        for i in range(selected_tensor.shape[0]):
            frame_tensor = selected_tensor[i]
            # 转为numpy [H, W, C] (0-255)
            frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
            # 转PIL
            image = Image.fromarray(frame_np)
            # 转Base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            frames_base64.append(f"data:image/jpeg;base64,{encoded}")
            
        return frames_base64, selected_tensor

    def _load_video_as_tensor(self, video_path, target_count=None, target_indices_str=None):
        """
        从视频文件加载为tensor
        优化: 如果指定了target_count或target_indices_str,则只读取需要的帧 (Sample before Load)
        """
        try:
            import imageio
            # 读取视频
            reader = imageio.get_reader(video_path, 'ffmpeg')
            
            # 获取总帧数
            # 尝试从元数据获取,如果失败则使用count_frames (较慢)
            try:
                total_frames = reader.count_frames()
            except Exception:
                # 如果无法获取帧数,回退到读取所有帧
                print(f"{self.PROCESS_PREFIX} 无法获取视频总帧数,将读取所有帧")
                frames = []
                for frame in reader:
                    frame_float = frame.astype(np.float32) / 255.0
                    frames.append(frame_float)
                reader.close()
                if not frames:
                    raise RuntimeError("视频文件中没有帧")
                frames_array = np.stack(frames, axis=0)
                tensor = torch.from_numpy(frames_array)
                
                # 如果有采样要求,进行后处理采样
                if target_count is not None or target_indices_str is not None:
                    print(f"{self.PROCESS_PREFIX} 视频加载回退: 读取全量帧后进行采样")
                    _, tensor = self._extract_frames_and_tensor(tensor, target_count, target_indices_str)
                    
                return tensor

            # 计算需要读取的帧索引
            indices_to_read = list(range(total_frames))
            
            if target_indices_str is not None:
                indices_to_read = self._parse_frame_indices(target_indices_str, total_frames)
                print(f"{self.PROCESS_PREFIX} 视频加载优化(手动): 从 {total_frames} 帧中提取 {len(indices_to_read)} 帧")
            elif target_count and total_frames > target_count:
                indices_to_read = self._uniform_sample(indices_to_read, target_count)
                print(f"{self.PROCESS_PREFIX} 视频加载优化(自动): 从 {total_frames} 帧中采样 {len(indices_to_read)} 帧")
            
            # 优化读取: 只读取需要的帧
            frames = []
           
            # imageio的get_data(index)支持随机访问
            for idx in indices_to_read:
                try:
                    frame = reader.get_data(idx)
                    frame_float = frame.astype(np.float32) / 255.0
                    frames.append(frame_float)
                except IndexError:
                    break
            
            reader.close()
            
            if not frames:
                raise RuntimeError("未能读取到有效帧")
            
            # 转换为torch tensor [N, H, W, C]
            frames_array = np.stack(frames, axis=0)
            tensor = torch.from_numpy(frames_array)
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"视频文件加载失败: {str(e)}")

# 节点映射，用于ComfyUI注册节点
NODE_CLASS_MAPPINGS = {
    "VideoCaptionNode": VideoCaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCaptionNode": "✨Video Caption",
}
