import os
import subprocess
from ..func import has_audio,getVideoInfo,set_file_name,video_type
import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

class PipVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "video1_path": ("STRING", {"default":"C:/Users/Desktop/video1.mp4", "tooltip": "说明：画中画背景画面！"}),
                "video2_path": ("STRING", {"default":"C:/Users/Desktop/video2.mp4", "tooltip": "说明：画中画前景画面！"}),
                "device": (["cpu","cuda"], {"default":device,}),
                "use_audio": (["video1","video2"], {"default":"video1", "tooltip": "说明：最终视频使用哪个视频的音轨！"}),
                "use_duration": (["video1","video2"], {"default":"video2", "tooltip": "说明：使用哪个视频作为最终参考时长！"}),
                "align_type":(["top-left","top-right", "bottom-left", "bottom-right", "center"], {"default":"center",}),
                "pip_fg_zoom": ("FLOAT", { "default": 2.5, "min": 1, "max": 100, "step": 0.5, "tooltip": "说明：画中画背景缩放系数，越大前景画面越小，值为背景宽高的缩小倍数！"}),
                "output_path": ("STRING", {"default": "C:/Users/Desktop/output"}),
                "scale_and_crop": (["none","540*960", "960*540"], {"default": "none", "tooltip": "说明：缩放和裁剪比例！"}),  # 新增参数控制缩放裁剪
                "fps": ("FLOAT", {"min": 0, "max": 60, "step": 0.1, "default": 30.0, "tooltip": "说明：画中画合并后的强制帧率，设置为0将使用video2的帧率，设置为1为将使用video2的帧率！"}),
                "is_chromakey": ("BOOLEAN", { "default": False ,"label_on": "绿幕去背景", "label_off": "关闭绿幕透明", "tooltip": "说明：是否进行绿幕去背景！"}),  #是否画中画 绿幕透明
            },
        }

    RETURN_TYPES = ("STRING","INT","INT","FLOAT","FLOAT",)
    RETURN_NAMES = ("video_complete_paths","width","height","duration","fps",)
    FUNCTION = "pip_video"
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg"
    DESCRIPTION = """两个视频叠加成一个画中画效果，可以控制前景video2出现在前景video1画面上的位置， 
                     可以设置前景画面的缩放系数和是否去掉绿幕背景等设置."""

    def pip_video(self, video1_path, video2_path,device,use_audio,use_duration, align_type,pip_fg_zoom, output_path,scale_and_crop,fps,is_chromakey):
        try:
            video1_path = os.path.abspath(video1_path).strip()
            video2_path = os.path.abspath(video2_path).strip()
            output_path = os.path.abspath(output_path).strip()
             # 视频不存在
            if not video1_path.lower().endswith(video_type()):
                raise ValueError("video1_path："+video1_path+"不是视频文件（video1_path:"+video1_path+" is not a video file）")
            if not os.path.isfile(video1_path):
                raise ValueError("video1_path："+video1_path+"不存在（video1_path:"+video1_path+" does not exist）")

            if not video2_path.lower().endswith(video_type()):
                raise ValueError("video2_path："+video2_path+"不是视频文件（video2_path:"+video2_path+" is not a video file）")
            if not os.path.isfile(video2_path):
                raise ValueError("video2_path："+video2_path+"不存在（video2_path:"+video2_path+" does not exist）")

            #判断output_path是否是一个目录
            if not os.path.isdir(output_path):
                raise ValueError("output_path："+output_path+"不是目录（output_path:"+output_path+" is not a directory）")

            video1_audio = has_audio(video1_path)
            video2_audio = has_audio(video2_path)

            final_output = set_file_name(video1_path)
            #文件名根据年月日时分秒来命名
            output_path = os.path.join(output_path, final_output)

            use_cuvid = ""
            use_encoder = "-c:v libx264" #默认用CPU编码

            if device == "cuda":
                use_cuvid = "-hwaccel cuda"
                use_encoder = "-c:v h264_nvenc"

            video_info = getVideoInfo(video1_path)
            video_info1 = getVideoInfo(video2_path)
            if use_duration == "video1":
                duration_1 = video_info['duration']
            else:
                duration_1 = video_info1['duration']
            if fps==0:
                fps=video_info['fps']
            elif fps==1:
                fps=video_info1['fps']
                
            # libx264、libx265等编码器要求宽高必须是2的倍数，如果报错，可以把device换为GPU
            width =  math.ceil(video_info['width']/2)*2
            height = math.ceil(video_info['height']/2)*2

            use_audio_index = {
                'video1': '0',
                'video2': '1',
            }.get(use_audio, '0')

            align_position = {
                "top-left": f"0:0",
                "top-right": f"(W-w):0",
                "bottom-left": f"0:(H-h)",
                "bottom-right": f"(W-w):(H-h)",
                "center": f"(W-w)/2:(H-h)/2",
            }.get(align_type, f"(W-w)/2:(H-h)/2")
            
            if height*540/width>960: #如果高同比缩放高度超出960，需要裁剪
                pad_or_crop1='crop=540:960:(ow-iw)/2:(oh-ih)/2'
            else: #比固定高小，需要填充黑边
                pad_or_crop1='pad=540:960:(ow-iw)/2:(oh-ih)/2:color=black'
            if height*960/width>540: #如果缩放到宽960时，高超出540，需要裁剪
                pad_or_crop2='crop=960:540:(ow-iw)/2:(oh-ih)/2'
            else:
                pad_or_crop2='pad=960:540:(ow-iw)/2:(oh-ih)/2:color=black'
            scale_and_crop_data = {
                'none': 'null',
                '540*960': f'scale=540:-1,setsar=1,{pad_or_crop1}',
                '960*540': f'scale=960:-1,setsar=1,{pad_or_crop2}',
            }.get(scale_and_crop, 'null')
            
            video2_width = {
                'none': f'{width}',
                '540*960': '540',
                '960*540': '960',
            }.get(scale_and_crop, f'{width}')
            
            final_out = {
                'none': f'scale={width}:{height}:force_original_aspect_ratio=disable,setsar=1',
                '540*960': 'scale=540:960:force_original_aspect_ratio=disable,setsar=1',
                '960*540': 'scale=960:540:force_original_aspect_ratio=disable,setsar=1',
            }.get(scale_and_crop, f'scale={width}:{height}:force_original_aspect_ratio=disable,setsar=1')
            
            #测试去绿幕效果
            if is_chromakey:
                chromakey="chromakey=0x00FF00:0.3:0.1,format=yuva420p"
            else:
                chromakey="null"

            
            if video1_audio or video2_audio:
                #-map 1:a 指定使用第二个视频的音频流
                command = fr'ffmpeg "-y" {use_cuvid} -stream_loop -1 -i "{video1_path}" -stream_loop -1 -i "{video2_path}" -filter_complex "[0:v]fps={fps},setpts=PTS-STARTPTS[bg];[1:v]fps={fps},setpts=PTS-STARTPTS[fg];[bg]{scale_and_crop_data}[bg_out];[fg]{chromakey}[fgd];[fgd]scale={video2_width}/{pip_fg_zoom}:-1,setsar=1[fg_out];[bg_out][fg_out]overlay={align_position}[out];[out]{final_out}[final_out]" -map "[final_out]" -map {use_audio_index}:a? {use_encoder} -c:a aac -t {duration_1} "{output_path}"'
            else:
                command = fr'ffmpeg "-y" {use_cuvid} -stream_loop -1 -i "{video1_path}" -stream_loop -1 -i "{video2_path}" -filter_complex "[0:v]fps={fps},setpts=PTS-STARTPTS[bg];[1:v]fps={fps},setpts=PTS-STARTPTS[fg];[bg]{scale_and_crop_data}[bg_out];[fg]{chromakey}[fgd];[fgd]scale={video2_width}/{pip_fg_zoom}:-1,setsar=1[fg_out];[bg_out][fg_out]overlay={align_position}[out];[out]{final_out}[final_out]" -map "[final_out]" -t {duration_1} "{output_path}"'
                
            print(f">>>{command}")

            # 执行命令并检查错误
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            #print("command result",result.returncode)

            # 检查返回码
            if result.returncode != 0:
                # 如果有错误，输出错误信息
                print(f"Error: {result.stderr.decode('utf-8')}")
                if device == "cuda":
                    print(f"***当前运算模式*[{device}]*************看下换成CPU重新执行，是否解决因编码问题的报错！********") 
                    self.pip_video(video1_path, video2_path,"cpu",use_audio,use_duration, align_type,pip_fg_zoom, os.path.dirname(output_path),scale_and_crop,fps,is_chromakey)
                
            else:
                # 输出标准输出信息
                print(f">>FFmpeg 执行完毕！Completed!\t stdout: {result.stdout}")

            return (output_path,width,height,duration_1,fps,)
        except Exception as e:
            raise ValueError(e)

#a=StitchingVideo()
#a.stitching_video("C:/Users/wtc/Desktop/tt/l.mp4", "C:/Users/wtc/Desktop/tt/r.mp4", "cpu", "video2",  "horizontal","picture-picture", "bottom-left", "C:/Users/wtc/Desktop/tt",  "yes")