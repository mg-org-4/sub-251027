import os
import subprocess
from ..func import set_file_name,video_type

class VideoPlayback:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "video_path": ("STRING", {"default":"C:/Users/Desktop/video.mp4",}),
                "output_path": ("STRING", {"default":"C:/Users/Desktop/output",}),
                "reverse_audio": (["True", "False"], {"default": "True"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_complete_path",)
    FUNCTION = "video_playback"
    OUTPUT_NODE = True
    CATEGORY = "🔥FFmpeg"
  
    def video_playback(self, video_path, output_path, reverse_audio):
        try:
            video_path = os.path.abspath(video_path).strip()
            output_path = os.path.abspath(output_path).strip()
             # 视频不存在
            if not video_path.lower().endswith(video_type()):
                raise ValueError("video_path："+video_path+"不是视频文件（video_path:"+video_path+" is not a video file）")
            if not os.path.isfile(video_path):
                raise ValueError("video_path："+video_path+"不存在（video_path:"+video_path+" does not exist）")
            
            #判断output_path是否是一个目录
            if not os.path.isdir(output_path):
                raise ValueError("output_path："+output_path+"不是目录（output_path:"+output_path+" is not a directory）")
            
            file_name = set_file_name(video_path)
            
            output_path = os.path.join(output_path, file_name)
            
            # 构建FFmpeg命令
            command = ['ffmpeg', '-i', video_path, '-vf', 'reverse']
            
            # 根据用户选择决定是否倒放音频
            if reverse_audio == "True":
                command.extend(['-af', 'areverse'])
            
            command.append(output_path)
            
            # 执行命令并检查错误
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            # 检查返回码
            if result.returncode != 0:
                # 如果有错误，输出错误信息
                 print(f"Error: {result.stderr.decode('utf-8')}")
                 raise ValueError(f"Error: {result.stderr.decode('utf-8')}")
            else:
                # 输出标准输出信息
                print(result.stdout)

            return (output_path,)
        except Exception as e:
            raise ValueError(e)