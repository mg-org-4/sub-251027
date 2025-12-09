@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 检查ffmpeg是否安装
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到ffmpeg。请先安装ffmpeg并将其添加到系统PATH中。
    pause
    exit /b 1
)

:: 获取用户输入
echo 请输入视频文件路径,不能有引号(input file path):
set /p "video_path="
echo.

echo 请输入分割长度(帧数/split length):
set /p "frame_count="
echo.

:: 验证输入文件是否存在
if not exist "%video_path%" (
    echo 错误: 文件 "%video_path%" 不存在。
    pause
    exit /b 1
)

:: 提取文件信息
for %%F in ("%video_path%") do (
    set "file_dir=%%~dpF"
    set "file_name=%%~nF"
    set "file_ext=%%~xF"
)

:: 使用ffprobe获取总帧数和帧率
echo 正在获取视频信息...
:: 获取总帧数
for /f "delims=" %%F in ('ffprobe -v error -select_streams v:0 -count_frames -show_entries stream^=nb_frames -of default^=nokey^=1:noprint_wrappers^=1 "%video_path%"') do set "total_frames=%%F"

:: 获取帧率
for /f "delims=" %%F in ('ffprobe -v error -select_streams v:0 -show_entries stream^=r_frame_rate -of default^=nokey^=1:noprint_wrappers^=1 "%video_path%"') do set "fps=%%F"


:: 检查是否成功获取信息
if not defined total_frames (
    echo 错误: 无法获取视频信息
    pause
    exit /b 1
)

:: 创建输出目录
set "output_dir=%file_dir%%file_name%_split"
if not exist "%output_dir%" mkdir "%output_dir%"

echo 视频总帧数: %total_frames%
echo 视频帧率: %fps%

:: 计算每段时长（秒）
set /a "segment_time=frame_count*1000/fps"
set "segment_time=!segment_time:~0,-3!.!segment_time:~-3!"


:: 计算按指定帧数分割的段数
set /a "segments=total_frames/frame_count"
set /a "remainder=total_frames%%frame_count"

echo 可分割为 %segments% 段完整%frame_count%帧的片段
if %remainder% gtr 0 echo 最后一段将包含 %remainder% 帧

pause

:: 创建临时文件
set "temp_video=%file_dir%%file_name%_temp%file_ext%"
ffmpeg -i "%video_path%" -c:v libx264 -x264opts keyint=1:min-keyint=1 -c:a copy "%temp_video%"

:: 检查临时文件是否创建成功
if not exist "%temp_video%" (
    echo 错误: 创建临时文件失败
    pause
    exit /b 1
)

:: 使用ffmpeg分割视频
echo 开始分割视频...
set /a "segment_num=0"
:loop
set /a "start_frame=segment_num*frame_count"
set /a "end_frame=start_frame+frame_count-1"

if %start_frame% geq %total_frames% goto end

:: 处理最后一段
if %end_frame% geq %total_frames% set "end_frame=%total_frames%"

:: 格式化段号
set "formatted_num=00000!segment_num!"
set "formatted_num=!formatted_num:~-5!"

ffmpeg -i "%temp_video%" -vf "trim=start_frame=%start_frame%:end_frame=%end_frame%" -c:v libx264 -c:a copy "%output_dir%\v%formatted_num%%file_ext%"

set /a "segment_num+=1"
if %start_frame% lss %total_frames% goto loop
:end


:: 删除临时文件
del "%temp_video%"

echo.
echo 视频分割完成！
echo 输出目录: "%output_dir%"
pause
