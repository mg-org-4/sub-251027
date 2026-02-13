[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 HeyGem 数字人节点

目前 (2025.05.22) 最好的开源数字人, 没有之一. 基本可生成全身, 动态, 任意分辨率数字人. **输入视频帧率要与合成帧率保持一致.**

![image](https://github.com/billwuhao/Comfyui_HeyGem/blob/main/images/2025-05-22_22-41-52.png)

## 📣 更新

[2025-05-25]⚒️: 修复 bug, 长视频自动截取匹配音频时长.

[2025-05-23]⚒️: 优化长视频 cpu 占用, 短视频可重复或 pingpong 到音频时长. 长视频自动截取匹配音频时长.

[2025-05-22]⚒️: 发布 v1.0.0.

## 节点安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/Comfyui_HeyGem.git
```

## WSL 和 Docker 安装

- Windows (以 X64 为例):

1, 安装 Windows 的 Linux 子系统: https://github.com/microsoft/WSL/releases (`wsl.2.5.7.0.x64.msi`). 已经安装的, 管理员权限执行 `wsl --update` 更新.

2, 安装 Docker: https://www.docker.com/ (下载 `AMD64` 版本). 安装完成后, 启动它:

![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/8.png)
![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/13.png)
![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/3.png)

镜像默认下载到 C 盘 **(大概需要 14g 空间)**, 可以在设置里修改为其他盘:

![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/7.png)

OK! 准备就绪, 每次运行节点, 先启动 docker 即可. 第一次运行节点, 需要下载镜像, 大概 30 分钟左右, 看网速. 安装不复杂, 就点击几下安装完软件即可, docker 是独立镜像环境, 不担心兼容问题, 还很少报错, 比其他插件安装还简单.

- Linux (以 Ubuntu 为例):

1, 安装 Docker: 执行 `docker --version` 查看是否安装, 没有安装的, 执行下列命令安装.
```
sudo apt update
sudo apt install docker.io
sudo apt install docker-compose
```

2, 安装驱动: 执行 `nvidia-smi` 查看是否安装, 没有安装的, 参考官方文档安装 (https://www.nvidia.cn/drivers/lookup/).

3, 安装 NVIDIA 容器工具包: 
  - 添加 NVIDIA 软件包存储库：
  ```
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  ```

  - 安装 NVIDIA 容器工具包：
  ```
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  ```

  - 配置 NVIDIA 容器运行时：
  ```
  sudo nvidia-ctk runtime configure --runtime=docker
  ```

  - 重启 Docker 守护进程以应用更改：
  ```
  sudo systemctl restart docker
  ```

OK! 准备就绪, 第一次运行节点, 需要下载镜像, 大概 30 分钟左右, 看网速.

## 鸣谢

- [Duix.Heygem](https://github.com/duixcom/Duix.Heygem) 
- https://github.com/duixcom/Duix.Heygem/blob/main/LICENSE