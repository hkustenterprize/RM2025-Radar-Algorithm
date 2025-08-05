# Environment Installation Checklist

This document outlines the steps to set up the environment for ROS2, Miniconda, NVIDIA drivers, PyTorch, Hik Camera drivers, and CH341 serial drivers.

## ROS2
Follow the tutorial guide on [ROS2 Humble Installation for Ubuntu](https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Development-Setup.html).

After installing Conda, ensure compatibility with `rclpy` (ROS2 Python node) by adding the ROS2 Humble setup script to your Conda environment activation. Add the following line to your Conda environment's activation script:
```bash
source /opt/ros/humble/setup.bash
```

## Miniconda
Install Miniconda with the following commands:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Initialize the Conda environment:
```bash
~/miniconda3/bin/conda init bash  # Replace 'bash' with 'zsh', 'tcsh', 'oh-my-zsh', etc., if needed
```

## NVIDIA Driver
Refer to the [NVIDIA driver installation guide](https://blog.csdn.net/ytusdc/article/details/132403852). Ensure you uninstall the **nouveau** driver, as it is an unofficial NVIDIA driver that may cause conflicts.

## PyTorch and CUDA Toolkit
Visit the [official PyTorch website](https://pytorch.org/) for installation instructions. For most cases, install CUDA 11.8. For NVIDIA 5070, use the latest CUDA version. Run:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Additional dependencies:
- **TensorRT**: Use version > 10.0.
- **ONNX Runtime GPU**: The latest version is compatible.



## Hik Camera
1. Go to the website and download the one called "***机器视觉工业相机客户端MVS (Linux)***", then install the correct version of MVS. 
2. [Download Link (https://www.hikrobotics.com/cn/machinevision/service/download?module=0)](https://www.hikrobotics.com/cn/machinevision/service/download?module=0)
3. After installation, go to path /opt/MVS/bin/ and check whether it is ok to use the SDK:

```
cd /opt/MVS/bin/
./MVS
```

## CH341 Serial Driver
Assuming you are using a CH340 USB-TTL chip, download the driver from the [official CH341 repository](https://github.com/WCHSoftGroup/ch341ser_linux) and follow the installation guide.

After installing the kernel module, verify that `/dev/ttyUSB*` or `/dev/ttyCH341USB*` appears by running:
```bash
ls /dev/tty*
```

**Attention**: The `brltty` service in Ubuntu may occupy the serial port, making it inaccessible. Refer to [this solution](https://blog.csdn.net/qq_27865227/article/details/125538516) for details.