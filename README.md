# XENBlocks GPU Miner

XENBlocks GPU Miner is a high-performance GPU mining software developed for XENBlocks Blockchain. 
It supports both CUDA and OpenCL, enabling it to run on various GPU architectures.

## Origin & Enhancements

This project is a fork of the original [XENMiner](https://github.com/jacklevin74/xenminer), a CPU miner developed by jacklevin74. We are thankful to jacklevin74 and all contributors to the original project for laying the groundwork.

### Enhancements:
- **GPU Mining Support**: This version provides support for GPU mining using either CUDA or OpenCL, enabling efficient mining on various GPU architectures.
- **Dual Mining Mode**: Users can easily switch between CPU and GPU mining based on their preferences and hardware capabilities, allowing for flexible deployment.

## Features

- Supports CUDA and OpenCL
- Efficient hashing computation
- Compatible with various GPU architectures
- User-friendly Command-Line Interface (CLI)
- Easy to configure and use

## Features Yet to be Implemented

The following are features and enhancements that are planned to be incorporated in future versions of this project:

- **AMD and OpenCL Support**: We are actively working to fully support AMD GPUs and OpenCL.
- **Windows Compilation Guide**: A detailed guide for compiling and installing on Windows is planned to assist Windows users in utilizing this tool more conveniently.
- **Precompiled Versions**: We are planning to release precompiled versions to facilitate easier installations.

## System Requirements

- CUDA Toolkit (if using CUDA)
- OpenCL (if using OpenCL)
- CMake (>= 3.7)
- C++ Compiler with C++11 support

## Building

### Preliminary Setup for Beginners

If you are unfamiliar with the building process or are encountering difficulties, you can run the following commands to install the necessary dependencies. If you are familiar with this process, feel free to adapt according to your preferences.

```sh
apt update && apt upgrade -y
apt install git cmake make sudo -y
```
Remember, these commands are for Debian-based systems like Ubuntu. Please adapt them according to your operating system if needed.

Note: The sudo command is generally installed by default on most Linux distributions, but it is included in the command just in case it is not present.

### Building from Source

To build from the source, follow the steps below. Please make sure you have `git`, `cmake`, and `make` installed on your system. If not, you can install them using your system's package manager.

Now, proceed with the following steps to clone the repository and build the project:
``` bash
git clone https://github.com/shanhaicoder/XENGPUMiner.git 
cd XENGPUMiner
chmod +x build.sh
```


### OpenCL Installation
To enable OpenCL support, you need to install the OpenCL development package. On Debian-based systems like Ubuntu, you can run the following command:

```sh
sudo apt install ocl-icd-opencl-dev
```
Please adapt the command according to your operating system if it's not a Debian-based system.

### CUDA Installation
For CUDA support, you need to have both the NVIDIA driver and the CUDA Toolkit installed. You can refer to the official [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) to download and install the CUDA Toolkit suitable for your system.

Before installing the CUDA Toolkit, run the `nvidia-smi` command to check the existing driver installation and verify the compatibility of your driver version with the CUDA Toolkit version you plan to install. If you need to update your NVIDIA driver, please refer to the official [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx) page.

Here's a general guideline for installing CUDA:

1. Check your NVIDIA driver installation and version with nvidia-smi.
2. Refer to the official CUDA Toolkit Archive and download a compatible version.
3. Follow the instructions provided in the download page to install the CUDA Toolkit.

### Execute build.sh
Before running the build script, you can specify the CUDA architecture you are targeting by using the -cuda_arch flag followed by your CUDA architecture version. For example, if you are targeting CUDA architecture 6.1, you would run:

``` bash
./build.sh -cuda_arch sm_61
```

When building the project, it is crucial to specify the correct `cuda_arch` value to optimize the performance of your application. This value should correspond to the Compute Capability of your Nvidia GPU.

You can refer to [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus#compute) to find the correct Compute Capability value for your GPU model. Once you have identified the right value, specify it with the `-cuda_arch` flag when running the build script, as shown below:

```sh
./build.sh -cuda_arch sm_86  # Example: For a GPU with Compute Capability 8.6
```
If you omit the `-cuda_arch` flag, the build script will default to using `sm_75`. While the script may still run without specifying `cuda_arch`, defining it correctly ensures that you are utilizing your GPU to its full potential.

## Usage

### Running the Miner.py

Before running the Python miner, if you have not run the CPU version before, make sure to install the necessary Python packages by running the following command:
```sh
pip install -U -r requirements.txt
```

To run the Python miner, use:

```sh
python miner.py
```

Quick Tips

To reattach to a screen session, use screen -r miner.
Check your system documentation for more advanced usage of nohup and screen.

### Additional Configuration Options
You can also specify whether to enable GPU mode by adding the --gpu parameter when running the Python miner:

```sh
python miner.py --gpu=true  # To enable GPU mode (default)
python miner.py --gpu=false  # To disable GPU mode and run in CPU mode
```
### Configuration Reminder
Before starting the miner, don't forget to configure your account address in the `config.conf` file. The default account address is set to `0x24691e54afafe2416a8252097c9ca67557271475`. Please replace it with your own account address to receive mining rewards.

Here’s how the `config.conf` file will look:

```ini
account = YOUR_ACCOUNT_ADDRESS
```
Replace `YOUR_ACCOUNT_ADDRESS` with your actual account address.

### Running the Real GPU Miner

To run the XEN GPUMiner, use the following command, where `-b 128` specifies the number of hashes processed in a single batch:

```sh
./xengpuminer -b 128
```
Note: The -b parameter represents the number of hashes to process in a single batch. While the maximum value for this parameter is dependent on the available GPU memory, it is recommended to choose a moderate value. As long as the total number of hashes is sufficient, a moderate batch size should suffice.

Normally, we need about `difficulty * [-b para] * 1024 Bytes` gpu memory

If you are running the miner with OpenCL, add the -m opencl parameter:

```sh
./xengpuminer -b 128 -m opencl
```

To run the miner in the background, you can use nohup or screen (depending on your system and preferences):
```sh
nohup ./xengpuminer -b 128 &
```
Or, with screen:
```sh
screen -S miner
./xengpuminer -b 128
# Press 'Ctrl-A' followed by 'D' to detach the screen session.
```
