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

## System Requirements

- CUDA Toolkit (if using CUDA)
- OpenCL (if using OpenCL)
- CMake (>= 3.7)
- C++ Compiler with C++11 support

## Building
``` bash
sudo apt install ocl-icd-opencl-dev
```
``` bash
git clone https://github.com/shanhaicoder/XENGPUMiner.git 
cd XENGPUMiner
chmod +x build.sh
./build.sh [-cuda_arch=<YOUR_CUDA_ARCH>]
```
### Specifying CUDA Architecture
To ensure optimal performance, specify the cuda_arch value that corresponds to the Compute Capability of your Nvidia GPU. Refer to the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus#compute) to find the suitable value for your GPU model.

```
./build.sh -cuda_arch=sm_52  # Example for a GPU with Compute Capability 5.2
```
If you do not specify the cuda_arch, the script will use a default value sm_75.

## Usage

### Running the GPU Miner

To run the XEN GPUMiner, use the following command, where `-b 1024` specifies the number of hashes processed in a single batch:

```sh
./xengpuminer -b 1024
```
Note: The -b parameter represents the number of hashes to process in a single batch. While the maximum value for this parameter is dependent on the available GPU memory, it is recommended to choose a moderate value. As long as the total number of hashes is sufficient, a moderate batch size should suffice.

If you are running the miner with OpenCL, add the -m opencl parameter:

```sh
./xengpuminer -b 1024 -m opencl
```

To run the miner in the background, you can use nohup or screen (depending on your system and preferences):
```sh
nohup ./xengpuminer -b 1024 &
```
Or, with screen:
```sh
screen -S miner
./xengpuminer -b 1024
# Press 'Ctrl-A' followed by 'D' to detach the screen session.
```

### Running the GPU Miner

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
python miner.py --gpu=true  # To enable GPU mode
python miner.py --gpu=false  # To disable GPU mode and run in CPU mode
```
### Configuration Reminder
Before starting the miner, don't forget to configure your account address in the `config.conf` file. The default account address is set to `0x24691e54afafe2416a8252097c9ca67557271475`. Please replace it with your own account address to receive mining rewards.

Here’s how the `config.conf` file will look:

```ini
account = 0xYOUR_ACCOUNT_ADDRESS
```
Replace 0xYOUR_ACCOUNT_ADDRESS with your actual account address.
