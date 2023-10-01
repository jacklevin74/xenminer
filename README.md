# XENBlocks GPU Miner

XENBlocks GPU Miner is a high-performance GPU mining software developed for XENBlocks Blockchain. 
It supports both CUDA and OpenCL, enabling it to run on various GPU architectures.

## Origin & Enhancements

This project is a fork of the original [XENMiner](https://github.com/jacklevin74/xenminer), a CPU miner developed by jacklevin74. We are thankful to jacklevin74 and all contributors to the original project for laying the groundwork.

### Enhancements:
- **GPU Mining Support**: This version provides support for GPU mining using either CUDA or OpenCL, enabling efficient mining on various GPU architectures.
- **Dual Mining Mode**: Users can easily switch between CPU and GPU mining based on their preferences and hardware capabilities, allowing for flexible deployment.

## Important Warning

This project involves certain build processes and, as of now, no compiled versions have been released. If you wish to use it, please be prepared mentally and be willing to explore and solve problems that may arise during the installation and usage.

We encourage users to be proactive in improving the project:
- If you encounter any issues or have solutions to existing ones, don’t hesitate to contribute to the documentation, making it more accurate and comprehensive.
- If you manage to install and compile the entire project on a specific system, consider contributing a detailed guide outlining the full process to aid others encountering similar environments or issues.

Your contributions will help in making the documentation and the project more robust and user-friendly, benefiting the entire user community and future adopters of this project. 

Thank you for your understanding and cooperation!

## Developer Fee and Continuous Improvement
We highly value community support and user satisfaction, and we strive to minimize any inconvenience for our users. To sustain the ongoing development, maintenance, and enhancements of the project, the miner operates with a nominal 1.67% developer fee. This fee is equivalent to the mining rewards earned in the first minute of every hour and is crucial for the continuous improvement of the project.

### Commitment to Excellence
The developer fee serves as a reinvestment into the project, allowing us to enhance the efficiency of the miner, implement innovative features, and release regular updates to address any potential bugs or issues. We are devoted to delivering an exceptional mining experience and relentlessly work on optimizing the software to ensure you reap maximum mining rewards.

### Option to Contribute and Adapt
Understanding the importance of user choice and transparency, we provide the flexibility to adjust or disable the developer fee. If you decide to contribute a different amount or opt out of the developer fee, you can easily do so when running the miner:

```sh
$ python miner.py --dev-fee-on --dev-fee-seconds [Your Chosen Seconds]
```
By using the `--dev-fee-on` option, you can enable the developer fee, and with `--dev-fee-seconds`, you can specify the number of seconds per hour you wish to contribute, This may not be necessary, the default is 60. If you choose not to use these options, all mining rewards will be directed to your account, with no contributions to the developer. We respect and appreciate your decisions and are always here to assist you with any concerns or questions you may have.

### Community Engagement
Our gratitude extends to the entire community for its unwavering support and active engagement. We encourage and welcome feedback, suggestions, and vibrant discussions as we believe in mutual growth and the shared vision of shaping a prosperous mining future together.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](License) file for details.

## Features

- Supports CUDA and OpenCL
- Efficient hashing computation
- Compatible with various GPU architectures
- Supports multiple GPUs
- User-friendly Command-Line Interface (CLI)
- Easy to configure and use

### Features Yet to be Implemented

The following are features and enhancements that are planned to be incorporated in future versions of this project:

- **Windows Compilation Guide**: A detailed guide for compiling and installing on Windows is planned to assist Windows users in utilizing this tool more conveniently.
- **Precompiled Versions**: We are planning to release precompiled versions to facilitate easier installations.

## System Requirements

- CUDA Toolkit (if using CUDA)
- OpenCL (if using OpenCL)
- CMake (>= 3.7)
- C++ Compiler with C++11 support

## Quick Start

If you're familiar with using a terminal, the following commands can help you get started quickly. 

```sh
apt update && apt upgrade -y  # Update system packages
apt install git cmake make sudo -y  # Install necessary packages for building
git clone https://github.com/shanhaicoder/XENGPUMiner.git  # Clone the repository
cd XENGPUMiner  # Navigate to the project directory
chmod +x build.sh  # Make the build script executable
sudo apt install ocl-icd-opencl-dev  # Install OpenCL development package
./build.sh  # Run the build script
pip install -U -r requirements.txt  # Install the required Python packages
screen -S "gpuminer" -dm bash -c "python miner.py --gpu=true"  # Start the Python miner in a new screen session
screen -S "gpuminer" -X screen bash -c "./xengpuminer -b 128"  # Start the GPU miner in the same screen session
```
Please note that this Quick Start assumes you are on a Debian-based system (like Ubuntu) and have some knowledge of Linux command line, and it is only intended to serve as a basic guide to get the software running quickly. For more detailed information on building and configuring the miner, refer to the relevant sections of this document.

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

You can refer to [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus#compute) to find the correct Compute Capability value for your GPU model. Or this is beter: [matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/), make sure you have a higher version CUDA toolkit installed. Once you have identified the right value, specify it with the `-cuda_arch` flag when running the build script, as shown below:

```sh
./build.sh -cuda_arch sm_86  # Example: For a GPU with Compute Capability 8.6
```
If you omit the `-cuda_arch` flag, the build script will default to using `sm_75`. While the script may still run without specifying `cuda_arch`, defining it correctly ensures that you are utilizing your GPU to its full potential.

## Usage

It is crucial to understand that `miner.py` and `xengpuminer` must operate concurrently within the same directory. `xengpuminer` executes the local, offline mining processes through files, while `miner.py` is responsible for outputting `difficulty.txt` and continuously monitoring and verifying the blocks output to the `gpu_found_blocks_tmp` directory before uploading them. Their interdependence means they need to be run in tandem in the same directory for the entire mining operation to function correctly.

### Specific Responsibilities of Each Component:

1. **xengpuminer:**
   - Performs offline, local mining.
   - Processes data through files.
   - Does not require network connectivity.

2. **miner.py:**
   - Outputs the `difficulty.txt` file.
   - Continuously scans, verifies, and uploads the blocks found in the `gpu_found_blocks_tmp` directory.
   - Requires network connectivity to upload verified blocks.

### Running the Components

Before running the Python miner, if you have not run the CPU version before, make sure to install the necessary Python packages by running the following command:
```sh
pip install -U -r requirements.txt
```

Ensure that both `miner.py` and `xengpuminer` are launched within the same directory. Their cooperative operation is pivotal for seamless mining and uploading of blocks.

```sh
# Running miner.py
$ python miner.py --gpu=true

# Running xengpuminer in a separate session or terminal, but in the same directory
$ ./xengpuminer
```

### Additional Configuration Options
You can also specify whether to enable GPU mode by adding the --gpu parameter when running the Python miner:

```sh
$ python miner.py --gpu=true  # To enable GPU mode (default)
$ python miner.py --gpu=false  # To disable GPU mode and run in CPU mode
```
Note: The -b parameter represents the number of hashes to process in a single batch. While the maximum value for this parameter is dependent on the available GPU memory, a moderate value is recommended. As long as the total number of hashes is sufficient, a moderate batch size should suffice.

Typically, about difficulty * [-b para] * 1024 Bytes of GPU memory is needed.

If you are running the miner with OpenCL, add the -m opencl parameter:

```sh
$ ./xengpuminer -m opencl
```
The -b parameter is not mandatory. If omitted, the system will automatically adjust the batch size for optimum resource utilization.
Note: If opencl is used, two-thirds of the total gpu memory will be used for computing

## Listing Available Devices

To view a list of all available devices that XENGPUMiner can utilize, you can use the `-l` argument, combined with either `-m opencl` or `-m cuda` to specify the type of devices you want to list:

```bash
./xengpuminer -l -m opencl
```
or
```bash
./xengpuminer -l -m cuda
```

This command will display a list of all the GPUs available on your system, along with their indices, which you can use with the `-d <GPU_INDEX>` argument to select a specific GPU for mining.


### Running with Specific GPU

XENGPUMiner supports multiple GPUs, and you can specify which GPU to use by providing the `-d` argument followed by the index of the GPU.

```bash
./xengpuminer -d <GPU_INDEX>
```
Where <GPU_INDEX> is the index of the GPU you want to use. For example, to use the first GPU, you can run:
```bash
./xengpuminer -d 0
```
To use the second GPU, you can run:
```bash
./xengpuminer -d 1
```
Running with Multiple GPUs
To run XENGPUMiner with multiple GPUs, you can start separate instances of the program, each specifying a different GPU index with the `-d` argument. Ensure each instance is run in a separate session or terminal window.

#### Quick Tips

To reattach to a screen session, use `screen -r miner`.
Check your system documentation for more advanced usage of `nohup` and `screen`.

To run the miner in the background, you can use nohup or screen (depending on your system and preferences):
```sh
nohup ./xengpuminer &
```
Or, with screen **recommanded**:
```sh
screen -S miner
./xengpuminer
# Press 'Ctrl-A' followed by 'D' to detach the screen session.
```
### Configuration Reminder
Before starting the miner, don't forget to configure your account address in the `config.conf` file. The default account address is set to `0x24691e54afafe2416a8252097c9ca67557271475`. Please replace it with your own account address to receive mining rewards.

Here’s how the `config.conf` file will look:

```ini
account = YOUR_ACCOUNT_ADDRESS
```
Replace `YOUR_ACCOUNT_ADDRESS` with your actual account address.
