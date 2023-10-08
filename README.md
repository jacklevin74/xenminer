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

## Quick Start[CUDA version]

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
chmod +x miner.sh
./miner.sh
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

### Preliminary Setup:
Install the necessary Python packages before running the CPU version for the first time:
```sh
pip install -U -r requirements.txt
```
### Streamlined Operation:

For a concise, unified approach to running and managing the miners, a Bash script can be used, allowing specified configurations, such as the number of GPUs, CPU cores, enabling OpenCL, and more. This simplifies the entire process, aiding users in launching and managing the mining operation effectively. Here's a quick running example. More details check `Bash Script Usage and Options`.

```sh
$ chmod +x miner.sh
$ ./miner.sh
```

### Bash Script Usage and Options:
``` sh
$ ./miner.sh -g <num_of_gpus> -c <num_of_cpu_cores> [-d] [-o] [-s] [-l] [-h]
```
Options:
```
  -g, --gpus <num>           Number of GPUs (Default: 1)
  -c, --cpucores <num>       Number of CPU cores; activates CPU mode if >0 (Default: 0)
  -d, --devfee, --dev-fee-on Enable dev fee (Default: off)
  -o, --opencl               Enable OpenCL computation (Default: off)
  -s, --silence              Run in silence/background mode (Default: off)
  -l, --logging-on           Record verified blocks into payload.log file (Default: off)"
  -h, --help                 Display help message and exit
```
### Further Assistance:
If you experience any issues or have any queries, refer to the provided documentation or seek support from the community or support channels.

## Advanced Operations and Customizations
The detailed operations and customizations provided in this section are not mandatory for all users but are intended for those who wish to understand the intricacies and tailor the mining operation to their needs.

### Component Responsibilities:
1. **xengpuminer:**
   - Executes offline, local mining through files.
   - Operates independently of network connectivity.

2. **miner.py:**
   - Outputs the `difficulty.txt` file.
   - Monitors, verifies, and uploads blocks found in the `gpu_found_blocks_tmp` directory.
   - Requires network connectivity to upload verified blocks.

### Using xengpuminer with Advanced Options

`xengpuminer` provides several advanced options to help you configure your mining processes more precisely. Here's a guide on how to utilize these options effectively:

1. **List Devices:**
   ```sh
   $ ./xengpuminer -l -m <DEVICE_TYPE>
   ```
   The `-l` flag lists all available devices. Replace `<DEVICE_TYPE>` with either `cuda` or `opencl` to list the corresponding type of devices available on your system. For instance:
   ```sh
   $ ./xengpuminer -l -m cuda
   ```
   Lists all CUDA-compatible devices available.

2. **Select Mode:**
   ```sh
   $ ./xengpuminer -m <MODE>
   ```
   The `-m` flag allows you to choose the mode in which to run xengpuminer. Replace `<MODE>` with `cuda` for CUDA, `opencl` for OpenCL. For example:
   
   ```sh
   $ ./xengpuminer -m opencl
   ```
   Executes xengpuminer in OpenCL mode.

3. **Specify Device:**
   ```sh
   $ ./xengpuminer -d <INDEX>
   ```
   The `-d` flag lets you use the device with the specified index. Replace `<INDEX>` with the index of the device you want to use. For instance:
   ```sh
   $ ./xengpuminer -d 0
   ```
   Uses the first device on the list of available devices.

4. **Define Batch Size:**
   ```sh
   $ ./xengpuminer -b <N>
   ```
   The `-b` flag denotes the number of tasks per batch. Replace `<N>` with the desired batch size. For example:
   ```sh
   $ ./xengpuminer -b 16
   ```
   Processes 16 tasks per batch, which can be optimal depending on your device's capabilities and your specific needs.

5. **Display Help:**
   ```sh
   $ ./xengpuminer -?
   ```
   Shows help for xengpuminer's options and exits, providing quick reference to the options available for configuration.

### Advanced Options for miner.py:

You can configure `miner.py` using command-line arguments or the `config.conf` file. Command-line arguments will override the corresponding settings in `config.conf`.

Command-Line Arguments:
`--account`: Specify the account address for receiving mining rewards.
`--dev-fee-on`: Enable the developer fee to support the project.
`--gpu`: Set to `true` to enable GPU mode, `false` to run in CPU mode.

Configuring Account in `config.conf`:
Replace the default account address in `config.conf` with your own to receive mining rewards:
```ini
account = YOUR_ACCOUNT_ADDRESS  # Replace with your actual account address
```
