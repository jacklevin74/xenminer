#!/bin/bash

# Default values
COMPUTE_TYPE="CUDA"

# Read parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -compute_type) COMPUTE_TYPE="$2"; shift ;;
        -cuda_arch) CUDA_ARCH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# Create and navigate to the build directory
mkdir -p build
cd build || exit

# Check if CUDA_ARCH is set, if not, perform a different action
if [ -z "$CUDA_ARCH" ]; then
    # Run CMake and pass the arguments
    cmake ..

    make

    # Output a message indicating the script has completed successfully
    echo "Build completed successfully"
else
    cmake -DCUDA_ARCH=$CUDA_ARCH ..
    # Compile the project
    make

    # Output a message indicating the script has completed successfully
    echo "Build completed successfully with CUDA_ARCH=${CUDA_ARCH}"

fi
