#ifndef ARGON2_OPENCL_KERNELLOADER_H
#define ARGON2_OPENCL_KERNELLOADER_H

#include "opencl.h"
#include "argon2-gpu-common/argon2-common.h"

#include <string>

namespace argon2 {
namespace opencl {

namespace KernelLoader
{
    cl::Program loadArgon2Program(
            const cl::Context &context,
            const std::string &sourceDirectory,
            Type type, Version version, bool debug = false);
};

} // namespace opencl
} // namespace argon2

#endif // ARGON2_OPENCL_KERNELLOADER_H
