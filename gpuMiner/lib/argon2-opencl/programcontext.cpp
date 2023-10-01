#include "programcontext.h"

#include "kernelloader.h"

namespace argon2 {
namespace opencl {

ProgramContext::ProgramContext(
        const GlobalContext *globalContext,
        const std::vector<Device> &devices,
        Type type, Version version)
    : globalContext(globalContext), devices(), type(type), version(version)
{
    this->devices.reserve(devices.size());
    for (auto &device : devices) {
        this->devices.push_back(device.getCLDevice());
    }
    context = cl::Context(this->devices);

    program = KernelLoader::loadArgon2Program(
                // FIXME path:
                context, "./gpuMiner/data/kernels", type, version);
}

} // namespace opencl
} // namespace argon2

