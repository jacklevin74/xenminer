#ifndef ARGON2_OPENCL_PROGRAMCONTEXT_H
#define ARGON2_OPENCL_PROGRAMCONTEXT_H

#include "globalcontext.h"
#include "argon2-gpu-common/argon2-common.h"

namespace argon2 {
namespace opencl {

class ProgramContext
{
private:
    const GlobalContext *globalContext;

    std::vector<cl::Device> devices;
    cl::Context context;
    cl::Program program;

    Type type;
    Version version;

public:
    const GlobalContext *getGlobalContext() const { return globalContext; }

    const std::vector<cl::Device> &getDevices() const { return devices; }
    const cl::Context &getContext() const { return context; }
    const cl::Program &getProgram() const { return program; }

    Type getArgon2Type() const { return type; }
    Version getArgon2Version() const { return version; }

    ProgramContext(
            const GlobalContext *globalContext,
            const std::vector<Device> &devices,
            Type type, Version version);
};

} // namespace opencl
} // namespace argon2

#endif // ARGON2_OPENCL_PROGRAMCONTEXT_H
