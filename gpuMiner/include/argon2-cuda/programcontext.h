#ifndef ARGON2_CUDA_PROGRAMCONTEXT_H
#define ARGON2_CUDA_PROGRAMCONTEXT_H

#include "globalcontext.h"
#include "argon2-gpu-common/argon2-common.h"

namespace argon2 {
namespace cuda {

#if HAVE_CUDA

class ProgramContext
{
private:
    const GlobalContext *globalContext;

    Type type;
    Version version;

public:
    const GlobalContext *getGlobalContext() const { return globalContext; }

    Type getArgon2Type() const { return type; }
    Version getArgon2Version() const { return version; }

    ProgramContext(
            const GlobalContext *globalContext,
            const std::vector<Device> &devices,
            Type type, Version version);
};

#else

class ProgramContext
{
private:
    const GlobalContext *globalContext;

    Type type;
    Version version;

public:
    const GlobalContext *getGlobalContext() const { return globalContext; }

    Type getArgon2Type() const { return type; }
    Version getArgon2Version() const { return version; }

    ProgramContext(
            const GlobalContext *globalContext,
            const std::vector<Device> &devices,
            Type type, Version version)
        : globalContext(globalContext), type(type), version(version)
    {
    }
};

#endif /* HAVE_CUDA */

} // namespace cuda
} // namespace argon2

#endif // ARGON2_CUDA_PROGRAMCONTEXT_H
