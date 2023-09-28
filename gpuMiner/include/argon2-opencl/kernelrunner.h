#ifndef ARGON2_OPENCL_KERNELRUNNER_H
#define ARGON2_OPENCL_KERNELRUNNER_H

#include "programcontext.h"
#include "argon2-gpu-common/argon2params.h"

#include <memory>

namespace argon2 {
namespace opencl {

class KernelRunner
{
private:
    const ProgramContext *programContext;
    const Argon2Params *params;

    std::size_t batchSize;
    bool bySegment;
    bool precompute;

    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Buffer memoryBuffer, refsBuffer;
    cl::Event start, end, kernelStart, kernelEnd;

    std::size_t memorySize;

    std::unique_ptr<std::uint8_t[]> blocksIn;
    std::unique_ptr<std::uint8_t[]> blocksOut;

    void copyInputBlocks();
    void copyOutputBlocks();

    void precomputeRefs();

public:
    std::uint32_t getMinLanesPerBlock() const
    {
        return bySegment ? 1 : params->getLanes();
    }
    std::uint32_t getMaxLanesPerBlock() const { return params->getLanes(); }

    std::size_t getMinJobsPerBlock() const { return 1; }
    std::size_t getMaxJobsPerBlock() const { return batchSize; }

    std::size_t getBatchSize() const { return batchSize; }

    void *getInputMemory(std::size_t jobId) const
    {
        std::size_t copySize = params->getLanes() * 2 * ARGON2_BLOCK_SIZE;
        return blocksIn.get() + jobId * copySize;
    }
    const void *getOutputMemory(std::size_t jobId) const
    {
        std::size_t copySize = params->getLanes() * ARGON2_BLOCK_SIZE;
        return blocksOut.get() + jobId * copySize;
    }

    KernelRunner(const ProgramContext *programContext,
                 const Argon2Params *params, const Device *device,
                 std::size_t batchSize, bool bySegment, bool precompute);

    void run(std::uint32_t lanesPerBlock, std::size_t jobsPerBlock);
    float finish();
};

} // namespace opencl
} // namespace argon2

#endif // ARGON2_OPENCL_KERNELRUNNER_H
