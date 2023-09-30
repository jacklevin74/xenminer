#ifndef ARGON2_OPENCL_PROCESSINGUNIT_H
#define ARGON2_OPENCL_PROCESSINGUNIT_H

#include <memory>

#include "kernelrunner.h"

namespace argon2 {
namespace opencl {

class ProcessingUnit
{
private:
    const ProgramContext *programContext;
    const Argon2Params *params;
    const Device *device;
    std::vector<std::string> passwordStorage;

    KernelRunner runner;
    std::uint32_t bestLanesPerBlock;
    std::size_t bestJobsPerBlock;

public:
    std::size_t getBatchSize() const { return runner.getBatchSize(); }

    ProcessingUnit(
            const ProgramContext *programContext, const Argon2Params *params,
            const Device *device, std::size_t batchSize,
            bool bySegment = true, bool precomputeRefs = false);

    /* You can safely call this function after the beginProcessing() call to
     * prepare the next batch: */
    void setPassword(std::size_t index, const void *pw, std::size_t pwSize);
    /* You can safely call this function after the beginProcessing() call to
     * process the previous batch: */
    void getHash(std::size_t index, void *hash);
    std::string getPW(std::size_t index);

    void beginProcessing();
    void endProcessing();
};

} // namespace opencl
} // namespace argon2

#endif // ARGON2_OPENCL_PROCESSINGUNIT_H
