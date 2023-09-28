#ifndef ARGON2_CUDA_PROCESSINGUNIT_H
#define ARGON2_CUDA_PROCESSINGUNIT_H

#if HAVE_CUDA

#include <memory>

#include "programcontext.h"
#include "kernelrunner.h"
#include "argon2-gpu-common/argon2params.h"
#include <string>
#include <vector>
namespace argon2 {
namespace cuda {

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

} // namespace cuda
} // namespace argon2

#else

#include <cstddef>

#include "programcontext.h"
#include "argon2-gpu-common/argon2params.h"

namespace argon2 {
namespace cuda {

class ProcessingUnit
{
public:
    std::size_t getBatchSize() const { return 0; }
    std::vector<std::string> passwordStorage;

    ProcessingUnit(
            const ProgramContext *programContext, const Argon2Params *params,
            const Device *device, std::size_t batchSize,
            bool bySegment = true, bool precomputeRefs = false)
    {
    }

    void setPassword(std::size_t index, const void *pw, std::size_t pwSize) { }

    void getHash(std::size_t index, void *hash) { }
    std::string getPW(std::size_t index) { return {}; }

    void beginProcessing() { }
    void endProcessing() { }
};

} // namespace cuda
} // namespace argon2

#endif /* HAVE_CUDA */

#endif // ARGON2_CUDA_PROCESSINGUNIT_H
