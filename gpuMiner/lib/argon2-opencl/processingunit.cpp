#include "processingunit.h"

#include <limits>
#ifndef NDEBUG
#include <iostream>
#endif

namespace argon2 {
namespace opencl {

static bool isPowerOfTwo(std::size_t x)
{
    return (x & (x - 1)) == 0;
}

ProcessingUnit::ProcessingUnit(
        const ProgramContext *programContext, const Argon2Params *params,
        const Device *device, std::size_t batchSize,
        bool bySegment, bool precomputeRefs)
    : programContext(programContext), params(params), device(device),
      runner(programContext, params, device, batchSize, bySegment,
             precomputeRefs),
      bestLanesPerBlock(runner.getMinLanesPerBlock()),
      bestJobsPerBlock(runner.getMinJobsPerBlock())
{
    /* pre-fill first blocks with pseudo-random data: */
    for (std::size_t i = 0; i < batchSize; i++) {
        setPassword(i, NULL, 0);
    }

    if (runner.getMaxLanesPerBlock() > runner.getMinLanesPerBlock()
            && isPowerOfTwo(runner.getMaxLanesPerBlock())) {
#ifndef NDEBUG
        std::cerr << "[INFO] Tuning lanes per block..." << std::endl;
#endif

        float bestTime = std::numeric_limits<float>::infinity();
        for (std::uint32_t lpb = 1; lpb <= runner.getMaxLanesPerBlock();
             lpb *= 2)
        {
            float time;
            try {
                runner.run(lpb, bestJobsPerBlock);
                time = runner.finish();
            } catch(cl::Error &ex) {
#ifndef NDEBUG
                std::cerr << "[WARN]   OpenCL error on " << lpb
                          << " lanes per block: " << ex.what() << std::endl;
#endif
                break;
            }

#ifndef NDEBUG
            std::cerr << "[INFO]   " << lpb << " lanes per block: "
                      << time << " ms" << std::endl;
#endif

            if (time < bestTime) {
                bestTime = time;
                bestLanesPerBlock = lpb;
            }
        }
#ifndef NDEBUG
        std::cerr << "[INFO] Picked " << bestLanesPerBlock
                  << " lanes per block." << std::endl;
#endif
    }

    /* Only tune jobs per block if we hit maximum lanes per block: */
    if (bestLanesPerBlock == runner.getMaxLanesPerBlock()
            && runner.getMaxJobsPerBlock() > runner.getMinJobsPerBlock()
            && isPowerOfTwo(runner.getMaxJobsPerBlock())) {
#ifndef NDEBUG
        std::cerr << "[INFO] Tuning jobs per block..." << std::endl;
#endif

        float bestTime = std::numeric_limits<float>::infinity();
        for (std::size_t jpb = 1; jpb <= runner.getMaxJobsPerBlock();
             jpb *= 2)
        {
            float time;
            try {
                runner.run(bestLanesPerBlock, jpb);
                time = runner.finish();
            } catch(cl::Error &ex) {
#ifndef NDEBUG
                std::cerr << "[WARN]   OpenCL error on " << jpb
                          << " jobs per block: " << ex.what() << std::endl;
#endif
                break;
            }

#ifndef NDEBUG
            std::cerr << "[INFO]   " << jpb << " jobs per block: "
                      << time << " ms" << std::endl;
#endif

            if (time < bestTime) {
                bestTime = time;
                bestJobsPerBlock = jpb;
            }
        }
#ifndef NDEBUG
        std::cerr << "[INFO] Picked " << bestJobsPerBlock
                  << " jobs per block." << std::endl;
#endif
    }
}

void ProcessingUnit::setPassword(std::size_t index, const void *pw,
                                 std::size_t pwSize)
{
    void *memory = runner.getInputMemory(index);
    params->fillFirstBlocks(memory, pw, pwSize,
                            programContext->getArgon2Type(),
                            programContext->getArgon2Version());
                                // Expand the storage if needed
    if (passwordStorage.size() <= index) {
        passwordStorage.resize(index + 1);
    }
    
    // Store the password at the specified index
    passwordStorage[index] = std::string(static_cast<const char*>(pw), pwSize);
}

void ProcessingUnit::getHash(std::size_t index, void *hash)
{
    const void *memory = runner.getOutputMemory(index);
    params->finalize(hash, memory);
}
std::string ProcessingUnit::getPW(std::size_t index){
    if (index < passwordStorage.size()) {
        return passwordStorage[index];
    }
    return {};  // Return an empty string if the index is out of bounds

}

void ProcessingUnit::beginProcessing()
{
    runner.run(bestLanesPerBlock, bestJobsPerBlock);
}

void ProcessingUnit::endProcessing()
{
    runner.finish();
}

} // namespace opencl
} // namespace argon2
