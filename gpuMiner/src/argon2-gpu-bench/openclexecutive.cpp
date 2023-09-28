#include "openclexecutive.h"

#include "argon2-opencl/processingunit.h"

#include <iostream>

static constexpr std::size_t HASH_LENGTH = 32;

class OpenCLRunner : public Argon2Runner
{
private:
    argon2::Argon2Params params;
    argon2::opencl::ProcessingUnit unit;

public:
    OpenCLRunner(const BenchmarkDirector &director,
                 const argon2::opencl::Device &device,
                 const argon2::opencl::ProgramContext &pc)
        : params(HASH_LENGTH, NULL, 0, NULL, 0, NULL, 0,
                 director.getTimeCost(), director.getMemoryCost(),
                 director.getLanes()),
          unit(&pc, &params, &device, director.getBatchSize(),
               director.isBySegment(), director.isPrecomputeRefs())
    {
    }

    nanosecs runBenchmark(const BenchmarkDirector &director,
                          PasswordGenerator &pwGen) override;
};

nanosecs OpenCLRunner::runBenchmark(const BenchmarkDirector &director,
                                    PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type;
    using namespace argon2;
    using namespace argon2::opencl;

    auto beVerbose = director.isVerbose();
    auto batchSize = unit.getBatchSize();
    if (beVerbose) {
        std::cout << "Starting computation..." << std::endl;
    }

    clock_type::time_point checkpt0 = clock_type::now();
    for (std::size_t i = 0; i < batchSize; i++) {
        const void *pw;
        std::size_t pwLength;
        pwGen.nextPassword(pw, pwLength);

        unit.setPassword(i, pw, pwLength);
    }
    clock_type::time_point checkpt1 = clock_type::now();

    unit.beginProcessing();
    unit.endProcessing();

    clock_type::time_point checkpt2 = clock_type::now();
    for (std::size_t i = 0; i < batchSize; i++) {
        uint8_t buffer[HASH_LENGTH];
        unit.getHash(i, buffer);
    }
    clock_type::time_point checkpt3 = clock_type::now();

    if (beVerbose) {
        clock_type::duration wrTime = checkpt1 - checkpt0;
        auto wrTimeNs = toNanoseconds(wrTime);
        std::cout << "    Writing took     "
                  << RunTimeStats::repr(wrTimeNs) << std::endl;
    }

    clock_type::duration compTime = checkpt2 - checkpt1;
    auto compTimeNs = toNanoseconds(compTime);
    if (beVerbose) {
        std::cout << "    Computation took "
                  << RunTimeStats::repr(compTimeNs) << std::endl;
    }

    if (beVerbose) {
        clock_type::duration rdTime = checkpt3 - checkpt2;
        auto rdTimeNs = toNanoseconds(rdTime);
        std::cout << "    Reading took     "
                  << RunTimeStats::repr(rdTimeNs) << std::endl;
    }
    return compTimeNs;
}

int OpenCLExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    using namespace argon2::opencl;

    GlobalContext global;
    auto &devices = global.getAllDevices();

    if (listDevices) {
        std::size_t i = 0;
        for (auto &device : devices) {
            std::cout << "Device #" << i << ": "
                      << device.getInfo() << std::endl;
            i++;
        }
        return 0;
    }
    if (deviceIndex > devices.size()) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl;
        return 1;
    }
    auto &device = devices[deviceIndex];
    if (director.isVerbose()) {
        std::cout << "Using device #" << deviceIndex << ": "
                  << device.getInfo() << std::endl;
    }
    ProgramContext pc(&global, { device },
                      director.getType(), director.getVersion());
    OpenCLRunner runner(director, device, pc);
    return director.runBenchmark(runner);
}
