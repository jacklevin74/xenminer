#include "cpuexecutive.h"

#include "argon2.h"

#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <iostream>

static constexpr std::size_t HASH_LENGTH = 32;
static constexpr std::size_t SALT_LENGTH = 32;

class ParallelRunner
{
private:
    const BenchmarkDirector &director;
    PasswordGenerator &pwGen;

    std::unique_ptr<std::uint8_t[]> salt;
    std::size_t nworkers, nthreads;
    std::vector<std::future<void>> futures;
    std::size_t jobsNotStarted;
    std::mutex pwGenMutex;

    void runWorker() {
        auto out = std::unique_ptr<std::uint8_t[]>(
                    new std::uint8_t[HASH_LENGTH]);

#ifdef ARGON2_PREALLOCATED_MEMORY
        std::size_t memorySize = argon2_memory_size(director.getMemoryCost(),
                                                    director.getLanes());
        auto memory = std::unique_ptr<std::uint8_t[]>(
                    new std::uint8_t[memorySize]);
#endif
        for (;;) {
            const void *pw;
            std::size_t pwSize;
            {
                std::lock_guard<std::mutex> guard(pwGenMutex);
                if (jobsNotStarted == 0)
                    break;

                pwGen.nextPassword(pw, pwSize);
                jobsNotStarted--;
            }

            argon2_context ctx;
            ctx.out = out.get();
            ctx.outlen = HASH_LENGTH;
            ctx.pwd = static_cast<std::uint8_t *>(const_cast<void *>(pw));
            ctx.pwdlen = pwSize;

            ctx.salt = salt.get();
            ctx.saltlen = SALT_LENGTH;
            ctx.secret = NULL;
            ctx.secretlen = 0;
            ctx.ad = NULL;
            ctx.adlen = 0;

            ctx.t_cost = director.getTimeCost();
            ctx.m_cost = director.getMemoryCost();
            ctx.lanes = director.getLanes();
            ctx.threads = nthreads;

            ctx.version = director.getVersion();

            ctx.allocate_cbk = NULL;
            ctx.free_cbk = NULL;
            ctx.flags = 0;

            argon2_type type;
            switch(director.getType()) {
            case argon2::ARGON2_I:
                type = Argon2_i;
                break;
            case argon2::ARGON2_D:
                type = Argon2_d;
                break;
            case argon2::ARGON2_ID:
                type = Argon2_id;
                break;
            default:
                throw std::runtime_error(
                            argon2_error_message(ARGON2_INCORRECT_TYPE));
            }

#ifdef ARGON2_PREALLOCATED_MEMORY
            int err = argon2_ctx_mem(&ctx, type, memory.get(), memorySize);
#else
            int err = argon2_ctx(&ctx, type);
#endif
            if (err) {
                throw std::runtime_error(argon2_error_message(err));
            }
        }
    }

public:
    ParallelRunner(const BenchmarkDirector &director, PasswordGenerator &pwGen)
        : director(director), pwGen(pwGen), salt(new std::uint8_t[SALT_LENGTH]),
          jobsNotStarted(director.getBatchSize())
    {
        std::size_t parallelism = std::thread::hardware_concurrency();
        if (parallelism > director.getLanes()) {
            nworkers = parallelism / director.getLanes();
            nthreads = director.getLanes();
        } else {
            nworkers = 1;
            nthreads = parallelism;
        }

        futures.reserve(nworkers);

        for (std::size_t i = 0; i < nworkers; i++) {
            futures.push_back(std::async(std::launch::async,
                                         &ParallelRunner::runWorker, this));
        }
    }

    void wait()
    {
        for (auto &fut : futures) {
            fut.wait();
        }
        for (auto &fut : futures) {
            fut.get();
        }
    }
};

class CpuRunner : public Argon2Runner
{
public:
    nanosecs runBenchmark(const BenchmarkDirector &director,
                          PasswordGenerator &pwGen) override;
};

nanosecs CpuRunner::runBenchmark(const BenchmarkDirector &director,
                                 PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type;

    auto beVerbose = director.isVerbose();
    if (beVerbose) {
        std::cout << "Starting computation..." << std::endl;
    }

    FLAG_clear_internal_memory = 0;

    clock_type::time_point start = clock_type::now();

    ParallelRunner runner(director, pwGen);
    runner.wait();

    clock_type::time_point end = clock_type::now();
    clock_type::duration compTime = end - start;
    auto compTimeNs = toNanoseconds(compTime);
    if (beVerbose) {
        std::cout << "    Computation took "
                  << RunTimeStats::repr(compTimeNs) << std::endl;
    }
    return compTimeNs;
}

int CpuExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    if (listDevices) {
        std::cout << "Device #0: CPU" << std::endl;
        return 0;
    }
    if (deviceIndex != 0) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl;
        return 1;
    }

#ifdef ARGON2_SELECTABLE_IMPL
    argon2_select_impl(director.isVerbose() ? stderr : nullptr, "[libargon2] ");
#endif

    CpuRunner runner;
    return director.runBenchmark(runner);
}
