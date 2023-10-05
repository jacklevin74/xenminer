#include "cpuexecutive.h"

#include "argon2.h"

#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <iostream>
#include <iomanip>
#include <regex>
#include <chrono>
#include <ctime>
#include <fstream>
static constexpr std::size_t HASH_LENGTH = 64;
static constexpr std::size_t SALT_LENGTH = 14;
#include <sstream>
#include <sys/stat.h>
#include <cstring>
#include <ctime>
#define _CRT_SECURE_NO_WARNINGS
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif
bool is_within_five_minutes_of_hour2() {
    auto now = std::chrono::system_clock::now();
    std::time_t time_now = std::chrono::system_clock::to_time_t(now);
    tm *timeinfo = std::localtime(&time_now);
    int minutes = timeinfo->tm_min;
    return 0 <= minutes && minutes < 5 || 55 <= minutes && minutes < 60;
}
static const std::string base64_chars2 = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode2(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars2[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars2[char_array_4[j]];
    }

    return ret;
}
static int file_counter = 0; 
bool create_directory2(const std::string& path) {
    size_t pos = 0;
    do {
        pos = path.find_first_of('/', pos + 1);
        std::string subdir = path.substr(0, pos);
        if (mkdir(subdir.c_str(), 0755) && errno != EEXIST) {
            std::cerr << "Error creating directory " << subdir << ": " << strerror(errno) << std::endl;
            return false;
        }
    } while (pos != std::string::npos);
    return true;
}
static void saveToFile2(const std::string& pw) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time);

    std::ostringstream dirStream;
    dirStream << "gpu_found_blocks_tmp/";
    std::string dirStr = dirStream.str();

    if (!create_directory2(dirStr)) {
        return;
    }

    std::ostringstream filename;
    filename << dirStr << "/" << std::put_time(&now_tm, "%m-%d_%H-%M-%S") << "_" << file_counter++ << ".txt";
    std::ofstream outFile(filename.str(), std::ios::app);
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl;
        return;
    }
    outFile << pw;
    outFile.close();
}

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

            {
                std::lock_guard<std::mutex> guard(pwGenMutex);
                if (jobsNotStarted == 0)
                    break;

                jobsNotStarted--;

            }
            const void *pw;
            std::size_t pwSize;

            //std::string input = "377a8864b41d15652f304159c7aa00510fcca4bd81ccf07d2ef5fdaebca6ce6e9c35685e183daa0f2d54bbefbf707ebc0ae25c2ff3dcc7c140b08d678082f37e";
            //pwSize = 128;
            //pw = input.c_str();
            pwGen.nextPassword(pw, pwSize);

            argon2_context ctx;
            ctx.out = out.get();
            ctx.outlen = HASH_LENGTH;
            ctx.pwd = static_cast<std::uint8_t *>(const_cast<void *>(pw));
            ctx.pwdlen = pwSize;

            const char* saltText = "XEN10082022XEN";
            ctx.salt = reinterpret_cast<uint8_t*>(const_cast<char*>(saltText));
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

#ifdef ARGON2_PREALLOCATED_MEMORY
            int err = argon2_ctx_mem(&ctx, Argon2_id, memory.get(), memorySize);
#else
            int err = argon2_ctx(&ctx, Argon2_id);
#endif
            if (err) {
                throw std::runtime_error(argon2_error_message(err));
            }
            std::regex pattern(R"(XUNI\d)");

            std::string decodedString = base64_encode2(out.get(), HASH_LENGTH);
            std::string pwString((static_cast<const char*>(pw)), pwSize);
            // std::cout << "Hash " << pwString << " (Base64): " << decodedString << std::endl;
            if (decodedString.find("XEN11") != std::string::npos) {
                std::cout << "XEN11 found Hash " << decodedString << std::endl;
                saveToFile2(pwString);
            } 
            if(std::regex_search(decodedString, pattern) && is_within_five_minutes_of_hour2()){
                std::cout << "XUNI found Hash " << decodedString << std::endl;
                saveToFile2(pwString);
            }
            else {
            }
        }
    }

public:
    ParallelRunner(const BenchmarkDirector &director, PasswordGenerator &pwGen)
        : director(director), pwGen(pwGen), salt(new std::uint8_t[SALT_LENGTH]{}),
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
        nworkers = 1;
        nthreads = 1;
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

    FLAG_clear_internal_memory = 0;

    clock_type::time_point start = clock_type::now();

    ParallelRunner runner(director, pwGen);
    runner.wait();

    clock_type::time_point end = clock_type::now();
    clock_type::duration compTime = end - start;
    auto compTimeNs = toNanoseconds(compTime);

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
