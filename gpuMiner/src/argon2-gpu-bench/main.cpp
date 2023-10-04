#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "benchmark.h"
#include "openclexecutive.h"
#include "cudaexecutive.h"
#include "cpuexecutive.h"

#include <iostream>
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
using namespace libcommandline;

struct Arguments
{
    std::string mode = "cuda";

    std::size_t deviceIndex = 0;

    std::string outputType = "ns";
    std::string outputMode = "verbose";

    std::size_t batchSize = 0;
    std::string kernelType = "oneshot";
    bool precomputeRefs = false;

    std::string benchmarkDeviceName = "unknowDevice";
    bool benchmark = false;
    
    bool showHelp = false;
    bool listDevices = false;
};

static CommandLineParser<Arguments> buildCmdLineParser()
{
    static const auto positional = PositionalArgumentHandler<Arguments>(
                [] (Arguments &, const std::string &) {});

    std::vector<const CommandLineOption<Arguments>*> options {
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.listDevices = true; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.mode = mode; },
            "mode", 'm', "mode in which to run ('cuda' for CUDA, 'opencl' for OpenCL, or 'cpu' for CPU)", "cuda", "MODE"),

        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                state.deviceIndex = index;
            }), "device", 'd', "use device with index INDEX", "0", "INDEX"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &name) { state.benchmarkDeviceName = name; state.benchmark = true; },
            "device-name", 't', "use device with name NAME", "unknowDevice", "NAME"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.outputType = type; },
            "output-type", 'o', "what to output (ns|ns-per-hash)", "ns", "TYPE"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.outputMode = mode; },
            "output-mode", '\0', "output mode (verbose|raw|mean|mean-and-mdev)", "verbose", "MODE"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.batchSize = num;
            }), "batch-size", 'b', "number of tasks per batch", "16", "N"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.kernelType = type; },
            "kernel-type", 'k', "kernel type (by-segment|oneshot)", "by-segment", "TYPE"),
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.precomputeRefs = true; },
            "precompute-refs", 'p', "precompute reference indices with Argon2i"),

        new FlagOption<Arguments>(
            [] (Arguments &state) { state.showHelp = true; },
            "help", '?', "show this help and exit")
    };

    return CommandLineParser<Arguments>(
        "XENBlocks gpu miner: CUDA and OpenCL are supported.",
        positional, options);
}

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include "shared.h"
#include <limits>

int difficulty = 1727;
std::mutex mtx;
void read_difficulty_periodically(const std::string& filename) {
    while (true) {
        std::ifstream file(filename);
        if (file.is_open()) {
            int new_difficulty;
            if (file >> new_difficulty) { // read difficulty
                std::lock_guard<std::mutex> lock(mtx);
                if(difficulty != new_difficulty){
                    difficulty = new_difficulty; // update difficulty
                    std::cout << "Updated difficulty to " << difficulty << std::endl;
                }
            }
            file.close(); 
        } else {
            std::cerr << "The local difficult.txt file was not recognized" << std::endl;
        }
        
        // sleep for 3 seconds
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}
#include <atomic>
#include <csignal>
std::atomic<bool> running(true);
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    running = false;
    {
        std::lock_guard<std::mutex> lock(mtx);
        difficulty = difficulty - 1;
        std::cout << "change difficulty to " << difficulty << ", waiting process end" << std::endl;
    }
}
#include <iomanip>

int main(int, const char * const *argv)
{
    difficulty = 1727;
    // register signal SIGINT and signal handler
    signal(SIGINT, signalHandler);

    CommandLineParser<Arguments> parser = buildCmdLineParser();

    Arguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return ret;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }
    if(args.listDevices){
        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, 120, 1, 1,
                false, args.precomputeRefs, 20000000,
                args.outputMode, args.outputType);
        if (args.mode == "opencl") {
            OpenCLExecutive exec(args.deviceIndex, args.listDevices);
            exec.runBenchmark(director);
        } else if (args.mode == "cuda") {
            CudaExecutive exec(args.deviceIndex, args.listDevices);
            exec.runBenchmark(director);
        }
        return 0;
    }
    if(args.mode == "cuda"){
        #if HAVE_CUDA
        #else
            printf("Have no CUDA!\n");
            return -1;
        #endif
    }
    if(args.benchmark){
        // difficulty from 50 to 1000000 step 100
        int min_difficulty = 100;
        int max_difficulty = 1000000;
        int step = 100;
        int batchSize = args.batchSize;
        size_t usingMemory = 0;
        size_t totalMemory = 0;
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        int samples = 5;
        std::ostringstream oss;
        oss << std::put_time(&tm, "benchmark_%Y%m%d_%H%M%S_") << args.benchmarkDeviceName << ".csv";
        std::string fileName = oss.str();
        if(args.batchSize == 0){
            if (args.mode == "opencl") {
                cl_platform_id platform;
                clGetPlatformIDs(1, &platform, NULL);

                cl_uint numDevices;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices); // Assuming you are interested in GPU devices

                if(args.deviceIndex >= numDevices) {
                    // Handle error: Invalid device index
                    printf("Opencl device index out of range");
                    return -1;
                }

                cl_device_id* devices = new cl_device_id[numDevices];
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

                cl_device_id device = devices[args.deviceIndex]; // Get device by index

                cl_ulong memorySize;
                cl_ulong globalSize;
                clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memorySize, NULL);
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalSize, NULL);
                usingMemory = memorySize;
                totalMemory = globalSize;
            } else if (args.mode == "cuda") {
                #if HAVE_CUDA
                    cudaSetDevice(args.deviceIndex); // Set device by index
                    size_t freeMemory, tMemory;
                    cudaMemGetInfo(&freeMemory, &tMemory);
                    usingMemory = freeMemory;
                    totalMemory = tMemory;
                #endif
            }
        }

        std::ofstream outputFile(fileName, std::ios::app);
        outputFile << "# GPU Model: " << args.benchmarkDeviceName << "\n";
        outputFile << "# Date: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "\n";
        outputFile << "# Difficulty: " << min_difficulty << " to " << max_difficulty << " step " << step << "\n";
        outputFile << "# Samples: " << samples << "\n";
        outputFile << "# Total Memory: " << totalMemory << "\n";
        outputFile << "# Using Memory: " << usingMemory << "\n";
        outputFile << "Difficulty,BatchSize,HashSpeed\n";
        for(int mcost =min_difficulty; mcost <= max_difficulty; mcost+=step){
            if(100<mcost && mcost<1000) step = 10;
            if(1000<mcost && mcost<10000) step = 100;
            if(10000<mcost && mcost<100000) step = 1000;
            if(100000<mcost && mcost<1000000) step = 10000;

            if(!running)break;
            // bs from 1 to batchsize, step 2^x
            batchSize = usingMemory / mcost / 1.01 / 1024;
            // int initbs = batchSize>16?16:1;
            int initbs = batchSize;
            for(int bs = initbs; bs <= batchSize; bs*=2){
                if(!running)break;
                int rate = 0;
                BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                    1, mcost, 1, batchSize,
                    false, args.precomputeRefs, samples,
                    args.outputMode, args.outputType, true);
                if (args.mode == "opencl") {
                    OpenCLExecutive exec(args.deviceIndex, args.listDevices);
                    rate = exec.runBenchmark(director);
                } else if (args.mode == "cuda") {
                    CudaExecutive exec(args.deviceIndex, args.listDevices);
                    rate = exec.runBenchmark(director);
                }
                outputFile << mcost << "," << batchSize << "," << rate << "\n";
            }
            printf("benchmark difficulty:%d, batchSize:%d\n", mcost, batchSize);
        }
        outputFile.close();
        return 0;
    }
    std::ifstream file("difficulty.txt");
    if (file.is_open()) {
        int new_difficulty;
        if (file >> new_difficulty) { // read difficulty
            std::lock_guard<std::mutex> lock(mtx);
            if(difficulty != new_difficulty){
                difficulty = new_difficulty; // update difficulty
                std::cout << "Updated difficulty to " << difficulty << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "The local difficult.txt file was not recognized" << std::endl;
    }
    // start a thread to read difficulty from file
    std::thread t(read_difficulty_periodically, "difficulty.txt"); 
    t.detach(); // detach thread from main thread, so it can run independently
    for(int i = 0; i < std::numeric_limits<size_t>::max(); i++){
        if(!running)break;

        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Current difficulty: " << difficulty << std::endl;
        }
        int mcost = difficulty;
        int batchSize = args.batchSize;
        if(args.batchSize == 0){
            if (args.mode == "opencl") {
                cl_platform_id platform;
                clGetPlatformIDs(1, &platform, NULL);

                cl_uint numDevices;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices); // Assuming you are interested in GPU devices

                if(args.deviceIndex >= numDevices) {
                    // Handle error: Invalid device index
                    printf("Opencl device index out of range");
                    return -1;
                }

                cl_device_id* devices = new cl_device_id[numDevices];
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

                cl_device_id device = devices[args.deviceIndex]; // Get device by index

                cl_ulong memorySize;
                clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memorySize, NULL);
                batchSize = memorySize / mcost / 1.01 / 1024;
            } else if (args.mode == "cuda") {
                #if HAVE_CUDA
                    cudaSetDevice(args.deviceIndex); // Set device by index
                    size_t freeMemory, totalMemory;
                    cudaMemGetInfo(&freeMemory, &totalMemory);

                    batchSize = freeMemory / 1.01 / mcost / 1024;
                #endif

            } else{
                batchSize = 100;
            }
            printf("using batchsize:%d\n", batchSize);
        }

        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, mcost, 1, batchSize,
                false, args.precomputeRefs, std::numeric_limits<size_t>::max(),
                args.outputMode, args.outputType);
        if (args.mode == "opencl") {
            OpenCLExecutive exec(args.deviceIndex, args.listDevices);
            exec.runBenchmark(director);
        } else if (args.mode == "cuda") {
            CudaExecutive exec(args.deviceIndex, args.listDevices);
            exec.runBenchmark(director);
        }else{
            CpuExecutive exec(args.deviceIndex, args.listDevices);
            exec.runBenchmark(director);
        }
    }
    return 0;
}

