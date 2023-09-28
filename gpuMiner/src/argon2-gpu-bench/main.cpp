#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "benchmark.h"
#include "openclexecutive.h"
#include "cudaexecutive.h"
#include "cpuexecutive.h"

#include <iostream>

using namespace libcommandline;

struct Arguments
{
    std::string mode = "cuda";

    std::size_t deviceIndex = 0;

    std::string outputType = "ns";
    std::string outputMode = "verbose";

    std::size_t batchSize = 16;
    std::string kernelType = "oneshot";
    bool precomputeRefs = false;

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
        
        // sleep for 10 seconds
        std::this_thread::sleep_for(std::chrono::seconds(10));
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

    for(int i = 0; i < 200000; i++){
        if(!running)break;

        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Current difficulty: " << difficulty << std::endl;
        }
        int mcost = difficulty;
        BenchmarkDirector director(argv[0], argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                1, mcost, 1, args.batchSize,
                false, args.precomputeRefs, 20000000,
                args.outputMode, args.outputType);
        CudaExecutive exec(args.deviceIndex, args.listDevices);
        exec.runBenchmark(director);
    }

    return 0;
}

