#include "benchmark.h"

#include <iostream>

Argon2Runner::~Argon2Runner() { }

BenchmarkExecutive::~BenchmarkExecutive() { }
#include "shared.h"
#include <mutex>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstring>
#ifdef _WIN32
#include <Windows.h>
#define GET_PROCESS_ID GetCurrentProcessId
#else
#include <unistd.h>
#define GET_PROCESS_ID getpid
#endif
static bool create_directory2(const std::string& path) {
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
static void saveHashSpeedToFile(double hashspeed) {
    pid_t processId = GET_PROCESS_ID();
    std::ostringstream dirStream;
    dirStream << "hash_rates/";
    std::string dirStr = dirStream.str();

    if (!create_directory2(dirStr)) {
        return;
    }
    std::ostringstream filename;
    filename << dirStr << "/" << "hashrate_" + std::to_string(processId) + ".txt";
    std::ofstream outFile(filename.str());
    if(!outFile) {
        std::cerr << "Error opening file " << filename.str() << std::endl;
        return;
    }
    outFile << hashspeed;
    outFile.close();
}


int BenchmarkDirector::runBenchmark(Argon2Runner &runner) const
{
    using namespace std;

    auto start_time = chrono::system_clock::now();
    const std::string desc = "Mining";
    const std::string unit = "Hashes";
    DummyPasswordGenerator pwGen;
    RunTimeStats stats(batchSize);
    long long int hashtotal = 0;
    if(this->benchmark){
        difficulty = m_cost;
    }
    for (std::size_t i = 0; i < samples; i++) {
        // break when mcost changed
        if(!this->benchmark){
            {
                std::lock_guard<std::mutex> lock(mtx);
                if(difficulty != m_cost){
                    std::cout << "difficulty changed: " <<m_cost<<">>"<< difficulty <<", end"<< std::endl;
                    break;
                }
            }
        }
        auto ctime = runner.runBenchmark(*this, pwGen);
        hashtotal += batchSize;

        auto elapsed_time = chrono::system_clock::now() - start_time;
        auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count();
        auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60;
        auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60;
        auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
        double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000;  // Multiply by 1000 to convert rate to per second
        std::cout << desc << ": " << hashtotal << " " << unit << " [";
        if (hours)
            std::cout << std::setw(2) << std::setfill('0') << hours << ":";
        
        std::cout << std::setw(2) << std::setfill('0') << minutes << ":"
                  << std::setw(2) << std::setfill('0') << seconds;
        std::cout << ", " << std::fixed << std::setprecision(2) << rate << " " << unit << "/s, "
                  << "Difficulty=" << difficulty << "]\r";
        std::cout.flush();
        stats.addSample(ctime);
        saveHashSpeedToFile(rate);
    }
    stats.close();
    auto elapsed_time = chrono::system_clock::now() - start_time;
    auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count();
    auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60;
    auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60;
    auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000;  // Multiply by 1000 to convert rate to per second
    std::cout << desc << ": " << hashtotal << " " << unit << " [";
    if (hours)
        std::cout << std::setw(2) << std::setfill('0') << hours << ":";
    
    std::cout << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds;
    std::cout << ", " << std::fixed << std::setprecision(2) << rate << " " << unit << "/s"<< "]"<< std::endl;
    auto rateNs = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_time).count();
    auto rr = static_cast<double>(rateNs) / hashtotal;
    std::cout << "Mean computation time (per hash): "
                 << std::fixed << std::setprecision(2) << RunTimeStats::repr(nanosecs(rr));
    std::cout << std::endl;

    return rate;
}
