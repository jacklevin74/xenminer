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
int BenchmarkDirector::runBenchmark(Argon2Runner &runner) const
{
    using namespace std;

    auto start_time = chrono::system_clock::now();
    const std::string desc = "Mining";
    const std::string unit = "Hashes";
    DummyPasswordGenerator pwGen;
    RunTimeStats stats(batchSize);
    long long int hashtotal = 0;
    for (std::size_t i = 0; i < samples; i++) {
        // break when mcost changed
        {
            std::lock_guard<std::mutex> lock(mtx);
            if(difficulty != m_cost){
                std::cout << "difficulty changed: " <<m_cost<<">>"<< difficulty <<", end"<< std::endl;
                break;
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

    return 0;
}
