#ifndef CPUEXECUTIVE_H
#define CPUEXECUTIVE_H

#include "benchmark.h"

class CpuExecutive : public BenchmarkExecutive
{
private:
    std::size_t deviceIndex;
    bool listDevices;

public:
    CpuExecutive(std::size_t deviceIndex, bool listDevices)
        : deviceIndex(deviceIndex), listDevices(listDevices)
    {
    }

    int runBenchmark(const BenchmarkDirector &director) const override;
};

#endif // CPUEXECUTIVE_H
