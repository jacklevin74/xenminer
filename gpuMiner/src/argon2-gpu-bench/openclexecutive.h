#ifndef OPENCLEXECUTIVE_H
#define OPENCLEXECUTIVE_H

#include "benchmark.h"

class OpenCLExecutive : public BenchmarkExecutive
{
private:
    std::size_t deviceIndex;
    bool listDevices;

public:
    OpenCLExecutive(std::size_t deviceIndex, bool listDevices)
        : deviceIndex(deviceIndex), listDevices(listDevices)
    {
    }

    int runBenchmark(const BenchmarkDirector &director) const override;
};

#endif // OPENCLEXECUTIVE_H
