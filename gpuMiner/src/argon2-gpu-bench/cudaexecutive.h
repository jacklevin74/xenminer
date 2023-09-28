#ifndef CUDAEXECUTIVE_H
#define CUDAEXECUTIVE_H

#include "benchmark.h"

class CudaExecutive : public BenchmarkExecutive
{
private:
    std::size_t deviceIndex;
    bool listDevices;

public:
    CudaExecutive(std::size_t deviceIndex, bool listDevices)
        : deviceIndex(deviceIndex), listDevices(listDevices)
    {
    }

    int runBenchmark(const BenchmarkDirector &director) const override;
};

#endif // CUDAEXECUTIVE_H
