#ifndef RUNTIMESTATISTICS_H
#define RUNTIMESTATISTICS_H

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

typedef uintmax_t nanosecs;

template<class duration>
static nanosecs toNanoseconds(const duration &d)
{
    return d.count() * ((nanosecs)1000 * 1000 * 1000 * duration::period::num / duration::period::den);
}

class DataSet
{
private:
    std::vector<uintmax_t> samples;
    uintmax_t sum;
    uintmax_t mean;
    uintmax_t devSum;
    uintmax_t devMean;

public:
    const std::vector<uintmax_t> &getSamples() const { return samples; }
    uintmax_t getMean() const { return mean; }
    uintmax_t getMeanDeviation() const { return devMean; }

    double getMeanDeviationPerMean() const { return (double)devMean / mean; }

    DataSet()
        : samples(), sum(0), mean(0),
          devSum(0), devMean(0)
    {
    }

    void addSample(uintmax_t sample)
    {
        sum += sample;
        samples.push_back(sample);
    }

    void close()
    {
        mean = sum / samples.size();

        devSum = std::accumulate(
            samples.begin(), samples.end(), (uintmax_t)0,
            [=](uintmax_t s, uintmax_t x)
        {
            auto dev = (intmax_t)(x - mean);
            return s + (dev >= 0 ? dev : -dev);
        });
        devMean = devSum / samples.size();
    }
};

class RunTimeStats
{
private:
    std::uintmax_t batchSize;
    DataSet ns;
    DataSet nsPerHash;

public:
    const DataSet &getNanoseconds() const { return ns; }
    const DataSet &getNanosecsPerHash() const { return nsPerHash; }

    RunTimeStats(std::size_t batchSize)
        : batchSize(batchSize), ns(), nsPerHash()
    {
    }

    void addSample(nanosecs sample)
    {
        ns.addSample(sample);
        nsPerHash.addSample(sample / batchSize);
    }

    void close()
    {
        ns.close();
        nsPerHash.close();
    }

    static double toMinutes(nanosecs ns)
    {
        return (double)ns / ((nanosecs)60 * 1000 * 1000 * 1000);
    }

    static double toSeconds(nanosecs ns)
    {
        return (double)ns / ((nanosecs)1000 * 1000 * 1000);
    }

    static double toMilliSeconds(nanosecs ns)
    {
        return (double)ns / ((nanosecs)1000 * 1000);
    }

    static double toMicroSeconds(nanosecs ns)
    {
        return (double)ns / (nanosecs)1000;
    }

    static std::string repr(nanosecs ns)
    {
        if (ns < (nanosecs)1000) {
            return std::to_string(ns) + " ns";
        }
        if (ns < (nanosecs)1000 * 1000) {
            return std::to_string(toMicroSeconds(ns)) + " us";
        }
        if (ns < (nanosecs)1000 * 1000 * 1000) {
            return std::to_string(toMilliSeconds(ns)) + " ms";
        }
        if (ns < (nanosecs)60 * 1000 * 1000 * 1000) {
            return std::to_string(toSeconds(ns)) + " s";
        }
        return std::to_string(toMinutes(ns)) + " min";
    }
};

#endif // RUNTIMESTATISTICS_H
