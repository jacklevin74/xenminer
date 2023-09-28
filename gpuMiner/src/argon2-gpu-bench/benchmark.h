#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "pwgen.h"
#include "runtimestatistics.h"

#include "argon2-gpu-common/argon2-common.h"

class BenchmarkDirector;

class Argon2Runner
{
public:
    virtual ~Argon2Runner();
    virtual nanosecs runBenchmark(const BenchmarkDirector &director,
                                  PasswordGenerator &pwGen) = 0;
};

class BenchmarkDirector
{
private:
    std::string progname;
    argon2::Type type;
    argon2::Version version;
    std::size_t t_cost, m_cost, lanes;
    std::size_t batchSize, samples;
    bool bySegment, precomputeRefs;
    std::string outputMode, outputType;
    bool beVerbose;

public:
    const std::string &getProgname() const { return progname; }
    argon2::Type getType() const { return type; }
    argon2::Version getVersion() const { return version; }
    std::size_t getTimeCost() const { return t_cost; }
    std::size_t getMemoryCost() const { return m_cost; }
    std::size_t getLanes() const { return lanes; }
    std::size_t getBatchSize() const { return batchSize; }
    bool isBySegment() const { return bySegment; }
    bool isPrecomputeRefs() const { return precomputeRefs; }
    bool isVerbose() const { return beVerbose; }

    BenchmarkDirector(const std::string &progname,
                      argon2::Type type, argon2::Version version,
                      std::size_t t_cost, std::size_t m_cost, std::size_t lanes,
                      std::size_t batchSize, bool bySegment,
                      bool precomputeRefs, std::size_t samples,
                      const std::string &outputMode,
                      const std::string &outputType)
        : progname(progname), type(type), version(version),
          t_cost(t_cost), m_cost(m_cost), lanes(lanes), batchSize(batchSize),
          samples(samples), bySegment(bySegment), precomputeRefs(precomputeRefs),
          outputMode(outputMode), outputType(outputType),
          beVerbose(outputMode == "verbose")
    {
    }

    int runBenchmark(Argon2Runner &runner) const;
};

class BenchmarkExecutive
{
public:
    virtual ~BenchmarkExecutive();
    virtual int runBenchmark(const BenchmarkDirector &director) const = 0;
};

#endif // BENCHMARK_H

