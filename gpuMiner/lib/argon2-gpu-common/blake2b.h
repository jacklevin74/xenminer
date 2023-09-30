#ifndef ARGON2_BLAKE2B_H
#define ARGON2_BLAKE2B_H

#include <cstdint>

#if defined(__APPLE__) || defined(__MACOSX) || defined(_WIN32)
    #include <cstddef>
#endif

#ifdef _WIN32
#  define EXPORT_SYMBOL1 __declspec(dllexport)
#else
#  define EXPORT_SYMBOL1
#endif

namespace argon2 {

class EXPORT_SYMBOL1 Blake2b
{
public:
    enum {
        BLOCK_BYTES = 128,
        OUT_BYTES = 64,
    };

private:
    std::uint64_t h[8];
    std::uint64_t t[2];
    std::uint8_t buf[BLOCK_BYTES];
    std::size_t bufLen;

    void compress(const void *block, std::uint64_t f0);
    void incrementCounter(std::uint64_t inc);

public:
    Blake2b() : h(), t(), buf(), bufLen(0) { }

    void init(std::size_t outlen);
    void update(const void *in, std::size_t inLen);
    void final(void *out, std::size_t outLen);
};

} // namespace argon2

#endif // ARGON2_BLAKE2B_H
