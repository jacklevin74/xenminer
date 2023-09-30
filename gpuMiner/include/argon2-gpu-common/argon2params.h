#ifndef ARGON2_ARGON2PARAMS_H
#define ARGON2_ARGON2PARAMS_H

#include <cstdint>

#if defined(__APPLE__) || defined(__MACOSX) || defined(_WIN32) || defined(_WIN32)
    #include <cstddef>// for size_t
#endif

#include "argon2-common.h"

namespace argon2 {

class Argon2Params
{
private:
    const void *salt, *secret, *ad;
    std::uint32_t outLen, saltLen, secretLen, adLen;
    std::uint32_t t_cost, m_cost, lanes;

    std::uint32_t segmentBlocks;

    static void digestLong(void *out, std::size_t outLen,
                           const void *in, std::size_t inLen);

    void initialHash(void *out, const void *pwd, std::size_t pwdLen,
                     Type type, Version version) const;

public:
    std::uint32_t getOutputLength() const { return outLen; }

    const void *getSalt() const { return salt; }
    std::uint32_t getSaltLength() const { return saltLen; }

    const void *getSecret() const { return secret; }
    std::uint32_t getSecretLength() const { return secretLen; }

    const void *getAssocData() const { return ad; }
    std::uint32_t getAssocDataLength() const { return adLen; }

    std::uint32_t getTimeCost() const { return t_cost; }
    std::uint32_t getMemoryCost() const { return m_cost; }
    std::uint32_t getLanes() const { return lanes; }

    std::uint32_t getSegmentBlocks() const { return segmentBlocks; }
    std::uint32_t getLaneBlocks() const {
        return segmentBlocks * ARGON2_SYNC_POINTS;
    }
    std::uint32_t getMemoryBlocks() const { return getLaneBlocks() * lanes; }
    std::size_t getMemorySize() const {
        return static_cast<std::size_t>(getMemoryBlocks()) * ARGON2_BLOCK_SIZE;
    }

    Argon2Params(
            std::size_t outLen,
            const void *salt, std::size_t saltLen,
            const void *secret, std::size_t secretLen,
            const void *ad, std::size_t adLen,
            std::size_t t_cost, std::size_t m_cost, std::size_t lanes);

    void fillFirstBlocks(void *memory, const void *pwd, std::size_t pwdLen,
                         Type type, Version version) const;

    void finalize(void *out, const void *memory) const;
};

} // namespace argon2

#endif // ARGON2_ARGON2PARAMS_H
