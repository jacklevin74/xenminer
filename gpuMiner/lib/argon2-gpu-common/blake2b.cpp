#include "blake2b.h"

#include <cstring>

namespace argon2 {

static const std::uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

static const unsigned int blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

#define rotr64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

#define G(m, r, i, a, b, c, d) \
    do { \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]]; \
        d = rotr64(d ^ a, 32); \
        c = c + d; \
        b = rotr64(b ^ c, 24); \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]]; \
        d = rotr64(d ^ a, 16); \
        c = c + d; \
        b = rotr64(b ^ c, 63); \
    } while ((void)0, 0)

#define ROUND(m, v, r) \
    do { \
        G(m, r, 0, v[0], v[4], v[ 8], v[12]); \
        G(m, r, 1, v[1], v[5], v[ 9], v[13]); \
        G(m, r, 2, v[2], v[6], v[10], v[14]); \
        G(m, r, 3, v[3], v[7], v[11], v[15]); \
        G(m, r, 4, v[0], v[5], v[10], v[15]); \
        G(m, r, 5, v[1], v[6], v[11], v[12]); \
        G(m, r, 6, v[2], v[7], v[ 8], v[13]); \
        G(m, r, 7, v[3], v[4], v[ 9], v[14]); \
    } while ((void)0, 0)

static std::uint64_t load64(const void *src)
{
    auto in = static_cast<const std::uint8_t *>(src);
    std::uint64_t res = *in++;
    res |= static_cast<std::uint64_t>(*in++) << 8;
    res |= static_cast<std::uint64_t>(*in++) << 16;
    res |= static_cast<std::uint64_t>(*in++) << 24;
    res |= static_cast<std::uint64_t>(*in++) << 32;
    res |= static_cast<std::uint64_t>(*in++) << 40;
    res |= static_cast<std::uint64_t>(*in++) << 48;
    res |= static_cast<std::uint64_t>(*in++) << 56;
    return res;
}

static void store64(void *dst, std::uint64_t v)
{
    auto out = static_cast<std::uint8_t *>(dst);
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v);
}

void Blake2b::init(std::size_t outlen)
{
    t[1] = t[0] = 0;
    bufLen = 0;

    std::memcpy(h, blake2b_IV, sizeof(h));

    h[0] ^= static_cast<std::uint64_t>(outlen) |
            (UINT64_C(1) << 16) | (UINT64_C(1) << 24);
}

void Blake2b::compress(const void *block, std::uint64_t f0)
{
    std::uint64_t m[16];
    std::uint64_t v[16];

    auto in = static_cast<const uint64_t *>(block);

    m[ 0] = load64(in +  0);
    m[ 1] = load64(in +  1);
    m[ 2] = load64(in +  2);
    m[ 3] = load64(in +  3);
    m[ 4] = load64(in +  4);
    m[ 5] = load64(in +  5);
    m[ 6] = load64(in +  6);
    m[ 7] = load64(in +  7);
    m[ 8] = load64(in +  8);
    m[ 9] = load64(in +  9);
    m[10] = load64(in + 10);
    m[11] = load64(in + 11);
    m[12] = load64(in + 12);
    m[13] = load64(in + 13);
    m[14] = load64(in + 14);
    m[15] = load64(in + 15);

    v[ 0] = h[0];
    v[ 1] = h[1];
    v[ 2] = h[2];
    v[ 3] = h[3];
    v[ 4] = h[4];
    v[ 5] = h[5];
    v[ 6] = h[6];
    v[ 7] = h[7];
    v[ 8] = blake2b_IV[0];
    v[ 9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ t[0];
    v[13] = blake2b_IV[5] ^ t[1];
    v[14] = blake2b_IV[6] ^ f0;
    v[15] = blake2b_IV[7];

    ROUND(m, v, 0);
    ROUND(m, v, 1);
    ROUND(m, v, 2);
    ROUND(m, v, 3);
    ROUND(m, v, 4);
    ROUND(m, v, 5);
    ROUND(m, v, 6);
    ROUND(m, v, 7);
    ROUND(m, v, 8);
    ROUND(m, v, 9);
    ROUND(m, v, 10);
    ROUND(m, v, 11);

    h[0] ^= v[0] ^ v[ 8];
    h[1] ^= v[1] ^ v[ 9];
    h[2] ^= v[2] ^ v[10];
    h[3] ^= v[3] ^ v[11];
    h[4] ^= v[4] ^ v[12];
    h[5] ^= v[5] ^ v[13];
    h[6] ^= v[6] ^ v[14];
    h[7] ^= v[7] ^ v[15];
}

void Blake2b::incrementCounter(std::uint64_t inc)
{
    t[0] += inc;
    t[1] += (t[0] < inc);
}

void Blake2b::update(const void *in, std::size_t inLen)
{
    auto bin = static_cast<const std::uint8_t *>(in);

    if (bufLen + inLen > BLOCK_BYTES) {
        std::size_t have = bufLen;
        std::size_t left = BLOCK_BYTES - have;
        std::memcpy(buf + have, bin, left);

        incrementCounter(BLOCK_BYTES);
        compress(buf, 0);

        bufLen = 0;
        inLen -= left;
        bin += left;

        while (inLen > BLOCK_BYTES) {
            incrementCounter(BLOCK_BYTES);
            compress(bin, 0);
            inLen -= BLOCK_BYTES;
            bin += BLOCK_BYTES;
        }
    }
    std::memcpy(buf + bufLen, bin, inLen);
    bufLen += inLen;
}

void Blake2b::final(void *out, std::size_t outLen)
{
    std::uint8_t buffer[OUT_BYTES] = {0};

    incrementCounter(bufLen);
    std::memset(buf + bufLen, 0, BLOCK_BYTES - bufLen);
    compress(buf, UINT64_C(0xFFFFFFFFFFFFFFFF));

    for (unsigned int i = 0; i < 8; i++) {
        store64(buffer + i * sizeof(std::uint64_t), h[i]);
    }

    std::memcpy(out, buffer, outLen);
}

} // namespace argon2
