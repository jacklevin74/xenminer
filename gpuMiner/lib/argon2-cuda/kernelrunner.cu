/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "kernelrunner.h"
#include "cudaexception.h"

#include <stdexcept>
#ifndef NDEBUG
#include <iostream>
#endif

#define ARGON2_D  0
#define ARGON2_I  1
#define ARGON2_ID 2

#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)

namespace argon2 {
namespace cuda {

using namespace std;

__device__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}

struct u64_shuffle_buf {
    uint32_t lo[THREADS_PER_LANE];
    uint32_t hi[THREADS_PER_LANE];
};

__device__ uint64_t u64_shuffle(uint64_t v, uint32_t thread_src,
                                uint32_t thread, struct u64_shuffle_buf *buf)
{
    uint32_t lo = u64_lo(v);
    uint32_t hi = u64_hi(v);

    buf->lo[thread] = lo;
    buf->hi[thread] = hi;

    __syncthreads();

    lo = buf->lo[thread_src];
    hi = buf->hi[thread_src];

    return u64_build(hi, lo);
}

struct block_g {
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_th {
    uint64_t a, b, c, d;
};

__device__ uint64_t cmpeq_mask(uint32_t test, uint32_t ref)
{
    uint32_t x = -(uint32_t)(test == ref);
    return u64_build(x, x);
}

__device__ uint64_t block_th_get(const struct block_th *b, uint32_t idx)
{
    uint64_t res = 0;
    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;
    return res;
}

__device__ void block_th_set(struct block_th *b, uint32_t idx, uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

__device__ void move_block(struct block_th *dst, const struct block_th *src)
{
    *dst = *src;
}

__device__ void xor_block(struct block_th *dst, const struct block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__ void load_block(struct block_th *dst, const struct block_g *src,
                           uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void load_block_xor(struct block_th *dst, const struct block_g *src,
                               uint32_t thread)
{
    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void store_block(struct block_g *dst, const struct block_th *src,
                            uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;
}

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = u64_lo(x);
    uint32_t ylo = u64_lo(y);
    return x + y + 2 * u64_build(__umulhi(xlo, ylo), xlo * ylo);
}

__device__ void g(struct block_th *block)
{
    uint64_t a, b, c, d;
    a = block->a;
    b = block->b;
    c = block->c;
    d = block->d;

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->a = a;
    block->b = b;
    block->c = c;
    block->d = d;
}

template<class shuffle>
__device__ void apply_shuffle(struct block_th *block, uint32_t thread,
                              struct u64_shuffle_buf *buf)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t src_thr = shuffle::apply(thread, i);

        uint64_t v = block_th_get(block, i);
        v = u64_shuffle(v, src_thr, thread, buf);
        block_th_set(block, i, v);
    }
}

__device__ void transpose(struct block_th *block, uint32_t thread,
                          struct u64_shuffle_buf *buf)
{
    uint32_t thread_group = (thread & 0x0C) >> 2;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t thr = (i << 2) ^ thread;
        uint32_t idx = thread_group ^ i;

        uint64_t v = block_th_get(block, idx);
        v = u64_shuffle(v, thr, thread, buf);
        block_th_set(block, idx, v);
    }
}

struct identity_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return thread;
    }
};

struct shift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct unshift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct shift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

struct unshift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

__device__ void shuffle_block(struct block_th *block, uint32_t thread,
                              struct u64_shuffle_buf *buf)
{
    transpose(block, thread, buf);

    g(block);

    apply_shuffle<shift1_shuffle>(block, thread, buf);

    g(block);

    apply_shuffle<unshift1_shuffle>(block, thread, buf);
    transpose(block, thread, buf);

    g(block);

    apply_shuffle<shift2_shuffle>(block, thread, buf);

    g(block);

    apply_shuffle<unshift2_shuffle>(block, thread, buf);
}

__device__ void next_addresses(struct block_th *addr, struct block_th *tmp,
                               uint32_t thread_input, uint32_t thread,
                               struct u64_shuffle_buf *buf)
{
    addr->a = u64_build(0, thread_input);
    addr->b = 0;
    addr->c = 0;
    addr->d = 0;

    shuffle_block(addr, thread, buf);

    addr->a ^= u64_build(0, thread_input);
    move_block(tmp, addr);

    shuffle_block(addr, thread, buf);

    xor_block(addr, tmp);
}

__device__ void compute_ref_pos(
        uint32_t lanes, uint32_t segment_blocks,
        uint32_t pass, uint32_t lane, uint32_t slice, uint32_t offset,
        uint32_t *ref_lane, uint32_t *ref_index)
{
    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    *ref_lane = *ref_lane % lanes;

    uint32_t base;
    if (pass != 0) {
        base = lane_blocks - segment_blocks;
    } else {
        if (slice == 0) {
            *ref_lane = lane;
        }
        base = slice * segment_blocks;
    }

    uint32_t ref_area_size = base + offset - 1;
    if (*ref_lane != lane) {
        ref_area_size = min(ref_area_size, base);
    }

    *ref_index = __umulhi(*ref_index, *ref_index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);

    if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
        *ref_index += (slice + 1) * segment_blocks;
        if (*ref_index >= lane_blocks) {
            *ref_index -= lane_blocks;
        }
    }
}

struct ref {
    uint32_t ref_lane;
    uint32_t ref_index;
};


template<uint32_t version>
__device__ void argon2_core(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp,
        struct u64_shuffle_buf *shuffle_buf, uint32_t lanes,
        uint32_t thread, uint32_t pass, uint32_t ref_index, uint32_t ref_lane)
{
    struct block_g *mem_ref = memory + ref_index * lanes + ref_lane;

    if (version != ARGON2_VERSION_10 && pass != 0) {
        load_block(tmp, mem_curr, thread);
        load_block_xor(prev, mem_ref, thread);
        xor_block(tmp, prev);
    } else {
        load_block_xor(prev, mem_ref, thread);
        move_block(tmp, prev);
    }

    shuffle_block(prev, thread, shuffle_buf);

    xor_block(prev, tmp);

    store_block(mem_curr, prev, thread);
}



template<uint32_t type, uint32_t version>
__device__ void argon2_step(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp, struct block_th *addr,
        struct u64_shuffle_buf *shuffle_buf, uint32_t lanes,
        uint32_t segment_blocks, uint32_t thread, uint32_t *thread_input,
        uint32_t lane, uint32_t pass, uint32_t slice, uint32_t offset)
{
    uint32_t ref_index, ref_lane;

    if (type == ARGON2_I || (type == ARGON2_ID && pass == 0 &&
            slice < ARGON2_SYNC_POINTS / 2)) {
        uint32_t addr_index = offset % ARGON2_QWORDS_IN_BLOCK;
        if (addr_index == 0) {
            if (thread == 6) {
                ++*thread_input;
            }
            next_addresses(addr, tmp, *thread_input, thread, shuffle_buf);
        }

        uint32_t thr = addr_index % THREADS_PER_LANE;
        uint32_t idx = addr_index / THREADS_PER_LANE;

        uint64_t v = block_th_get(addr, idx);
        v = u64_shuffle(v, thr, thread, shuffle_buf);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    } else {
        uint64_t v = u64_shuffle(prev->a, 0, thread, shuffle_buf);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    }

    compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset,
                    &ref_lane, &ref_index);

    argon2_core<version>(memory, mem_curr, prev, tmp, shuffle_buf, lanes,
                         thread, pass, ref_index, ref_lane);
}

template<uint32_t type, uint32_t version>
__global__ void argon2_kernel_oneshot(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks)
{
    extern __shared__ struct u64_shuffle_buf shuffle_bufs[];
    struct u64_shuffle_buf *shuffle_buf =
            &shuffle_bufs[0];

    uint32_t job_id = blockIdx.x;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += (size_t)job_id * 1 * lane_blocks;

    struct block_th prev, addr, tmp;
    uint32_t thread_input;

    switch (thread) {
    case 1:
        thread_input = 0;
        break;
    case 3:
        thread_input = lane_blocks;
        break;
    case 4:
        thread_input = 1;
        break;
    case 5:
        thread_input = type;
        break;
    default:
        thread_input = 0;
        break;
    }

    if (segment_blocks > 2) {
        if (thread == 6) {
            ++thread_input;
        }
        next_addresses(&addr, &tmp, thread_input, thread, shuffle_buf);
    }

    struct block_g *mem_lane = memory;
    struct block_g *mem_prev = mem_lane + 1;
    struct block_g *mem_curr = mem_lane + 2;

    load_block(&prev, mem_prev, thread);

    uint32_t skip = 2;
    for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
        for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
            if (skip > 0) {
                --skip;
                continue;
            }

            argon2_step<type, version>(
                        memory, mem_curr, &prev, &tmp, &addr, shuffle_buf,
                        1, segment_blocks, thread, &thread_input,
                        0, 0, slice, offset);

            mem_curr ++;
        }

        __syncthreads();

        if (thread == 2) {
            ++thread_input;
        }
        if (thread == 6) {
            thread_input = 0;
        }
    }
    mem_curr = mem_lane;
}

KernelRunner::KernelRunner(uint32_t type, uint32_t version, uint32_t passes,
                           uint32_t lanes, uint32_t segmentBlocks,
                           size_t batchSize, bool bySegment, bool precompute)
    : type(type), version(version), passes(passes), lanes(lanes),
      segmentBlocks(segmentBlocks), batchSize(batchSize), bySegment(bySegment),
      precompute(precompute), stream(), memory(), refs(),
      start(), end(), kernelStart(), kernelEnd(),
      blocksIn(new uint8_t[batchSize * 1 * 2 * ARGON2_BLOCK_SIZE]),
      blocksOut(new uint8_t[batchSize * 1 * ARGON2_BLOCK_SIZE])
{
    // FIXME: check overflow:
    size_t memorySize = batchSize * 1 * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;

#ifndef NDEBUG
        std::cerr << "[INFO] Allocating " << memorySize << " bytes for memory..."
                  << std::endl;
#endif

    CudaException::check(cudaMalloc(&memory, memorySize));

    CudaException::check(cudaEventCreate(&start));
    CudaException::check(cudaEventCreate(&end));
    CudaException::check(cudaEventCreate(&kernelStart));
    CudaException::check(cudaEventCreate(&kernelEnd));

    CudaException::check(cudaStreamCreate(&stream));
}

KernelRunner::~KernelRunner()
{
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (end != nullptr) {
        cudaEventDestroy(end);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (memory != nullptr) {
        cudaFree(memory);
    }
    if (refs != nullptr) {
        cudaFree(refs);
    }
}

void *KernelRunner::getInputMemory(size_t jobId) const
{
    size_t copySize = 1 * 2 * ARGON2_BLOCK_SIZE;
    return blocksIn.get() + jobId * copySize;
}
const void *KernelRunner::getOutputMemory(size_t jobId) const
{
    size_t copySize = 1 * ARGON2_BLOCK_SIZE;
    return blocksOut.get() + jobId * copySize;
}

void KernelRunner::copyInputBlocks()
{
    size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    size_t copySize = 1 * 2 * ARGON2_BLOCK_SIZE;

    CudaException::check(cudaMemcpy2DAsync(
                             memory, jobSize,
                             blocksIn.get(), copySize,
                             copySize, batchSize, cudaMemcpyHostToDevice,
                             stream));
}

void KernelRunner::copyOutputBlocks()
{
    size_t jobSize = static_cast<size_t>(lanes) * segmentBlocks
            * ARGON2_SYNC_POINTS * ARGON2_BLOCK_SIZE;
    size_t copySize = lanes * ARGON2_BLOCK_SIZE;
    uint8_t *mem = static_cast<uint8_t *>(memory);

    CudaException::check(cudaMemcpy2DAsync(
                             blocksOut.get(), copySize,
                             mem + (jobSize - copySize), jobSize,
                             copySize, batchSize, cudaMemcpyDeviceToHost,
                             stream));
}


void KernelRunner::runKernelOneshot(uint32_t lanesPerBlock,
                                    size_t jobsPerBlock)
{
    if (lanesPerBlock != lanes) {
        throw std::logic_error("Invalid lanesPerBlock!");
    }

    if (jobsPerBlock > batchSize || batchSize % jobsPerBlock != 0) {
        throw std::logic_error("Invalid jobsPerBlock!");
    }

    struct block_g *memory_blocks = (struct block_g *)memory;
    dim3 blocks = dim3(1, 1, batchSize);
    uint32_t shared_size = 1 * sizeof(struct u64_shuffle_buf);
    argon2_kernel_oneshot<ARGON2_ID, ARGON2_VERSION_13>
            <<<batchSize, THREADS_PER_LANE, shared_size, stream>>>(
                memory_blocks, 1, 1, segmentBlocks);
}

void KernelRunner::run(uint32_t lanesPerBlock, size_t jobsPerBlock)
{
    CudaException::check(cudaEventRecord(start, stream));

    copyInputBlocks();

    CudaException::check(cudaEventRecord(kernelStart, stream));
    
    runKernelOneshot(lanesPerBlock, jobsPerBlock);

    CudaException::check(cudaGetLastError());

    CudaException::check(cudaEventRecord(kernelEnd, stream));

    copyOutputBlocks();

    CudaException::check(cudaEventRecord(end, stream));
}

float KernelRunner::finish()
{
    float time = 0.0;
    CudaException::check(cudaStreamSynchronize(stream));
    CudaException::check(cudaEventElapsedTime(&time, kernelStart, kernelEnd));
    return time;
}

} // cuda
} // argon2
