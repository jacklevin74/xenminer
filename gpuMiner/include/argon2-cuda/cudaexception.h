#ifndef ARGON2_CUDA_CUDAEXCEPTION_H
#define ARGON2_CUDA_CUDAEXCEPTION_H

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <exception>

namespace argon2 {
namespace cuda {

#if HAVE_CUDA

class CudaException : public std::exception {
private:
    cudaError_t res;

public:
    CudaException(cudaError_t res) : res(res) { }

    const char *what() const noexcept override;

    static void check(cudaError_t res)
    {
        if (res != cudaSuccess) {
            throw CudaException(res);
        }
    }
};

#else

class CudaException : public std::exception { };

#endif

} // namespace cuda
} // namespace argon2

#endif // ARGON2_CUDA_CUDAEXCEPTION_H
