#ifndef CUDA_CHECK_HPP
#define CUDA_CHECK_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Abort with file/line context if a CUDA runtime call fails.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s (%d) at %s:%d\n  in: %s\n",     \
                         cudaGetErrorString(err_), static_cast<int>(err_),      \
                         __FILE__, __LINE__, #call);                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Check a kernel launch. Launch-configuration errors are caught immediately;
// in debug builds we also synchronize so that faults inside the kernel are
// reported at the launch site instead of at some later API call.
#ifdef NDEBUG
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#else
#define CUDA_CHECK_LAUNCH()                     \
    do {                                        \
        CUDA_CHECK(cudaGetLastError());         \
        CUDA_CHECK(cudaDeviceSynchronize());    \
    } while (0)
#endif

#endif
