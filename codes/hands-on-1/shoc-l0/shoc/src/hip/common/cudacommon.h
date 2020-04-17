#ifndef CUDACOMMON_H
#define CUDACOMMON_H



#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                              \
    if(error != HIPBLAS_STATUS_SUCCESS)                         \
    {                                                           \
        fprintf(stderr, "rocBLAS error: ");                     \
        if(error == HIPBLAS_STATUS_NOT_INITIALIZED)             \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
        if(error == HIPBLAS_STATUS_ALLOC_FAILED)                \
            fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
        if(error == HIPBLAS_STATUS_INVALID_VALUE)               \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
        if(error == HIPBLAS_STATUS_MAPPING_ERROR)               \
            fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
        if(error == HIPBLAS_STATUS_EXECUTION_FAILED)            \
            fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
        if(error == HIPBLAS_STATUS_INTERNAL_ERROR)              \
            fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
        if(error == HIPBLAS_STATUS_NOT_SUPPORTED)               \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

// workaround for OS X Snow Leopard w/ gcc 4.2.1 and CUDA 2.3a
// (undefined __sync_fetch_and_add)
#if defined(__APPLE__)
# if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBC_ATOMIC_BUILTINS
#endif // __APPLE__

#include <stdio.h>
#include <cuda.h>
#include <hip/hip_runtime_api.h>

// On Windows, if we call exit, our console may disappear,
// taking the error message with it, so prompt before exiting.
#if defined(_WIN32)
#define safe_exit(val)                          \
{                                               \
    cout << "Press return to exit\n";           \
    cin.get();                                  \
    exit(val);                                  \
}
#else
#define safe_exit(val) exit(val)
#endif

#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    hipError_t err = hipGetLastError();                                     \
    if (err != hipSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,hipGetErrorString(err),__LINE__);            \
        safe_exit(-1);                                                        \
    }                                                                         \
}

// Alternative macro to catch CUDA errors
#define CUDA_SAFE_CALL( call) do {                                            \
   hipError_t err = call;                                                      \
   if (hipSuccess != err) {                                                  \
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
           __FILE__, __LINE__, hipGetErrorString( err) );                    \
       safe_exit(EXIT_FAILURE);                                               \
   }                                                                          \
} while (0)

// Alleviate aliasing issues
#define RESTRICT __restrict__

#endif // CUDACOMMON_H
