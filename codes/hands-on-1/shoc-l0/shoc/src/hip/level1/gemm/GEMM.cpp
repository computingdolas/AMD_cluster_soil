#include <iostream>
#include <sstream>
#include <string>
//#include "cuda.h"
#include "cudacommon.h"
//#include "cublas.h"
#include "hipblas.h"
#include <hip/hip_runtime.h>
#include "Timer.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

#ifndef _WIN32
#include <sys/time.h>
#endif

using namespace std;

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

template <class T>
inline void devGEMM(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, T *alpha,
        const T *A, int lda, const T *B, int ldb, T *beta, T *C, int ldc);

// ********************************************************
// Function: toString
//
// Purpose:
//   Simple templated function to convert objects into
//   strings using stringstream
//
// Arguments:
//   t: the object to convert to a string
//
// Returns:  a string representation of t
//
// Modifications:
//
// ********************************************************
template<class T> inline std::string toString(const T& t)
{
    stringstream ss;
    ss << t;
    return ss.str();
}

// ********************************************************
// Function: error
//
// Purpose:
//   Simple routine to print an error message and exit
//
// Arguments:
//   message: an error message to print before exiting
//
// ********************************************************
void error(char *message)
{
    cerr << "ERROR: " << message << endl;
    exit(1);
}

// ********************************************************
// Function: fill
//
// Purpose:
//   Simple routine to initialize input array
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//
// ********************************************************
template <class T>
void fill(T *A, int n, int maxi)
{
    for (int j = 0; j < n; j++)
        A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / (maxi + 1.);
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("KiB", OPT_INT, "0", "data size (in Kibibytes)");
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of the single precision general
//   matrix multiplication (SGEMM) operation in GFLOPS.  Data transfer time
//   over the PCIe bus is not included in this measurement.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, device);

    cout << "Running single precision test" << endl;
    RunTest<float>("SGEMM", resultDB, op);

    // Test to see if this device supports double precision
    if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
               (deviceProp.major >= 2))
    {
        cout << "Running double precision test" << endl;
        RunTest<double>("DGEMM", resultDB, op);
    } else {
        cout << "Skipping double precision test" << endl;
        char atts[1024] = "DP_Not_Supported";
        // resultDB requires neg entry for every possible result
        int passes = op.getOptionInt("passes");
        for (; passes > 0; --passes) {
            for (int i = 0; i < 2; i++) {
                const char transb = i ? 'T' : 'N';
                string testName="DGEMM";
                resultDB.AddResult(testName+"-"+transb, atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", atts, "GFlops", FLT_MAX);
                resultDB.AddResult(testName+"-"+transb+"_Parity", atts, "N", FLT_MAX);
            }
        }
    }
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op)
{
    int passes = op.getOptionInt("passes");
    int N;
    if (op.getOptionInt("KiB") == 0)
    {
        int probSizes[4] = { 1, 4, 8, 16 };
        N = probSizes[op.getOptionInt("size")-1] * 1024 / sizeof(T);
    } else {
        N = op.getOptionInt("KiB") * 1024 / sizeof(T);
    }

    // Initialize the cublas library
    //cublasInit();
    //hipblasCreate();
    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    // Allocate GPU memory
    T *dA, *dB, *dC;
    CUDA_SAFE_CALL(hipMalloc(&dA, N * N * sizeof(T)));
    CUDA_SAFE_CALL(hipMalloc(&dB, N * N * sizeof(T)));
    CUDA_SAFE_CALL(hipMalloc(&dC, N * N * sizeof(T)));

    // Initialize host memory
    T *A;
    T *B;
    T *C;

    CUDA_SAFE_CALL(hipHostMalloc(&A, N * N * sizeof(T)));
    CUDA_SAFE_CALL(hipHostMalloc(&B, N * N * sizeof(T)));
    CUDA_SAFE_CALL(hipHostMalloc(&C, N * N * sizeof(T)));

    fill<T>(A, N * N, 31);
    fill<T>(B, N * N, 31);
    fill<T>(C, N * N, 31);

    // Copy input to GPU
    hipEvent_t start, stop;
    CUDA_SAFE_CALL(hipEventCreate(&start));
    CUDA_SAFE_CALL(hipEventCreate(&stop));
    CUDA_SAFE_CALL(hipEventRecord(start, 0));
    CUDA_SAFE_CALL(hipMemcpy(dA, A, N * N * sizeof(T),
            hipMemcpyHostToDevice));
    CUDA_SAFE_CALL(hipMemcpy(dB, B, N * N * sizeof(T),
            hipMemcpyHostToDevice));
    hipEventRecord(stop, 0);
    CUDA_SAFE_CALL(hipEventSynchronize(stop));

    // Get elapsed time
    float transferTime = 0.0f;
    hipEventElapsedTime(&transferTime, start, stop);
    transferTime *= 1.e-3;

    bool first = true;
    for (; passes > 0; --passes)
    {
        for (int i = 0; i < 2; i++)
        {
            hipblasOperation_t transa = HIPBLAS_OP_N;
            hipblasOperation_t transb = i ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            const int nb = 128;
            const int idim = N / nb;

            int dim = idim * nb;

            int m = dim;
            int n = dim;
            int k = dim;
            int lda = dim;
            int ldb = dim;
            int ldc = dim;
            T alpha = 1;
            T beta = 0;//-1;

            // Warm Up
            devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta,
                    dC, ldc);
            CUDA_SAFE_CALL(hipDeviceSynchronize());

            double blas_time;
            float kernel_time = 0.0f;
            for (int ii = 0; ii < 4; ++ii)
            {
                CUDA_SAFE_CALL(hipEventRecord(start, 0));
                devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb,
                        &beta, dC, ldc);
                CHECK_CUDA_ERROR();
                hipEventRecord(stop, 0);
                CUDA_SAFE_CALL(hipEventSynchronize(stop));
                float currTime = 0.0f;
                hipEventElapsedTime(&currTime, start, stop);
                kernel_time += currTime;
            }
            blas_time = (kernel_time / 4.0) * 1.e-3;

            CUDA_SAFE_CALL(hipEventRecord(start, 0));
            CUDA_SAFE_CALL(hipMemcpy(C, dC, N * N * sizeof(float),
                    hipMemcpyDeviceToHost));
            hipEventRecord(stop, 0);
            CUDA_SAFE_CALL(hipEventSynchronize(stop));

            float oTransferTime = 0.0f;
            hipEventElapsedTime(&oTransferTime, start, stop);
            oTransferTime *= 1.e-3;

            // Add the PCIe transfer time to total transfer time only once
            if (first)
            {
                transferTime += oTransferTime;
                first = false;
            }

            {
                const char transa = 'N';
                const char transb = i ? 'T' : 'N';
                double blas_glops = 2. * m * n * k / blas_time / 1e9;
                double pcie_gflops = 2. * m * n * k / (blas_time + transferTime)
                        / 1e9;
                resultDB.AddResult(testName+"-"+transb, toString(dim), "GFlops",
                        blas_glops);
                resultDB.AddResult(testName+"-"+transb+"_PCIe", toString(dim),
                        "GFlops", pcie_gflops);
                resultDB.AddResult(testName+"-"+transb+"_Parity", toString(dim),
                        "N", transferTime / blas_time);
            }
        }
    }

    // Clean Up
    CUDA_SAFE_CALL(hipFree(dA));
    CUDA_SAFE_CALL(hipFree(dB));
    CUDA_SAFE_CALL(hipFree(dC));
    CUDA_SAFE_CALL(hipHostFree(A));
    CUDA_SAFE_CALL(hipHostFree(B));
    CUDA_SAFE_CALL(hipHostFree(C));
    CUDA_SAFE_CALL(hipEventDestroy(start));
    CUDA_SAFE_CALL(hipEventDestroy(stop));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));

}

template<>
inline void devGEMM<double>(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
        double *alpha, const double *A, int lda, const double *B, int ldb,
        double *beta, double *C, int ldc) {
    hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void devGEMM<float>(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k,
        float *alpha, const float *A, int lda, const float *B, int ldb,
        float *beta, float *C, int ldc) {
    hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
