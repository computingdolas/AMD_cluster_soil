#include "hip/hip_runtime.h"
#include "reduction_kernel.h"

template<class T>
void
RunTestLaunchKernel( int num_blocks,
                    int num_threads,
                    int smem_size,
                    T* d_idata,
                    T* d_odata,
                    int size )
{
    // In CUDA 4.0 we will be able to remove this level of indirection
    // if we use the cuConfigureCall and cuLaunchKernel functions.
    hipLaunchKernelGGL((reduce<T,256>), dim3(num_blocks), dim3(num_threads), smem_size, 0, d_idata, d_odata, size);
}


// ensure that the template functions are instantiated
// Unlike the Stencil2D CUDA version that needs to instantiate objects,
// we need to instantiate template functions.  Declaration of the needed
// specializations seem to work for several recent versions of g++ that
// people are likely to be using underneath nvcc.
template void RunTestLaunchKernel<float>( int, int, int, float*, float*, int );
template void RunTestLaunchKernel<double>( int, int, int, double*, double*, int );

