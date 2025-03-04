#ifndef PMSMEMMGMT_H
#define PMSMEMMGMT_H

#include <stdlib.h>
#include "cudacommon.h"
#include <cuda.h>
#include <hip/hip_runtime_api.h>

// Programming Model-Specific Memory Management
// Some programming models for heterogeneous systems provide
// memory management functions for allocating memory on the host
// and on the device.  These functions provide an abstract interface
// to that programming-model-specific interface.

template<class T>
T*
pmsAllocHostBuffer( size_t nItems )
{
    T* ret = NULL;
    size_t nBytes = nItems * sizeof(T);
    CUDA_SAFE_CALL(hipHostMalloc(&ret, nBytes));
    return ret;
}


template<class T>
void
pmsFreeHostBuffer( T* buf )
{
    CUDA_SAFE_CALL(hipHostFree(buf));
}

#endif // PMSMEMMGMT_H
