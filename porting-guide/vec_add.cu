#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

#define ERRCHECK {                                                            \
    hipError_t err;                                                       \
    if ((err = hipGetLastError()) != hipSuccess) {                       \
        std::cout << "CUDA error: " << hipGetErrorString(err) << " : "    \
                  << __FILE__ << ", line " << __LINE__ << std::endl;       \
        exit(1);                                                           \
    }                                                                      \
}

#define TIMERSTART(label)                                                  \
    hipSetDevice(0);                                                      \
    hipEvent_t start##label, stop##label;                                 \
    float time##label;                                                     \
    hipEventCreate(&start##label);                                        \
    hipEventCreate(&stop##label);                                         \
    hipEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                   \
        hipSetDevice(0);                                                  \
        hipEventRecord(stop##label, 0);                                   \
        hipEventSynchronize(stop##label);                                 \
        hipEventElapsedTime(&time##label, start##label, stop##label);     \
        std::cout << "TIMING: " << time##label << " ms (" << #label << ")" \
                  << std::endl;



__global__ void vectorAddKernel(float* deviceA, float* deviceB, float* deviceC) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    deviceC[index] = deviceA[index] + deviceB[index];
}

__global__ void emptyKernel(float* deviceA, float* deviceB, float* deviceC) {}


void vectorAddCuda(int n, float* hostA, float* hostB, float* hostC) {
    int threadBlockSize = 512;

    /* Allocate the vectors on the GPU. */
    float *deviceA, *deviceB, *deviceC;

    hipMalloc((void **) &deviceA, n * sizeof(float));                                      ERRCHECK
    hipMalloc((void **) &deviceB, n * sizeof(float));                                      ERRCHECK
    hipMalloc((void **) &deviceC, n * sizeof(float));                                      ERRCHECK

    /* Copy the original vectors to the GPU. */
    TIMERSTART(host_to_device)
    hipMemcpy(deviceA, hostA, n*sizeof(float), hipMemcpyHostToDevice);                    ERRCHECK
    hipMemcpy(deviceB, hostB, n*sizeof(float), hipMemcpyHostToDevice);                    ERRCHECK
    TIMERSTOP(host_to_device)

    hipLaunchKernelGGL((emptyKernel), dim3(1), dim3(1), 0, 0, deviceA, deviceB, deviceC);                                       ERRCHECK

    /* Execute and time the kernel */
    TIMERSTART(kernel)
    hipLaunchKernelGGL((vectorAddKernel), dim3(n/threadBlockSize), dim3(threadBlockSize), 0, 0, deviceA, deviceB, deviceC);     ERRCHECK
    TIMERSTOP(kernel)

    TIMERSTART(device_to_host)
    /* Copy back results */
    hipMemcpy(hostC, deviceC, n * sizeof(float), hipMemcpyDeviceToHost);                  ERRCHECK
    TIMERSTOP(device_to_host)

    hipFree(deviceA);                                                                      ERRCHECK
    hipFree(deviceB);                                                                      ERRCHECK
    hipFree(deviceC);                                                                      ERRCHECK
}


int main(int argc, char* argv[]) {
    int n = 65536000;
    float* hostA = new float[n];
    float* hostB = new float[n];
    float* hostC = new float[n];

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device Number: %d\n", 0);
    printf("  Device name: %s\n\n", prop.name);

    /* initialize the vectors. */
    for(int i = 0; i < n; i++) {
        hostA[i] = i;
        hostB[i] = i;
    }

    vectorAddCuda(n, hostA, hostB, hostC);

    /* verify the resuls. */
    for(int i=0; i<n; i++) {
        if(hostC[i] != 2*i) {
            cout << "error in results! Element " << i << " is " << hostC[i] << ", but should be " << (2*i) << endl;
            exit(0);
        }
    }

    cout << "results OK!" << endl;

    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
}
