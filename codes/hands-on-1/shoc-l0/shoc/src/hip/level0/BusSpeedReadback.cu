#include <stdio.h>
#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
//   -nopinned
//   This option controls whether page-locked or "pinned" memory is used.
//   The use of pinned memory typically results in higher bandwidth for data
//   transfer between host and device.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory", 'p');
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the host from the device and calculates the
//   bandwidth for each chunk size.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB,
                  OptionParser &op)
{
    bool verbose = op.getOptionBool("verbose");
    bool pinned  = !op.getOptionBool("nopinned");

    // Sizes are in kb
    int nSizes  = 20;
    int sizes[20] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
		     32768,65536,131072,262144,524288};
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    float *hostMem1;
    float *hostMem2;
    if (pinned)
    {
        hipHostMalloc((void**)&hostMem1, sizeof(float)*numMaxFloats);
        hipError_t err1 = hipGetLastError();
        hipHostMalloc((void**)&hostMem2, sizeof(float)*numMaxFloats);
        hipError_t err2 = hipGetLastError();
	while (err1 != hipSuccess || err2 != hipSuccess)
	{
	    // free the first buffer if only the second failed
	    if (err1 == hipSuccess)
	        hipHostFree((void*)hostMem1);

	    // drop the size and try again
	    if (verbose) cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
		cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            hipHostMalloc((void**)&hostMem1, sizeof(float)*numMaxFloats);
            err1 = hipGetLastError();
            hipHostMalloc((void**)&hostMem2, sizeof(float)*numMaxFloats);
            err2 = hipGetLastError();
	}
   }
    else
    {
        hostMem1 = new float[numMaxFloats];
        hostMem2 = new float[numMaxFloats];
    }
    for (int i=0; i<numMaxFloats; i++)
        hostMem1[i] = i % 77;

    float *device;
    hipMalloc((void**)&device, sizeof(float) * numMaxFloats);
    while (hipGetLastError() != hipSuccess)
    {
	// drop the size and try again
	if (verbose) cout << " - dropping size allocating device mem\n";
	--nSizes;
	if (nSizes < 1)
	{
	    cerr << "Error: Couldn't allocated any device buffer\n";
	    return;
	}
	numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
        hipMalloc((void**)&device, sizeof(float) * numMaxFloats);
    }

    hipMemcpy(device, hostMem1,
               numMaxFloats*sizeof(float), hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    const unsigned int passes = op.getOptionInt("passes");

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    CHECK_CUDA_ERROR();

    // Three passes, forward and backward both
    for (int pass = 0; pass < passes; pass++)
    {
        // store the times temporarily to estimate latency
        //float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
        for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            if ((pass % 2) == 0)
                sizeIndex = i;
            else
                sizeIndex = (nSizes - 1) - i;

            int nbytes = sizes[sizeIndex] * 1024;

            hipEventRecord(start, 0);
            hipMemcpy(hostMem2, device,
                       nbytes, hipMemcpyDeviceToHost);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            float t = 0;
            hipEventElapsedTime(&t, start, stop);
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (verbose)
            {
                cerr << "size " <<sizes[sizeIndex] << "k took " << t <<
                        " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);
            resultDB.AddResult("ReadbackSpeed", sizeStr, "GB/sec", speed);
            resultDB.AddResult("ReadbackTime", sizeStr, "ms", t);
        }
	//resultDB.AddResult("ReadbackLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	//resultDB.AddResult("ReadbackLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	//resultDB.AddResult("ReadbackLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
    }

    // Cleanup
    hipFree((void*)device);
    CHECK_CUDA_ERROR();
    if (pinned)
    {
        hipHostFree((void*)hostMem1);
        CHECK_CUDA_ERROR();
        hipHostFree((void*)hostMem2);
        CHECK_CUDA_ERROR();
    }
    else
    {
        delete[] hostMem1;
        delete[] hostMem2;
        hipEventDestroy(start);
	    hipEventDestroy(stop);
    }
}
