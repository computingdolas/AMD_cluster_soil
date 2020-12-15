# HIP porting guide

HIP is a C++ Runtime API and kernel language developed by AMD. It enables developers to execute the same source code on both AMD and NVIDIA GPUs. HIP also provides a set of tools that can be used to automatically port CUDA code to HIP.

A list of the provided tools and there usage can be found [here](https://github.com/ROCm-Developer-Tools/HIP#tour-of-the-hip-directories).
For a more elaborate guide we refer to [this guide provided by AMD](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md)

## Basic Porting

This guide is specifically focused for porting CUDA to HIP on the kleurplaat-01 machine, however, most of the information provided is not bound to this machine. 

### Environment setup

On the kleurplaat-01 machine all ROCm and HIP tools can be found in */opt/rocm-3.5.1/bin*.
For easy access we recommend adding this folder to your path by adding the following line to your .bashrc

```bash
PATH=$PATH:/opt/rocm-3.5.1/bin/
```
For the rest of the guide we will assume that this folder is added to your path, if this is not the case you will have to call the tools using the full path. To check if you are able to reach the tools you can attempt to run hipconfig to print the configuration of the hip installation.

```bash
$ hipconfig -f
```

### Porting CUDA application to HIP

The [hipify-toolchain](https://github.com/ROCm-Developer-Tools/HIP#tour-of-the-hip-directories) provides a small set of tool to use when porting CUDA code to HIP. For simple porting we only need **hipconvertinplace-perl.sh**. This script uses pearl search and replace all CUDA code with HIP code in the provided file or directory.  Using this tool is as simple as running 

```bash
$ cd /path/to/your/cuda_code/
$ hipconvertinplace-perl.sh
```

Running this tool will perform two actions:
1. Make a copy of all code before porting. These copies are stored as filename.extention.prehip.
2. Search and replace all CUDA calls to HIP calls in the original files.

The output will provide information on the porting, such as, warnings, number of lines of code changed, etc. If all went well this should have ported all your code to HIP. 

#### Some Remarks

- AMD also provides hipconvertinplace.sh which uses hipify-clang to port the code. The clang version, if setup properly, should be able to port more complicated pieces of code, because it parses the code instead of doing a search and replace.
- hipexamine-perl.sh and hipexamine-perl.sh show you the porting output without changing the code. This can be useful to see if the code can be ported without any issues.
- If there are already .prehip files present, the hipify tools will port the .prehip files and overwrite the existing code.

## Compiling

Compiling HIP code can be done in one of two ways. Firstly we can compile code as normal but use ***Hipcc*** instead of ***nvcc***. Alternately because HIP is a c++ runtime library, we can use any c++ compiler to compile the code. To do this we need to provided the compiler with some includes and defines, to get this extra information we can use the hipconfig tool.

```bash
$ hipconfig --cpp_config
```

Running this command will output the compiler options required to compile HIP code using any compatible c++ compiler.

Most codes will have some form of build system. We can use the following lines in a makefile to compile HIP code using the gcc compiler.

```makefile
HIP_INCLUDE=$(shell hipconfig --cpp_config)

all: test.cpp
	g++ ${HIP_INCLUDE} test.cpp
```

#### Some Remarks
- AMD recomends using the hipcc compiler

- We can use the HIP_PLATFORM environment value to tell hipcc which backed to target. If we set this to nvcc the compiled code will be for a CUDA device

  ```bash
  $ export HIP_PLATFORM=hcc # Target AMD backend
  $ export HIP_PLATFORM=nvcc # Target CUDA backend
  ```

## More complicated porting

### Library porting

If your code uses a more popular CUDA library there is change that there is a ROCm equivalent library. A list of equivalent libraries can be found here: [Library Equivalants](https://github.com/ROCm-Developer-Tools/HIP/blob/main/docs/markdown/hip_porting_guide.md#library-equivalents).

### Platform specific optimizations

If you want to add platform specific optimizations while still keeping the code portable you can use the defines to detect the targeted platform at compile time. see: [This Link](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#distinguishing-compiler-modes)

### Porting without code alterations

If you have a large code base it might preferable if you don't have to alter the code, by using the hipify tools. Because HIP and CUDA are similar in syntax, and hipifying is only a glorified search and replace, you could add a file that redefines CUDA calls into HIP calls and include this only when you want to compile using the HIP tools. Only the kernel launch has to be altered differently, because the CUDA kernel launch is not valid c++ code. This can be achieved as follows:

```c++
#if defined(__HIPCC__)
	hipLaunchKernelGGL((vectorAddKernel), dim3(n/threadBlockSize), dim3(threadBlockSize), 0 /*Shared*/, 0/*stream*/, deviceA, deviceB, deviceC);
#else
	vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceC);
#endif
```




## Links

- https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md
- https://github.com/ROCm-Developer-Tools/HIP
- https://rocmdocs.amd.com/en/latest/ROCm_API_References/HIP-API.html#hip-api
- https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-terminology.html#hip-terminology
- https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP_Debugging.html#hip-debugging
- https://rocmdocs.amd.com/en/latest/Programming_Guides/Kernel_language.html#kernel-language
- https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html