# Multi GPU Jacobi using MPI
Jacobi-cuda contains a multi GPU Jacobi example taken from [https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts/cuda-aware-mpi-example/src], the program is described in [this][https://devblogs.nvidia.com/benchmarking-cuda-aware-mpi/] blog post.
Jacobi-hip is a working hipified version of the jacobi program.

## How to port
1. Take the jacobi-cuda code
2. Use hipconvertinplace to automaticly hipify all the source files to HIP.
    For this step you can either use `hipconvertinplace-perl.sh` or `hipconvertinplace.sh`.
    When using the perl script some warnings will be generated, these can be ignored.
    Using the clang version, i.e., `hipconvertinplace.sh`, will not generated errors.
    However in the file Device.cu you will need to alter 2 lines. On lines 315 and 314 the last argument of the kernel launce, i.e., HasNeighbor, needs to be replaced with HasNeighbor(neighbors, DIR_RIGHT)
3. Replace the makefile with the the one in this folder.
    The Makefile in this folder is an edited version of the original Makefile.
    Main difference are, it uses hipcc instead of nvcc and uses hipcc to link the files instead of mpicc. 

## How to run
1. Make sure the environment variable `MPI_HOME` is set to the correct MPI installations, for kleurplaat `MPI_HOME=...`
2. run `make`
3. run `mpiexec -np 2 bin/jacobi_cuda_normal_mpi -t 2 1 -fs`
    -t x y sets the array decomposition. x * y should be equal to the nummber of mpi processes running.
    -fs uses fast pointer swap instead of full block copy.
    Run `bin/jacobi_cuda_normal_mpi -h` for the full list of arguments.

To run on specific GPU's use `HIP_VISIBLE_DEVICES=1,2`, e.g, `HIP_VISIBLE_DEVICES=0,1 mpiexec -np 2 bin/jacobi_cuda_normal_mpi -t 2 1 -fs` runs on the first 2 GPU's.
For the Kleurplaat machine GPU 0, 1, 2, 3 are on the same islands as well as GPU 4, 5, 6, 7.
