# Porting CUDA to HIP

For porting applications from cuda to hip there are multiple
ways of making it work. This mostly depends on the
scale of the application what the fastest way is.
For simple C/C++ projects with a few kernels running
hipconvertinplace-perl.sh is often enough. This
will use a simple search replace changing the cuda
calls to HIP. 

If there is a complicated build or includes needed running with hipify-clang
as the compiler in the build fase will replace all the relevent files using the
clang frontend to parse and replace the more complicated calls.

For large projects it is often the easiest to make a translation file
with defines that mapp the used cuda calls to HIP, and only ifdef or change
the kernel calls if it is compiled with HIP, this resulsts in the least amount
of codechange, and enables a "slow" inplace rewrite to HIP. 
