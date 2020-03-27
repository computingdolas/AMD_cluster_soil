#!/bin/sh

# A "good" set of optimization flags is compiler dependent.
# These might be reasonable flags to start from.
#
# GNU
OPTFLAGS="-O2 -g"

# Intel
#OPTFLAGS="-g -xHOST -O3 -ip -no-prec-div"
#export CXX=icpc
#export CC=icc

# PGI
#OPTFLAGS="-g -fastsse"
#export CXX=pgcpp
#export CC=pgcc

sh ./configure \
    CUDA_CPPFLAGS="-gencode=arch=compute_61,code=sm_61" \
    CXXFLAGS="$OPTFLAGS" \
    CFLAGS="$OPTFLAGS" \
    LDFLAGS="" \
    HIP_INCDIR=". $(hipconfig --cpp_config) -I/usr/local/cuda/include" \
    --with-hip
