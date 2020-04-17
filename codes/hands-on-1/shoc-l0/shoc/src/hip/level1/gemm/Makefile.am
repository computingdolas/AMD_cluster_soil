if BUILD_MPI
    MAYBE_MPI = epmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(HIP_LDFLAGS) -L/opt/rocm/hipblas/lib -lhipblas
AM_CPPFLAGS = $(HIP_INC) -I/opt/rocm/hipblas/include

# What is the destination for programs built from this directory?
serialhipdir = $(bindir)/Serial/HIP

# What programs should be installed to that destination?
serialhip_PROGRAMS = GEMM

CXXLD=$(HIPCC)

# How to build those programs?
GEMM_SOURCES = GEMM.cpp \
                main.cpp
GEMM_LDADD = $(HIP_LIBS) $(LIBS)

