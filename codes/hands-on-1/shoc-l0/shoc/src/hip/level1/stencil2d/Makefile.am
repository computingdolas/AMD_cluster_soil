if BUILD_MPI
    MAYBE_MPI = tpmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(HIP_LDFLAGS) -L/usr/lib/openmpi/lib
AM_CPPFLAGS = $(HIP_INC) -I/usr/lib/openmpi/include

# What is the destination for programs built from this directory?
serialhipdir = $(bindir)/Serial/HIP

# What programs should be installed to that destination?
serialhip_PROGRAMS = Stencil2D

CXXLD=$(HIPCC)
# How to build those programs?
Stencil2D_SOURCES = CUDAStencil.cpp \
				CommonCUDAStencilFactory.cpp \
				Stencil2Dmain.cpp \
				CUDAStencilFactory.cpp \
                main.cpp
Stencil2D_LDADD = CUDAStencilKernel.o $(HIP_LIBS) $(LIBS)

