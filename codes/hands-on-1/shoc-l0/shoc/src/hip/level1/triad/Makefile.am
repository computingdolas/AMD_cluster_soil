if BUILD_MPI
    MAYBE_MPI = epmpi
endif
SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(HIP_LDFLAGS)
AM_CPPFLAGS = $(HIP_INC)

# What is the destination for programs built from this directory?
serialhipdir = $(bindir)/Serial/hip

# What programs should be installed to that destination?
serialhip_PROGRAMS = Triad

CXXLD=$(HIPCC)
# How to build those programs?
Triad_SOURCES = main.cpp
Triad_LDADD = Triad.o $(HIP_LIBS) $(LIBS)

