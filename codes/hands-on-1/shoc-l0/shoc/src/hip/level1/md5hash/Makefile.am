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
serialhipdir = $(bindir)/Serial/HIP

# What programs should be installed to that destination?
serialhip_PROGRAMS = MD5Hash

CXXLD=$(HIPCC)

# How to build those programs?
MD5Hash_SOURCES = main.cpp
MD5Hash_LDADD = MD5Hash.o $(HIP_LIBS) $(LIBS)
