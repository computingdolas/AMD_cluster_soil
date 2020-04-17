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
serialhip_PROGRAMS = NeuralNet

CXXLD=$(HIPCC)
# How to build those programs?
NeuralNet_SOURCES = main.cpp
NeuralNet_LDADD = NeuralNet.o $(HIP_LIBS) $(LIBS) -lcublas

# Unzip and move data to the bin directory, if supported
if DATA_UNZIP
#Adding these lines would cause automake to execute these lines
#with install-data flags but this requires listing each file
#separately; instead we use a custom command below
#data_DATA = nn_data
#nn_data: $(srcdir)/nn_data.zip
#	$(UNZIP) -o $(srcdir)/nn_data.zip
# cp -prf nn_data $(bindir)/.

#This overrides the default install-data command
install-data-local:
	$(UNZIP) -o $(srcdir)/nn_data.zip
	cp -prf nn_data $(bindir)/.
endif

clean: 
	rm -rf nn_data NeuralNet
