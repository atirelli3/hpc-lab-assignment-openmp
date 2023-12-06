-include ../../../utilities/options.mk
-include ../../../utilities/c2.mk

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

OPT:=-O2 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math

CXXFLAGS:=$(OPT) -I. $(EXT_CXXFLAGS) ../../../../common/polybench.h
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp

SRCS:= utils.c polybench.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(EXERCISE:%=$(BUILD_DIR)/%.o)
EXE=$(EXERCISE:.cu=.exe)

$(EXE): $(OBJS)
        $(MKDIR_P) $(dir $@)
        $(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
        $(MKDIR_P) $(dir $@)
        $(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
        $(MKDIR_P) $(dir $@)
        $(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
        $(MKDIR_P) $(dir $@)
        $(CXX) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

.PHONY: run profile clean
run: $(EXE)
        ./$(EXE)

profile: $(EXE)
        sudo LD_LIBRARY_PATH=$(CUDA_HOME)/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} $(CUDA_HOME)/bin/nvprof ./$(EXE)

clean:
        -rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p