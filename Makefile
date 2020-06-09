CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_61,code=compute_61
        
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS       := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
	CCFLAGS   	  := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS   := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32
else
	NVCCFLAGS := -m64
endif

TARGETS = build/slicer build/slicer_no_cairo

$(shell mkdir -p build)
$(shell mkdir -p img_cpu)
$(shell mkdir -p img_gpu)
$(shell mkdir -p obj)

all: $(TARGETS)

build/slicer: src/Slicer.cpp src/main.cpp obj/transpose.o
	$(CC) $^ -o $@ -O3 -Iinclude -DUSE_CAIRO=1 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -lcairo -pthread


build/slicer_no_cairo: src/Slicer.cpp src/main.cpp obj/Slicer.o
	$(CC) $^ -o $@ -O3 -Iinclude -DUSE_CAIRO=0 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -pthread

obj/Slicer.o: src/Slicer.cu
	$(NVCC) $(NVCCFLAGS) -O3 -Iinclude $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
