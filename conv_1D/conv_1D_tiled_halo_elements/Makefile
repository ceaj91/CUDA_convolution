# Makefile for CUDA hello_world.cu

# CUDA compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDA_FLAGS := -arch=sm_35

# Source file
SRC := conv_1D_tiled_halo_elements.cu

# Executable name
EXECUTABLE := conv_1D_tiled_halo_elements

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -o $@ $<

clean:
	rm -f $(EXECUTABLE)

