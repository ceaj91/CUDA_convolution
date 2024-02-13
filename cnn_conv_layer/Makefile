# Makefile for CUDA hello_world.cu

# CUDA compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDA_FLAGS := -arch=sm_35

# Source file
SRC := cnn_conv_layer.cu

# Executable name
EXECUTABLE := cnn_conv_layer

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(NVCC) $(CFLAGS) $(CUDA_FLAGS) -o $@ $<

clean:
	rm -f $(EXECUTABLE)

