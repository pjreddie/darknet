GPU=1
DEBUG=0

VPATH=./src/
EXEC=cnn
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-O3
LINKER=$(CC)
LDFLAGS=`pkg-config --libs opencv` -lm -pthread
COMMON=`pkg-config --cflags opencv` -I/usr/local/cuda/include/
CFLAGS=-Wall -Wfatal-errors
CFLAGS+=$(OPTS)

ifeq ($(DEBUG), 1) 
COMMON+=-O0 -g
CFLAGS+=-O0 -g
endif

ifeq ($(GPU), 1) 
LINKER=$(NVCC)
COMMON+=-DGPU
CFLAGS+=-DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
endif

#OBJ=network.o network_gpu.o image.o cnn.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o data.o matrix.o softmax_layer.o convolutional_layer.o gemm.o normalization_layer.o opencl.o im2col.o col2im.o axpy.o dropout_layer.o crop_layer.o freeweight_layer.o cost_layer.o server.o
OBJ=gemm.o utils.o cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o normalization_layer.o parser.o option_list.o cnn.o
ifeq ($(GPU), 1) 
OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)%.o: %.c 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu 
	$(NVCC) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

