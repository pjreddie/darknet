GPU=1
DEBUG=0
ARCH= -arch=sm_35

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

CC=gcc
NVCC=nvcc
OPTS=-O3
LDFLAGS=`pkg-config --libs opencv` -lm -pthread -lstdc++
COMMON=`pkg-config --cflags opencv` -I/usr/local/cuda/include/
CFLAGS=-Wall -Wfatal-errors

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(GPU), 1) 
COMMON+=-DGPU
CFLAGS+=-DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o normalization_layer.o parser.o option_list.o darknet.o detection_layer.o imagenet.o captcha.o detection.o
ifeq ($(GPU), 1) 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)%.o: %.c 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu 
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

