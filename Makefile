GPU=1
CLBLAS=0

CC=gcc
COMMON=-Wall -Wfatal-errors `pkg-config --cflags opencv` -I/usr/local/cuda/include/
ifeq ($(GPU), 1) 
COMMON+=-DGPU
endif

ifeq ($(CLBLAS), 1) 
COMMON+=-DCLBLAS
LDFLAGS=-lclBLAS
endif

UNAME = $(shell uname)
OPTS=-Ofast -flto
#OPTS=-O3
ifeq ($(UNAME), Darwin)
COMMON+= -isystem /usr/local/Cellar/opencv/2.4.6.1/include/opencv -isystem /usr/local/Cellar/opencv/2.4.6.1/include
ifeq ($(GPU), 1)
LDFLAGS= -framework OpenCL
endif
else
OPTS+= -march=native
ifeq ($(GPU), 1)
LDFLAGS+= -lOpenCL
endif
endif
CFLAGS= $(COMMON) $(OPTS)
#CFLAGS= $(COMMON) -O0 -g
LDFLAGS+=`pkg-config --libs opencv` -lm
VPATH=./src/
EXEC=cnn
OBJDIR=./obj/

OBJ=network.o network_gpu.o image.o cnn.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o data.o matrix.o softmax_layer.o mini_blas.o convolutional_layer.o gemm.o normalization_layer.o opencl.o im2col.o col2im.o axpy.o dropout_layer.o crop_layer.o freeweight_layer.o cost_layer.o
OBJS = $(addprefix $(OBJDIR), $(OBJ))

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

