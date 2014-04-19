CC=gcc
GPU=0
COMMON=-Wall `pkg-config --cflags opencv` -I/usr/local/cuda/include/
UNAME = $(shell uname)
OPTS=-O3
ifeq ($(UNAME), Darwin)
COMMON+= -isystem /usr/local/Cellar/opencv/2.4.6.1/include/opencv -isystem /usr/local/Cellar/opencv/2.4.6.1/include
ifeq ($(GPU), 1)
LDFLAGS= -framework OpenCL
endif
else
OPTS+= -march=native
ifeq ($(GPU), 1)
LDFLAGS= -lOpenCL
endif
endif
CFLAGS= $(COMMON) $(OPTS)
#CFLAGS= $(COMMON) -O0 -g 
LDFLAGS+=`pkg-config --libs opencv` -lm
VPATH=./src/
EXEC=cnn

OBJ=network.o image.o tests.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o data.o matrix.o softmax_layer.o mini_blas.o convolutional_layer.o cpu_gemm.o normalization_layer.o
ifeq ($(GPU), 1)
OBJ+=gpu_gemm.o opencl.o 
endif

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJ) $(EXEC)

