CC=gcc
GPU=0
COMMON=-Wall -Werror -Wfatal-errors `pkg-config --cflags opencv` -I/usr/local/cuda/include/
ifeq ($(GPU), 1) 
COMMON+=-DGPU
else
endif
UNAME = $(shell uname)
OPTS=-Ofast -flto
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

OBJ=network.o image.o tests.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o data.o matrix.o softmax_layer.o mini_blas.o convolutional_layer.o gemm.o normalization_layer.o opencl.o im2col.o col2im.o axpy.o

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJ) $(EXEC)

