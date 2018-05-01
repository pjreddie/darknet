
#
# Default make builds both original darknet, and its CPP equivalent darknet-cpp
# make darknet - only darknet (original code), OPENCV=0
# make darknet-cpp - only the CPP version, OPENCV=1
# make darknet-cpp-shared - build the shared-lib version (without darknet.c calling wrapper), OPENCV=1
# 
# CPP version supports OpenCV3. Tested on Ubuntu 16.04
#
# OPENCV=1 (C++ && CV3, or C && CV2 only - check with pkg-config --modversion opencv)
# When building CV3 and C version, will get errors like
# ./obj/image.o: In function `cvPointFrom32f':
# /usr/local/include/opencv2/core/types_c.h:929: undefined reference to `cvRound'
#
# 

GPU=1
CUDNN=1
OPENCV=1

OPENMP=0
DEBUG=0
CUDA_MEM_DEBUG=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify

# ARCH=  -gencode arch=compute_52,code=compute_52

# C Definitions

# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/
CC=gcc


# C++ Definitions
EXEC_CPP=darknet-cpp
SHARED_CPP=darknet-cpp-shared
OBJDIR_CPP=./obj-cpp/
OBJDIR_CPP_SHARED=./obj-cpp-shared/
CC_CPP=g++
CFLAGS_CPP=-Wno-write-strings -std=c++0x

NVCC=nvcc
AR=ar
ARFLAGS=rcs

OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif


ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

# Place the IPP .a file from OpenCV here for easy linking
LDFLAGS += -L./3rdparty

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif


ifeq ($(CUDA_MEM_DEBUG), 1)
CFLAGS_CPP+= -D_ENABLE_CUDA_MEM_DEBUG
endif

OBJ-SHARED=segmenter.o upsample_layer.o l2norm_layer.o logistic_layer.o lstm_layer.o gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o captcha.o route_layer.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o yolo.o detector.o layer.o regressor.o classifier.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o  crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o lsd.o super.o tree.o yolo_layer.o  

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ-GPU=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
OBJ-SHARED+=$(OBJ-GPU)
endif

OBJ=$(OBJ-SHARED) darknet.o

EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o darknet.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)


OBJS_CPP = $(addprefix $(OBJDIR_CPP), $(OBJ))
OBJS_CPP_SHARED = $(addprefix $(OBJDIR_CPP_SHARED), $(OBJ-SHARED))

all: backup obj obj-cpp results $(SLIB) $(ALIB) $(EXEC) $(EXEC_CPP)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)


$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(EXEC_CPP): obj-cpp clean-cpp $(OBJS_CPP)
	$(CC_CPP) $(COMMON) $(CFLAGS) $(OBJS_CPP) -o $@ $(LDFLAGS)
$(SHARED_CPP): obj-shared-cpp clean-cpp $(OBJS_CPP_SHARED)
	$(CC_CPP) $(COMMON) $(CFLAGS) $(OBJS_CPP_SHARED) -o lib$@.so $(LDFLAGS) -shared	

$(OBJDIR_CPP)%.o: %.c $(DEPS)
	$(CC_CPP) $(COMMON) $(CFLAGS_CPP) $(CFLAGS) -c $< -o $@
$(OBJDIR_CPP_SHARED)%.o: %.c $(DEPS)
	$(CC_CPP) $(COMMON) $(CFLAGS_CPP) $(CFLAGS) -fPIC -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@
$(OBJDIR_CPP)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@
$(OBJDIR_CPP_SHARED)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS) -fPIC" -c $< -o $@

obj:
	mkdir -p obj
obj-cpp:
	mkdir -p obj-cpp
obj-shared-cpp:
	mkdir -p obj-cpp-shared

backup:
	mkdir -p backup

results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)
clean-cpp:
	rm -rf $(OBJS_CPP) $(OBJS_CPP_SHARED) $(EXEC_CPP) $(SHARED_CPP)
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*


