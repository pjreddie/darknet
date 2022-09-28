cmake_minimum_required(VERSION 2.8)
project(darknet)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

# TODO: Add CUDA support.
option(USE_CUDA "CUDA support." ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

set(CORE_FILES
        ${PROJECT_SOURCE_DIR}/src/activation_layer.c
        ${PROJECT_SOURCE_DIR}/src/activations.c
        ${PROJECT_SOURCE_DIR}/src/avgpool_layer.c
        ${PROJECT_SOURCE_DIR}/src/batchnorm_layer.c
        ${PROJECT_SOURCE_DIR}/src/blas.c
        ${PROJECT_SOURCE_DIR}/src/box.c
        ${PROJECT_SOURCE_DIR}/src/col2im.c
        ${PROJECT_SOURCE_DIR}/src/connected_layer.c
        ${PROJECT_SOURCE_DIR}/src/convolutional_layer.c
        ${PROJECT_SOURCE_DIR}/src/cost_layer.c
        ${PROJECT_SOURCE_DIR}/src/crnn_layer.c
        ${PROJECT_SOURCE_DIR}/src/crop_layer.c
        ${PROJECT_SOURCE_DIR}/src/cuda.c
        ${PROJECT_SOURCE_DIR}/src/data.c
        ${PROJECT_SOURCE_DIR}/src/deconvolutional_layer.c
        ${PROJECT_SOURCE_DIR}/src/demo.c
        ${PROJECT_SOURCE_DIR}/src/detection_layer.c
        ${PROJECT_SOURCE_DIR}/src/dropout_layer.c
        ${PROJECT_SOURCE_DIR}/src/gemm.c
        ${PROJECT_SOURCE_DIR}/src/gru_layer.c
        ${PROJECT_SOURCE_DIR}/src/im2col.c
        ${PROJECT_SOURCE_DIR}/src/image.c
        ${PROJECT_SOURCE_DIR}/src/image_opencv.cpp
        ${PROJECT_SOURCE_DIR}/src/iseg_layer.c
        ${PROJECT_SOURCE_DIR}/src/l2norm_layer.c
        ${PROJECT_SOURCE_DIR}/src/layer.c
        ${PROJECT_SOURCE_DIR}/src/list.c
        ${PROJECT_SOURCE_DIR}/src/local_layer.c
        ${PROJECT_SOURCE_DIR}/src/logistic_layer.c
        ${PROJECT_SOURCE_DIR}/src/lstm_layer.c
        ${PROJECT_SOURCE_DIR}/src/matrix.c
        ${PROJECT_SOURCE_DIR}/src/maxpool_layer.c
        ${PROJECT_SOURCE_DIR}/src/network.c
        ${PROJECT_SOURCE_DIR}/src/normalization_layer.c
        ${PROJECT_SOURCE_DIR}/src/option_list.c
        ${PROJECT_SOURCE_DIR}/src/parser.c
        ${PROJECT_SOURCE_DIR}/src/region_layer.c
        ${PROJECT_SOURCE_DIR}/src/reorg_layer.c
        ${PROJECT_SOURCE_DIR}/src/rnn_layer.c
        ${PROJECT_SOURCE_DIR}/src/route_layer.c
        ${PROJECT_SOURCE_DIR}/src/shortcut_layer.c
        ${PROJECT_SOURCE_DIR}/src/softmax_layer.c
        ${PROJECT_SOURCE_DIR}/src/tree.c
        ${PROJECT_SOURCE_DIR}/src/upsample_layer.c
        ${PROJECT_SOURCE_DIR}/src/utils.c
        ${PROJECT_SOURCE_DIR}/src/yolo_layer.c)

set(CORE_CU_FILES
        ${PROJECT_SOURCE_DIR}/src/activation_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/avgpool_layer_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/col2im_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/blas_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/convolutional_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/crop_layer_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/deconvolutional_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/dropout_layer_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/im2col_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/maxpool_layer_kernels.cu)

INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(darknet STATIC ${CORE_FILES})