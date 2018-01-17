cmake_minimum_required(VERSION 3.9)
project(darknet)

set(CMAKE_CXX_STANDARD 11)

# so that we will find .h
include_directories("./include" "./src" "/usr/local/log4c/include" "/usr/local/include/hiredis")

add_definitions(-DOPENCV -DGPU -DDEBUG)
set(COMMON_FLAGS "-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast")
set(CMAKE_C_FLAGS "${COMMON_FLAGS}")

add_executable(darknet
        examples/captcha.c
        examples/lsd.c
        examples/super.c
        examples/art.c
        examples/tag.c
        examples/cifar.c
        examples/go.c
        examples/rnn.c
        examples/segmenter.c
        examples/regressor.c
        examples/classifier.c
        examples/coco.c
        examples/yolo.c
        examples/detector.c
        examples/nightmare.c
        examples/attention.c
        examples/darknet.c
        )

add_library(darklib
        src/activation_layer.c
        src/activations.c
        src/avgpool_layer.c
        src/batchnorm_layer.c
        src/blas.c
        src/box.c
        src/col2im.c
        src/connected_layer.c
        src/convolutional_layer.c
        src/cost_layer.c
        src/crnn_layer.c
        src/crop_layer.c
        src/cuda.c
        src/data.c
        src/deconvolutional_layer.c
        src/demo.c
        src/detection_layer.c
        src/dropout_layer.c
        src/gemm.c
        src/gru_layer.c
        src/im2col.c
        src/image.c
        src/layer.c
        src/list.c
        src/local_layer.c
        src/lstm_layer.c
        src/matrix.c
        src/maxpool_layer.c
        src/network.c
        src/normalization_layer.c
        src/option_list.c
        src/parser.c
        src/region_layer.c
        src/reorg_layer.c
        src/rnn_layer.c
        src/route_layer.c
        src/shortcut_layer.c
        src/softmax_layer.c
        src/stb_image.h
        src/tree.c
        src/utils.c)

set(OPENCV_CFLAGS "--cflags opencv")
set(OPENCV_LIBRARIES "--libs opencv")
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(OPENCV "opencv")
    if (OPENCV_FOUND)
        add_definitions(${OPENCV_CFLAGS})
        link_libraries(darknet ${OPENCV_LIBRARIES})
    else()
    endif()
endif()

find_package(CUDA)
if(CUDA_FOUND)

    # Pass options to NVCC
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -O3 -gencode arch=compute_52,code=compute_52
    )
    cuda_add_library( cudalib
            ./src/convolutional_kernels.cu
            ./src/deconvolutional_kernels.cu
            ./src/activation_kernels.cu
            ./src/im2col_kernels.cu
            ./src/col2im_kernels.cu
            ./src/blas_kernels.cu
            ./src/crop_layer_kernels.cu
            ./src/dropout_layer_kernels.cu
            ./src/maxpool_layer_kernels.cu
            ./src/avgpool_layer_kernels.cu
    )
    link_directories("/usr/local/cuda/lib64")
    target_link_libraries(cudalib -lcuda -lcudart -lcublas -lcurand)
endif()

set(PThreadLib -pthread)
set(MATHLIB -lm)
set(REDISLIB /usr/local/lib/libhiredis.so)
set(LOG4CLIB /usr/local/log4c/lib/liblog4c.so)
target_link_libraries(darklib cudalib ${PThreadLib} ${MATHLIB})
target_link_libraries(darknet darklib ${LOG4CLIB} ${REDISLIB})
