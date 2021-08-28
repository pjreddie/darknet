cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 52)

project(darknet LANGUAGES C CXX CUDA)

option(DARKNET_CPU "Enable CPU support" OFF)
option(DARKNET_OPENCV "Enable OpenCV support" ON)
option(DARKNET_OPENMP "Enable OpenMP support" OFF)
option(DARKNET_GPU "Enable OpenCL support" ON)
option(DARKNET_ARM "Enable ARM support" OFF)
option(DARKNET_BENCHMARK "Some Benchmark Stats" OFF)
option(DARKNET_LOSS_ONLY "Loss Only Stats" OFF)
option(DARKNET_TESTING "Build the unit tests" OFF)
option(DARKNET_DEBUG "Enable debug mode" OFF)
option(DARKNET_STATIC "Build static library" ON)
option(DARKNET_SHARED "Build shared library" ON)
option(DARKNET_NO_BINARY "Do not build binary" OFF)

include(GNUInstallDirs)

include_directories("include" "src")

set (DARKNET_DEFINITIONS "")
set (DARKNET_BINARY bindarknet)
set (DARKNET_LIBRARY libdarknet)
set (DARKNET_LIBRARY_STATIC libdarknet_s)

if (DARKNET_DEBUG OR DARKNET_TESTING)
    list (APPEND DARKNET_CFLAGS "-g")
else()
	if (DARKNET_OPENMP)
		list (APPEND DARKNET_CFLAGS "-openmp")
	endif()
#	list (APPEND DARKNET_CFLAGS "-Wfatal-errors")
#	list (APPEND DARKNET_CFLAGS "-Wno-unused-variable")
#	list (APPEND DARKNET_CFLAGS "-Wno-unused-result")
#	list (APPEND DARKNET_CFLAGS "-Wno-deprecated-declarations")
	list (APPEND DARKNET_CFLAGS "-O2")
endif()

# Find OpenCV
if (DARKNET_OPENCV)
	find_package(OpenCV REQUIRED)
	add_library(opencv INTERFACE IMPORTED)
	set_property(TARGET opencv PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
    set_property(TARGET opencv PROPERTY
        INTERFACE_LINK_LIBRARIES ${OpenCV_LIBRARIES})
    list (APPEND DARKNET_DEFINITIONS OPENCV)
endif()

# Find OpenCL
if (DARKNET_GPU)
	enable_language(CUDA)
	find_package(CUDA REQUIRED)
	include_directories("${CUDA_INCLUDE_DIRS}")
	list (APPEND DARKNET_DEFINITIONS GPU)
	# Raspberry PI
	if (DARKNET_ARM)
		list (APPEND DARKNET_DEFINITIONS ARM)
	endif()
endif()

# Benchmark Stats
if (DARKNET_BENCHMARK)
	list (APPEND DARKNET_DEFINITIONS BENCHMARK)
endif()

# Loss Stats
if (DARKNET_LOSS_ONLY)
	list (APPEND DARKNET_DEFINITIONS LOSS_ONLY)
endif()

# Find Catch
if (DARKNET_UNIT)
	set (DARKNET_CPU OFF)
	find_package(Catch REQUIRED)
	add_library(catch INTERFACE IMPORTED)
	set_property(TARGET catch PROPERTY
		INTERFACE_INCLUDE_DIRECTORIES ${Catch_INCLUDE_DIRS})
endif()

set (DARKNET_SOURCES
    "src/activation_layer.c"
    "src/activations.c"
    "examples/art.c"
    "examples/attention.c"
    "src/avgpool_layer.c"
    "src/batchnorm_layer.c"
    "src/blas.c"
    "src/box.c"
    "examples/captcha.c"
    "examples/cifar.c"
    "examples/classifier.c"
    "examples/coco.c"
    "src/col2im.c"
    "src/compare.c"
    "src/connected_layer.c"
    "src/convolutional_layer.c"
    "src/cost_layer.c"
    "src/crnn_layer.c"
    "src/crop_layer.c"
    "src/data.c"
    "src/deconvolutional_layer.c"
    "src/demo.c"
    "src/detection_layer.c"
    "examples/detector.c"
    "examples/dice.c"
    "src/dropout_layer.c"
    "src/gemm.c"
    "examples/go.c"
    "src/gru_layer.c"
    "src/im2col.c"
    "src/image.c"
	"src/iseg_layer.c"
	"examples/instance-segmenter.c"
    "src/layer.c"
    "src/list.c"
    "src/local_layer.c"
    "src/lstm_layer.c"
    "examples/lsd.c"
    "src/matrix.c"
    "src/maxpool_layer.c"
    "src/network.c"
    "examples/nightmare.c"
    "src/normalization_layer.c"
    "src/option_list.c"
    "src/parser.c"
    "src/region_layer.c"
    "examples/regressor.c"
    "src/reorg_layer.c"
    "examples/rnn.c"
    "src/rnn_layer.c"
    "examples/rnn_vid.c"
    "src/route_layer.c"
    "src/shortcut_layer.c"
    "src/softmax_layer.c"
    "examples/segmenter.c"
    "examples/super.c"
    "examples/swag.c"
    "examples/tag.c"
    "src/tree.c"
    "src/utils.c"
    "src/cuda.c"
    "examples/voxel.c"
    "examples/writing.c"
    "examples/yolo.c"
    "src/yolo_layer.c"
    "src/yolo4_layer.c"
    "src/upsample_layer.c"
    "src/logistic_layer.c"
    "src/l2norm_layer.c"
)

set (DARKNET_PUBLIC_HEADERS
    "include/darknet.h"
    "src/activation_layer.h"
    "src/activations.h"
    "src/avgpool_layer.h"
    "src/batchnorm_layer.h"
    "src/blas.h"
    "src/box.h"
    "src/bugfixes.h"
    "src/classifier.h"
    "src/col2im.h"
    "src/connected_layer.h"
    "src/convolutional_layer.h"
    "src/cost_layer.h"
    "src/crnn_layer.h"
    "src/crop_layer.h"
    "src/cuda.h"
    "src/data.h"
    "src/deconvolutional_layer.h"
    "src/demo.h"
    "src/detection_layer.h"
    "src/dropout_layer.h"
    "src/gemm.h"
    "src/gru_layer.h"
    "src/im2col.h"
    "src/image.h"
    "src/iseg_layer.h"
    "src/layer.h"
    "src/list.h"
    "src/local_layer.h"
    "src/lstm_layer.h"
    "src/matrix.h"
    "src/maxpool_layer.h"
    "src/network.h"
    "src/normalization_layer.h"
    "src/option_list.h"
    "src/parser.h"
    "src/region_layer.h"
    "src/reorg_layer.h"
    "src/rnn_layer.h"
    "src/route_layer.h"
    "src/shortcut_layer.h"
    "src/softmax_layer.h"
    "src/stb_image.h"
    "src/stb_image_write.h"
    "src/tree.h"
    "src/unit.h"
    "src/utils.h"
    "src/cuda.h"
    "src/yolo_layer.h"
    "src/yolo4_layer.h"
    "src/upsample_layer.h"
    "src/logistic_layer.h"
    "src/l2norm_layer.h"
)

set (DARKNET_MAIN_SOURCE
    "examples/darknet.c"
)

set (DARKNET_SOURCES_CXX
    "src/image_opencv.cpp"
)

if (DARKNET_CPU)
	set (DARKNET_CPU_SOURCES
        "src/cpu.c"
    )
endif()

if (DARKNET_GPU)
	set (DARKNET_GPU_SOURCES
	    "src/activation_kernels.cu"
	    "src/avgpool_layer_kernels.cu"
	    "src/blas_kernels.cu"
	    "src/col2im_kernels.cu"
	    "src/convolutional_kernels.cu"
	    "src/crop_layer_kernels.cu"
        "src/deconvolutional_kernels.cu"
	    "src/dropout_layer_kernels.cu"
	    "src/im2col_kernels.cu"
	    "src/maxpool_layer_kernels.cu"
	)
endif()

if (DARKNET_UNIT)
	set (DARKNET_UNIT_SOURCES
	    "src/unit.c"
	    "src/unit.h"
	    "src/blas_unit.c"
	    "src/col2im_unit.c"
	    "src/convolutional_unit.c"
	    "src/gemm_unit.c"
	    "src/maxpool_unit.c"
	    "src/network_unit.c"
	    "src/region_unit.c"
	)
endif()

function (ConfigureBinary target)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "darknet")    
    target_compile_definitions(${target} PUBLIC ${DARKNET_DEFINITIONS})
    target_compile_options(${target} PRIVATE ${DARKNET_CFLAGS})

    if (DARKNET_ARM)
		link_libraries(curand cublas cuda)
		file(GLOB cuda_sources "${DARKNET_GPU_SOURCES}")
    elseif (DARKNET_GPU)
		link_libraries(curand cublas cuda)
		file(GLOB cuda_sources "${DARKNET_GPU_SOURCES}")
    endif()

    if (DARKNET_OPENCV)
        target_link_libraries(${target} opencv)
    endif()

#    if (DARKNET_TESTING)
#        target_link_libraries(${target} catch)
#    endif()

    target_link_libraries(${target} "m" "pthread")
endfunction()

function (ConfigureLibrary target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    install(TARGETS ${target} EXPORT ${CMAKE_PROJECT_NAME}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if (DARKNET_SHARED)
    add_library(${DARKNET_LIBRARY} SHARED ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
	   ${DARKNET_GPU_SOURCES} ${DARKNET_WINDOWS_SOURCES} ${DARKNET_SOURCES_CXX})
    ConfigureBinary(${DARKNET_LIBRARY})
    ConfigureLibrary(${DARKNET_LIBRARY})
endif()

if (DARKNET_STATIC)
    add_library(${DARKNET_LIBRARY_STATIC} STATIC ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
       ${DARKNET_GPU_SOURCES} ${DARKNET_WINDOWS_SOURCES} ${DARKNET_SOURCES_CXX})
    ConfigureBinary(${DARKNET_LIBRARY_STATIC})
    ConfigureLibrary(${DARKNET_LIBRARY_STATIC})
endif()

if (NOT DARKNET_NO_BINARY)
    add_executable(${DARKNET_BINARY} ${DARKNET_SOURCES} ${DARKNET_CPU_SOURCES}
        ${DARKNET_GPU_SOURCES} ${DARKNET_WINDOWS_SOURCES} ${DARKNET_SOURCES_CXX}
        ${DARKNET_MAIN_SOURCE})

    ConfigureBinary(${DARKNET_BINARY})
endif()

install(FILES ${DARKNET_PUBLIC_HEADERS} ${DARKNET_WINDOWS_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${CMAKE_PROJECT_NAME})

install(EXPORT ${CMAKE_PROJECT_NAME} DESTINATION share/${CMAKE_PROJECT_NAME}/cmake)

if (DARKNET_SHARED)
    export(TARGETS ${DARKNET_LIBRARY} FILE darknetConfig.cmake)
endif()
