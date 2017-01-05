#
# Build file for Darknet
#

OPTION(BUILD_SHARED_LIBS     "Build components shared or not" TRUE )

OPTION( USE_GPU "Use GPU support" FALSE )
OPTION( USE_CUDNN "Use CUDNN support" FALSE )
OPTION( USE_OPENCV "Use OpenCV support" FALSE )

find_package (Threads)

set( gpu_source
  convolutional_kernels.cu
  activation_kernels.cu
  im2col_kernels.cu
  col2im_kernels.cu
  blas_kernels.cu
  crop_layer_kernels.cu
  dropout_layer_kernels.cu
  maxpool_layer_kernels.cu
  network_kernels.cu
  avgpool_layer_kernels.cu
)

set( source
  activation_layer.c
  activations.c
  art.c
  avgpool_layer.c
  batchnorm_layer.c
  blas.c
  box.c
  captcha.c
  cifar.c
  classifier.c
  coco.c
  col2im.c
  compare.c
  connected_layer.c
  convolutional_layer.c
  cost_layer.c
  crnn_layer.c
  crop_layer.c
  cuda.c
  data.c
  deconvolutional_layer.c
  demo.c
  detection_layer.c
  detector.c
  dice.c
  dropout_layer.c
  gemm.c
  go.c
  gru_layer.c
  im2col.c
  image.c
  layer.c
  list.c
  local_layer.c
  matrix.c
  maxpool_layer.c
  network.c
  nightmare.c
  normalization_layer.c
  option_list.c
  parser.c
  region_layer.c
  reorg_layer.c
  rnn.c
  rnn_layer.c
  rnn_vid.c
  route_layer.c
  shortcut_layer.c
  softmax_layer.c
  super.c
  swag.c
  tag.c
  tree.c
  utils.c
  voxel.c
  writing.c
  yolo.c
  )

set( headers
  activation_layer.h
  activations.h
  avgpool_layer.h
  batchnorm_layer.h
  blas.h
  box.h
  classifier.h
  col2im.h
  connected_layer.h
  convolutional_layer.h
  cost_layer.h
  crnn_layer.h
  crop_layer.h
  cuda.h
  data.h
  deconvolutional_layer.h
  demo.h
  detection_layer.h
  dropout_layer.h
  gemm.h
  gru_layer.h
  im2col.h
  image.h
  layer.h
  list.h
  local_layer.h
  matrix.h
  maxpool_layer.h
  network.h
  normalization_layer.h
  option_list.h
  parser.h
  region_layer.h
  reorg_layer.h
  rnn_layer.h
  route_layer.h
  shortcut_layer.h
  softmax_layer.h
  stb_image.h
  stb_image_write.h
  tree.h
  utils.h
  )

set( libs m )

if (USE_GPU)
  find_package(CUDA QUIET REQUIRED)

  set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_20,code=[sm_20,sm_21]
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52] )

  #  COMMON+= -DGPU -I/usr/local/cuda/include/
  #  CFLAGS+= -DGPU
  #  LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
  #  LDFLAGS+= -lstdc++
  list( APPEND libs cuda cudart cublas curand )
  add_definitions( -DGPU )
  set( source ${source} ${gpu_source} )
endif()

if (USE_CUDNN)
  # COMMON+= -DCUDNN
  # CFLAGS+= -DCUDNN
  # LDFLAGS+= -lcudnn
  list( APPEND libs cudnn )
  add_definitions( -DCUDNN )

endif()

if (USE_OPENCV)
  find_package( OpenCV 2.4.6 REQUIRED )
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

  if( OpenCV_VERSION VERSION_GREATER "2.4" )
    set( USE_OPENCV True )
    if( OpenCV_VERSION VERSION_GREATER "3.0" OR
        OpenCV_VERSION VERSION_EQUAL "3.0")
      message(STATUS "Found OPENCV 3.0")
      set (KWIVER_HAS_OPENCV_VER_3 True)
    endif()

  else()
    message( FATAL_ERROR "OpenCV version must be at least 2.4" )
  endif()

  list( APPEND libs opencv_core opencv_highgui )
endif()

set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wfatal-errors")


###
#
add_library( darknet_lib
  ${headers}
  ${source}
  )

target_link_libraries( darknet_lib
  PRIVATE  ${CMAKE_THREAD_LIBS_INIT}
           ${libs}
  )

set_target_properties( darknet_lib
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )


###
#
add_executable( darknet
  darknet.c )

target_link_libraries( darknet
  darknet_lib
  ${CMAKE_THREAD_LIBS_INIT} ${libs}
  )

set_target_properties( darknet
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

###
#
install( TARGETS darknet_lib darknet
  ARCHIVE DESTINATION     lib
  LIBRARY DESTINATION     lib
  RUNTIME DESTINATION     bin
  COMPONENT    runtime
  )

install( FILES ${headers}
  DESTINATION "include/darknet"
  )
