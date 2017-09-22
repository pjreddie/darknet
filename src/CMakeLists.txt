##
# Build file for Darknet
##
if( WIN32 )
  OPTION( BUILD_SHARED_LIBS  "Build components shared or not"  FALSE )

  if( BUILD_SHARED_LIBS )
    message( FATAL_ERROR "Currently only static builds are supported on windows" )
  endif()
else()
  OPTION( BUILD_SHARED_LIBS  "Build components shared or not"  TRUE )
endif()

OPTION( USE_GPU      "Use GPU support"      FALSE )
OPTION( USE_CUDNN    "Use CUDNN support"    FALSE )
OPTION( USE_OPENCV   "Use OpenCV support"   FALSE )

find_package( Threads )

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

if( WIN32 )
  set( source
    ${source}
    cpu_gemm.c
    getopt.c
    gettimeofday.c )

  set( headers
    ${headers}
    getopt.h
    gettimeofday.h )
endif()

if( WIN32 )
  set( DARKNET_LINKED_LIBS )
  include_directories( ${CMAKE_SOURCE_DIR}/3rdparty/include )
  add_definitions( -D_TIMESPEC_DEFINED )
else()
  set( DARKNET_LINKED_LIBS m )
endif()

if( USE_GPU )
  find_package( CUDA QUIET REQUIRED )
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DGPU )

  set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_20,code=[sm_20,sm_21]
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -Xcompiler )

  if( NOT WIN32 )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -fPIC )
  endif()

  CUDA_COMPILE( CUDA_SRC ${gpu_source} )
  list( APPEND DARKNET_LINKED_LIBS
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY} )
endif()

if( USE_CUDNN )
  
  if(NOT USE_GPU)
    message(FATAL_ERROR "Enable GPU support when wanting to use CUDNN")
  endif()
  
  find_library( CUDNN_LIBRARIES
    NAMES cudnn libcudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
          ${CUDNN_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86 targets/aarch64-linux
  )
  
  if(NOT CUDNN_LIBRARIES)
    # Set this if cuda and cudnn are in seperate locations
    set( CUDNN_ROOT_DIR "" CACHE PATH "CUDNN root folder" )
    message(FATAL_ERROR "Unable to find cudnn libraries, please ensure CUDA_TOOLKIT_ROOT_DIR has cudnn or the CUDNN_ROOT_DIR variable is properly set or set CUDNN_LIBRARIES")
  endif()
  
  if(CUDNN_ROOT_DIR)
    include_directories( SYSTEM ${CUDNN_ROOT_DIR}/include )
  endif()
  
  list( APPEND DARKNET_LINKED_LIBS ${CUDNN_LIBRARIES} )
  add_definitions( -DCUDNN )
endif()

if( USE_OPENCV )
  find_package( OpenCV REQUIRED )

  include_directories( SYSTEM ${OpenCV_INCLUDE_DIRS} )
  add_definitions( -DOPENCV )
  list( APPEND DARKNET_LINKED_LIBS opencv_core opencv_highgui opencv_videoio opencv_imgcodecs)
endif()

if( WIN32 )
  if( NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)" )
    link_directories( ${CMAKE_SOURCE_DIR}/3rdparty/lib/x86 )
  else()
    link_directories( ${CMAKE_SOURCE_DIR}/3rdparty/lib/x64 )
  endif()
  list( APPEND DARKNET_LINKED_LIBS pthreadVC2 )
else()
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wfatal-errors" )
endif()

add_library( darknet_lib
  ${headers}
  ${source}
  ${CUDA_SRC}
  )

target_link_libraries( darknet_lib
  PRIVATE  ${CMAKE_THREAD_LIBS_INIT}
	         ${DARKNET_LINKED_LIBS}
  )

set_target_properties( darknet_lib
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

install( TARGETS darknet_lib
  ARCHIVE DESTINATION     lib
  LIBRARY DESTINATION     lib
  RUNTIME DESTINATION     bin
  COMPONENT    runtime
  )

add_executable( darknet
  darknet.c )

target_link_libraries( darknet
  darknet_lib
  ${CMAKE_THREAD_LIBS_INIT} ${DARKNET_LINKED_LIBS}
  )

set_target_properties( darknet
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

install( TARGETS darknet
  ARCHIVE DESTINATION     lib
  LIBRARY DESTINATION     lib
  RUNTIME DESTINATION     bin
  COMPONENT    runtime
  )

if( INSTALL_HEADER_FILES )
  install( FILES ${headers}
    DESTINATION "include/darknet"
    )
endif()

install( FILES ${CMAKE_BINARY_DIR}/DarknetConfig.cmake
    DESTINATION "CMake"
  )
