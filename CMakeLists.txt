project( Darknet )

cmake_minimum_required( VERSION 3.0 )

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

include_directories( ${CMAKE_SOURCE_DIR}/include )

if( WIN32 )
  set( DARKNET_LINKED_LIBS )
  include_directories( ${CMAKE_SOURCE_DIR}/3rdparty/include )
  add_definitions( -D_TIMESPEC_DEFINED )
  if( MSVC )
    add_definitions( -DMSVC )
  endif()
elseif( APPLE )
  add_definitions( -DMAC )
else()
  set( DARKNET_LINKED_LIBS m )
endif()

if( USE_GPU )
  find_package( CUDA QUIET REQUIRED )
  include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include )
  add_definitions( -DGPU )

  if( USE_CUDNN )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DCUDNN )
  endif()

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
endif()

if( USE_CUDNN )

  if(NOT USE_GPU)
    message(FATAL_ERROR "Enable GPU support when wanting to use CUDNN")
  endif()

  find_library( CUDNN_LIBRARIES
    NAMES cudnn libcudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
          ${CUDNN_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86 targets/aarch64-linux
  )

  if(NOT CUDNN_LIBRARIES)
    # Set this if cuda and cudnn are in seperate locations
    set( CUDNN_TOOLKIT_ROOT_DIR "" CACHE PATH "CUDNN root folder" )
    message(FATAL_ERROR "Unable to find cudnn libraries, please ensure \
      CUDA_TOOLKIT_ROOT_DIR has cudnn or the CUDNN_TOOLKIT_ROOT_DIR variable is \
      properly set or set CUDNN_LIBRARIES" )
  endif()

  if(CUDNN_TOOLKIT_ROOT_DIR)
    include_directories( SYSTEM ${CUDNN_TOOLKIT_ROOT_DIR}/include )
  endif()

  list( APPEND DARKNET_LINKED_LIBS ${CUDNN_LIBRARIES} )
  add_definitions( -DCUDNN )
endif()

if( USE_OPENCV )
  find_package( OpenCV REQUIRED )

  include_directories( SYSTEM ${OpenCV_INCLUDE_DIRS} )
  add_definitions( -DOPENCV )
  add_definitions( -DCV_MAJOR_VERSION=${OpenCV_VERSION_MAJOR} )
  list( APPEND DARKNET_LINKED_LIBS opencv_core opencv_highgui opencv_imgproc )

  if( OpenCV_VERSION_MAJOR GREATER 2 )
    list( APPEND DARKNET_LINKED_LIBS opencv_videoio opencv_imgcodecs )
  endif()
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

add_subdirectory( include )
add_subdirectory( src )
add_subdirectory( examples )

if( WIN32 )
  add_subdirectory( 3rdparty )
endif()

# configure the package description file.
configure_file( DarknetConfig.cmake.in
  ${CMAKE_BINARY_DIR}/DarknetConfig.cmake @ONLY )

install( FILES ${CMAKE_BINARY_DIR}/DarknetConfig.cmake
    DESTINATION "CMake"
  )
