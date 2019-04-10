cmake_minimum_required(VERSION 3.8)

set(Darknet_MAJOR_VERSION 0)
set(Darknet_MINOR_VERSION 2)
set(Darknet_PATCH_VERSION 5)
set(Darknet_VERSION ${Darknet_MAJOR_VERSION}.${Darknet_MINOR_VERSION}.${Darknet_PATCH_VERSION})

set(CMAKE_VERBOSE_MAKEFILE "FALSE" CACHE BOOL "Create verbose makefile")
set(BUILD_SHARED_LIBS "TRUE" CACHE BOOL "Create darklib as a shared library")

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  message(STATUS "VCPKG found: $ENV{VCPKG_ROOT}")
  message(STATUS "Using VCPKG integration")
endif()

project(Darknet VERSION ${Darknet_VERSION})

if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(USE_INTEGRATED_LIBS "TRUE" CACHE BOOL "Use libs distributed with this repo")
else()
  set(USE_INTEGRATED_LIBS "FALSE" CACHE BOOL "Use libs distributed with this repo")
endif()

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "lib"                       CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include"                   CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/darknet"             CACHE PATH "Path where cmake configs will be installed")
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/src/version.h.in"
  "${CMAKE_CURRENT_LIST_DIR}/src/version.h"
)

set(ENABLE_CUDA "TRUE" CACHE BOOL "Enable CUDA build")
if(${CMAKE_VERSION} VERSION_LESS "3.8.0")
  set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA build" FORCE)
  message(WARNING "To build with CUDA support you need CMake 3.8.0+")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA build" FORCE)
  endif()
endif()

set(CUDA_COMPUTE_MODEL_75 "75") #Compute capability for Tesla T4, RTX 2080
set(CUDA_COMPUTE_MODEL_72 "72") #Compute capability for Jetson Xavier
set(CUDA_COMPUTE_MODEL_70 "70") #Compute capability for Tesla V100
set(CUDA_COMPUTE_MODEL_62 "62") #Compute capability for Jetson TX2
set(CUDA_COMPUTE_MODEL_61 "61") #Compute capability for Tesla P40
set(CUDA_COMPUTE_MODEL_60 "60") #Compute capability for Tesla P100
set(CUDA_COMPUTE_MODEL_53 "53") #Compute capability for Jetson TX1
set(CUDA_COMPUTE_MODEL_52 "52") #Compute capability for Tesla M40/M60
set(CUDA_COMPUTE_MODEL_37 "37") #Compute capability for Tesla K80
set(CUDA_COMPUTE_MODEL_35 "35") #Compute capability for Tesla K20/K40
set(CUDA_COMPUTE_MODEL_30 "30") #Compute capability for Tesla K10, Quadro K4000

set(default_cuda_compute_model ${CUDA_COMPUTE_MODEL_30})
if(ENABLE_CUDA AND NOT CUDA_COMPUTE_MODEL)
  message(STATUS "Setting the compute model to ${default_cuda_compute_model} as none was specified.")
  set(CUDA_COMPUTE_MODEL "${default_cuda_compute_model}" CACHE
      STRING "Choose the best compute model supported by your GPU" FORCE)
  set_property(CACHE CUDA_COMPUTE_MODEL PROPERTY STRINGS ${CUDA_COMPUTE_MODEL_75} ${CUDA_COMPUTE_MODEL_72} ${CUDA_COMPUTE_MODEL_70} ${CUDA_COMPUTE_MODEL_62} ${CUDA_COMPUTE_MODEL_61} ${CUDA_COMPUTE_MODEL_60} ${CUDA_COMPUTE_MODEL_53} ${CUDA_COMPUTE_MODEL_52} ${CUDA_COMPUTE_MODEL_37} ${CUDA_COMPUTE_MODEL_35} ${CUDA_COMPUTE_MODEL_30})
endif()

if(NOT ENABLE_CUDA)
  set(CUDA_COMPUTE_MODEL "0" CACHE STRING "CUDA Compute Model set to 0 because CUDA is missing" FORCE)
endif()

if(USE_INTEGRATED_LIBS)
  set(PThreads_windows_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/pthreads CACHE PATH "Path where pthreads for windows can be located")
  add_definitions(-D_TIMESPEC_DEFINED)
endif()
set(Stb_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb CACHE PATH "Path where Stb image library can be located")

set(CMAKE_DEBUG_POSTFIX d)
add_definitions(-DUSE_CMAKE_LIBS)

if(MSVC)
  set(CMAKE_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  add_definitions(-D_CRT_RAND_S)
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-Wno-unused-result -Wno-write-strings ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")  #force pthread to avoid bugs in some cmake setups
  endif()
  set(CMAKE_CXX_FLAGS "-Wno-deprecated -Wno-writable-strings ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
endif()

set(ENABLE_OPENCV "TRUE" CACHE BOOL "Enable OpenCV integration")
if(ENABLE_OPENCV)
  find_package(OpenCV)
endif()
find_package(Stb REQUIRED)

set(SKIP_USELIB_TRACK "FALSE" CACHE BOOL "Skip building uselib_track" FORCE)
if(OpenCV_FOUND)
  if(ENABLE_CUDA AND NOT OpenCV_CUDA_VERSION)
    set(SKIP_USELIB_TRACK "TRUE" CACHE BOOL "Skip building uselib_track" FORCE)
    message(STATUS "  ->  darknet is fine for now, but uselib_track has been disabled!")
    message(STATUS "  ->  Please rebuild OpenCV from sources with CUDA support to enable it")
  endif()
  add_definitions(-DOPENCV)
endif()

if(ENABLE_CUDA)
  add_definitions(-DGPU)
  if(NOT WIN32)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
  endif()
  if(CUDA_COMPUTE_MODEL VERSION_GREATER_EQUAL ${CUDA_COMPUTE_MODEL_75} AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 9)
    message(STATUS "CUDA 10 is required for Compute Model 7.5 - Lowering settings to 7.0")
    set(CUDA_COMPUTE_MODEL ${CUDA_COMPUTE_MODEL_70})
  endif()
  find_package(CUDNN)
  if(CUDNN_FOUND)
    add_definitions(-DCUDNN)
    if(CUDA_COMPUTE_MODEL VERSION_GREATER_EQUAL ${CUDA_COMPUTE_MODEL_70})
      add_definitions(-DCUDNN_HALF)
      message(STATUS "Enabled CUDNN_HALF")
    endif()
  endif()
  if (MSVC)
    if(CUDNN_FOUND)
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/wd4819,/DGPU,/DCUDNN,/DOPENCV\" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/wd4819,/DGPU,/DCUDNN\" ${CMAKE_CUDA_FLAGS}")
      endif()
    else()
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/wd4819,/DGPU,/DOPENCV\" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/wd4819,/DGPU\" ${CMAKE_CUDA_FLAGS}")
      endif()
    endif()
  else()
    if(CUDNN_FOUND)
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DOPENCV -DGPU -DCUDNN \" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DGPU -DCUDNN \" ${CMAKE_CUDA_FLAGS}")
      endif()
    else()
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DOPENCV -DGPU \" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${CUDA_COMPUTE_MODEL},code=[sm_${CUDA_COMPUTE_MODEL},compute_${CUDA_COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DGPU \" ${CMAKE_CUDA_FLAGS}")
      endif()
    endif()
  endif()
endif()

set(ENABLE_ZED_CAMERA "TRUE" CACHE BOOL "Enable ZED Camera support")
if(ENABLE_CUDA)
  if(ENABLE_ZED_CAMERA)
    find_package(ZED 2 QUIET)
    if(ZED_FOUND)
      include_directories(${ZED_INCLUDE_DIRS})
      link_directories(${ZED_LIBRARY_DIR})
      add_definitions(-DZED_STEREO)
      message(STATUS "ZED SDK enabled")
    else()
      message(STATUS "ZED SDK not found")
      set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
    endif()
  endif()
else()
  message(STATUS "ZED SDK not enabled, since it requires CUDA")
  set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
if(MSVC)
  find_package(PThreads_windows REQUIRED)
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
  message(WARNING "To build with OpenMP support you need CMake 3.9.0+")
else()
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

#look for all *.h files in src folder
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/src/*.h")
#add also files in the include folder
list(APPEND headers
  ${CMAKE_CURRENT_LIST_DIR}/include/darknet.h
)
#remove windows only files, will be re-added if necessary later
list(REMOVE_ITEM headers
  ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.h
  ${CMAKE_CURRENT_LIST_DIR}/src/getopt.h
)
set(exported_headers ${headers})

#look for all *.c files in src folder
file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
#add also .cpp files
list(APPEND sources
  ${CMAKE_CURRENT_LIST_DIR}/src/http_stream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/image_opencv.cpp
)
#remove darknet.c file which is necessary only for the executable, not for the lib
list(REMOVE_ITEM sources
  ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c
)
#remove windows only files, will be re-added if necessary later
list(REMOVE_ITEM sources
  ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.c
  ${CMAKE_CURRENT_LIST_DIR}/src/getopt.c
)

if(ENABLE_CUDA)
  file(GLOB cuda_sources "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
endif()

if(WIN32)
  list(APPEND sources
    ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.c
    ${CMAKE_CURRENT_LIST_DIR}/src/getopt.c
  )

  list(APPEND headers
    ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.h
    ${CMAKE_CURRENT_LIST_DIR}/src/getopt.h
  )
endif()

set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)

add_library(darklib ${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp ${CMAKE_CURRENT_LIST_DIR}/src/yolo_v2_class.cpp ${sources} ${headers} ${cuda_sources})
set_target_properties(darklib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  set_target_properties(darklib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_compile_definitions(darklib PRIVATE LIB_EXPORTS=1)
set_target_properties(darklib PROPERTIES LINKER_LANGUAGE CXX)

if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND NOT SKIP_USELIB_TRACK)
  add_executable(uselib_track ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
  target_compile_definitions(uselib_track PRIVATE TRACK_OPTFLOW=1)
  set_target_properties(uselib_track PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(uselib_track PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
  target_link_libraries(uselib_track PRIVATE ${OpenCV_LIBS})
endif()

add_executable(uselib ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
set_target_properties(uselib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(darknet ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c ${sources} ${headers} ${cuda_sources})
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/darknet.c PROPERTIES LANGUAGE CXX)
set_target_properties(darknet PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(darknet PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
target_include_directories(darklib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
target_include_directories(uselib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

if(CUDNN_FOUND)
  target_link_libraries(darknet PRIVATE CuDNN::CuDNN)
  target_link_libraries(darklib PRIVATE CuDNN::CuDNN)
endif()

if(OpenCV_FOUND)
  target_link_libraries(darknet PRIVATE ${OpenCV_LIBS})
  target_link_libraries(uselib PRIVATE ${OpenCV_LIBS})
  target_link_libraries(darklib PRIVATE ${OpenCV_LIBS})
endif()

if(OPENMP_FOUND)
  target_link_libraries(darknet PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(darklib PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(darknet PRIVATE m)
  target_link_libraries(darklib PRIVATE m)
endif()

if(MSVC)
  target_link_libraries(darknet PRIVATE PThreads_windows::PThreads_windows)
  target_link_libraries(darknet PRIVATE wsock32 ws2_32)
  target_link_libraries(darklib PRIVATE PThreads_windows::PThreads_windows)
  target_link_libraries(darklib PRIVATE wsock32 ws2_32)
endif()

target_link_libraries(darknet PRIVATE Threads::Threads)
target_link_libraries(darklib PRIVATE Threads::Threads)

if(ENABLE_ZED_CAMERA)
  target_link_libraries(darknet PRIVATE ${ZED_LIBRARIES})
  target_link_libraries(uselib PRIVATE ${ZED_LIBRARIES})
endif()

if(ENABLE_CUDA)
  target_link_libraries(darknet PRIVATE curand cublas cuda)
  target_link_libraries(darklib PRIVATE curand cublas cuda)
  set_target_properties(darklib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_link_libraries(uselib PRIVATE darklib)
if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND NOT SKIP_USELIB_TRACK)
  target_link_libraries(uselib_track PRIVATE darklib)
endif()

set_target_properties(darklib PROPERTIES PUBLIC_HEADER "${exported_headers};${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp")

install(TARGETS darklib EXPORT DarknetTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
  COMPONENT dev
)
install(TARGETS uselib darknet
  DESTINATION "${INSTALL_BIN_DIR}"
)
if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND NOT SKIP_USELIB_TRACK)
  install(TARGETS uselib_track
    DESTINATION "${INSTALL_BIN_DIR}"
  )
endif()

install(EXPORT DarknetTargets
  FILE DarknetTargets.cmake
  NAMESPACE Darknet::
  DESTINATION "${INSTALL_CMAKE_DIR}"
)

# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export(PACKAGE Darknet)

# Create the DarknetConfig.cmake
# First of all we compute the relative path between the cmake config file and the include path
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}/DarknetConfig.cmake" @ONLY)
set(CONF_INCLUDE_DIRS "\${Darknet_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DarknetConfig.cmake" @ONLY)

# Create the DarknetConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/DarknetConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DarknetConfig.cmake"
  "${PROJECT_BINARY_DIR}/DarknetConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}"
)
