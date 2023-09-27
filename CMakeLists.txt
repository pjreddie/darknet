cmake_minimum_required(VERSION 3.19)
include(CMakeDependentOption)

option(CMAKE_VERBOSE_MAKEFILE "Create verbose makefile" OFF)
option(CUDA_VERBOSE_BUILD "Create verbose CUDA build" OFF)
option(BUILD_SHARED_LIBS "Create dark as a shared library" ON)
option(BUILD_AS_CPP "Build Darknet using C++ compiler also for C files" OFF)
option(BUILD_USELIB_TRACK "Build uselib_track" ON)
option(MANUALLY_EXPORT_TRACK_OPTFLOW "Manually export the TRACK_OPTFLOW=1 define" OFF)
option(ENABLE_OPENCV "Enable OpenCV integration" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
cmake_dependent_option(ENABLE_CUDA_OPENGL_INTEGRATION "Build darknet with support for running networks straight from OpenGL textures" ON "ENABLE_CUDA" OFF)
option(ENABLE_CUDNN "Enable CUDNN" ON)
option(ENABLE_CUDNN_HALF "Enable CUDNN Half precision" ON)
option(ENABLE_ZED_CAMERA "Enable ZED Camera support" ON)
option(ENABLE_VCPKG_INTEGRATION "Enable VCPKG integration" OFF)
option(ENABLE_DEPLOY_CUSTOM_CMAKE_MODULES "Copy custom CMake modules for downstream integration" OFF)
option(ENABLE_CSHARP_WRAPPER "Enable building a csharp wrapper" OFF)
option(ENABLE_INSTALLER "Enable building an installer" OFF)
option(VCPKG_BUILD_OPENCV_WITH_CUDA "Build OpenCV with CUDA extension integration" ON)
option(VCPKG_USE_OPENCV2 "Use legacy OpenCV 2" OFF)
option(VCPKG_USE_OPENCV3 "Use legacy OpenCV 3" OFF)
option(VCPKG_USE_OPENCV4 "Use OpenCV 4" ON)
option(USE_NSIS "Use NSIS as a CPack backend on Windows" ON)

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
  message(STATUS "Setting default vcpkg target triplet to $ENV{VCPKG_DEFAULT_TRIPLET}")
  set(VCPKG_TARGET_TRIPLET $ENV{VCPKG_DEFAULT_TRIPLET})
endif()

if(VCPKG_USE_OPENCV4 AND VCPKG_USE_OPENCV2)
  message(STATUS "You required vcpkg feature related to OpenCV 2 but forgot to turn off those for OpenCV 4, doing that for you")
  set(VCPKG_USE_OPENCV4 OFF CACHE BOOL "Use OpenCV 4" FORCE)
endif()
if(VCPKG_USE_OPENCV4 AND VCPKG_USE_OPENCV3)
  message(STATUS "You required vcpkg feature related to OpenCV 3 but forgot to turn off those for OpenCV 4, doing that for you")
  set(VCPKG_USE_OPENCV4 OFF CACHE BOOL "Use OpenCV 4" FORCE)
endif()
if(VCPKG_USE_OPENCV2 AND VCPKG_USE_OPENCV3)
  message(STATUS "You required vcpkg features related to both OpenCV 2 and OpenCV 3. Impossible to satisfy, keeping only OpenCV 3")
  set(VCPKG_USE_OPENCV2 OFF CACHE BOOL "Use legacy OpenCV 2" FORCE)
endif()

if(ENABLE_CUDA AND NOT APPLE)
  list(APPEND VCPKG_MANIFEST_FEATURES "cuda")
endif()
if(ENABLE_CUDNN AND ENABLE_CUDA AND NOT APPLE)
  list(APPEND VCPKG_MANIFEST_FEATURES "cudnn")
endif()
if(ENABLE_OPENCV)
  if(VCPKG_BUILD_OPENCV_WITH_CUDA AND NOT APPLE)
    if(VCPKG_USE_OPENCV4)
      list(APPEND VCPKG_MANIFEST_FEATURES "opencv-cuda")
    elseif(VCPKG_USE_OPENCV3)
      list(APPEND VCPKG_MANIFEST_FEATURES "opencv3-cuda")
    elseif(VCPKG_USE_OPENCV2)
      list(APPEND VCPKG_MANIFEST_FEATURES "opencv2-cuda")
    endif()
  else()
    if(VCPKG_USE_OPENCV4)
      list(APPEND VCPKG_MANIFEST_FEATURES "opencv-base")
    elseif(VCPKG_USE_OPENCV3)
      list(APPEND VCPKG_MANIFEST_FEATURES "opencv3-base")
    elseif(VCPKG_USE_OPENCV2)
      list(APPEND VCPKG_MANIFEST_FEATURES "opencv2-base")
    endif()
  endif()
endif()

if(NOT CMAKE_HOST_SYSTEM_PROCESSOR AND NOT WIN32)
  execute_process(COMMAND "uname" "-m" OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86")
  set(IS_X86 TRUE)
else()
  set(IS_X86 FALSE)
endif()

if(ENABLE_VCPKG_INTEGRATION AND DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(X_VCPKG_APPLOCAL_DEPS_INSTALL ON)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  #set(_VCPKG_INSTALLED_DIR ${CMAKE_CURRENT_LIST_DIR}/vcpkg CACHE STRING "")  #folder for manifest-installed dependencies
  message(STATUS "VCPKG found: $ENV{VCPKG_ROOT}")
  message(STATUS "Using VCPKG integration")
  set(USE_INTEGRATED_LIBS "FALSE" CACHE BOOL "Use libs distributed with this repo")
  if(VCPKG_MANIFEST_FEATURES)
    message(STATUS "VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")
  endif()
elseif(DEFINED CMAKE_TOOLCHAIN_FILE)
  message(STATUS "Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
  if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake")
    message(STATUS "Toolchain uses VCPKG integration")
    if(VCPKG_MANIFEST_FEATURES)
      message(STATUS "VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")
    endif()
  endif()
  set(USE_INTEGRATED_LIBS "FALSE" CACHE BOOL "Use libs distributed with this repo")
elseif(WIN32)
  message(STATUS "vcpkg not found, toolchain not defined, using integrated libs on win32")
  set(USE_INTEGRATED_LIBS "TRUE" CACHE BOOL "Use libs distributed with this repo")
else()
  message(WARNING "vcpkg not found, toolchain not defined, system not win32 so build might fail")
  set(USE_INTEGRATED_LIBS "TRUE" CACHE BOOL "Use libs distributed with this repo")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json VCPKG_JSON_STRING)
  string(JSON VERSION_STRING GET ${VCPKG_JSON_STRING} version)
else()
  set(VERSION_STRING "0.2.5.4")
endif()

string(REPLACE "." ";" VERSION_LIST ${VERSION_STRING})
list(LENGTH VERSION_LIST VERSION_LIST_LENGTH)
if(VERSION_LIST_LENGTH LESS 3)
  message(FATAL_ERROR "Darknet needs at least major.minor.patch version numbers to properly configure")
endif()
list(GET VERSION_LIST 0 Darknet_MAJOR_VERSION)
list(GET VERSION_LIST 1 Darknet_MINOR_VERSION)
list(GET VERSION_LIST 2 Darknet_PATCH_VERSION)
if(VERSION_LIST_LENGTH GREATER 3)
  list(GET VERSION_LIST 3 Darknet_TWEAK_VERSION)
else()
  set(Darknet_TWEAK_VERSION 0)
endif()

set(Darknet_VERSION ${Darknet_MAJOR_VERSION}.${Darknet_MINOR_VERSION}.${Darknet_PATCH_VERSION}.${Darknet_TWEAK_VERSION})
message("Darknet_VERSION: ${Darknet_VERSION}")

project(Darknet VERSION ${Darknet_VERSION})

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
if(USE_INTEGRATED_LIBS)
  set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
endif()

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_C_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_COMPILER_IS_GNUCC_OR_CLANG TRUE)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "clang")
    set(CMAKE_COMPILER_IS_CLANG TRUE)
  else()
    set(CMAKE_COMPILER_IS_CLANG FALSE)
  endif()
else()
  set(CMAKE_COMPILER_IS_GNUCC_OR_CLANG FALSE)
  set(CMAKE_COMPILER_IS_CLANG FALSE)
endif()

cmake_dependent_option(ENABLE_SSE_AND_AVX_FLAGS "Enable AVX and SSE optimizations (x86-only)" ON "CMAKE_COMPILER_IS_GNUCC_OR_CLANG;IS_X86" OFF)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT ENABLE_INSTALLER)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
  endif()
  set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
  set(INSTALL_LIB_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where lib will be installed")
else()
  set(INSTALL_BIN_DIR      "bin"                       CACHE PATH "Path where exe and dll will be installed")
  set(INSTALL_LIB_DIR      "lib"                       CACHE PATH "Path where lib will be installed")
endif()

set(INSTALL_INCLUDE_DIR  "include/darknet"           CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/darknet"             CACHE PATH "Path where cmake configs will be installed")

find_library(MATH_LIBRARY m)

if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA_PATH: $ENV{CUDA_PATH}")
    message(STATUS "CUDACXX: $ENV{CUDACXX}")
    message(FATAL_ERROR "CUDA not found, please build explicitly with -DENABLE_CUDA=OFF if you do not want CUDA.")
  else()
    enable_language(CUDA)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
      message(STATUS "CUDA_PATH: $ENV{CUDA_PATH}")
      message(STATUS "CUDACXX: $ENV{CUDACXX}")
      message(FATAL_ERROR "Unsupported CUDA version, please upgrade to CUDA 9+ or disable CUDA with explicitly with -DENABLE_CUDA=OFF")
    else()
      if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23.0" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES all-major)
      endif()
      message(STATUS "Selected CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
      if("all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES OR
         "all" IN_LIST CMAKE_CUDA_ARCHITECTURES OR
         70 IN_LIST CMAKE_CUDA_ARCHITECTURES OR
         72 IN_LIST CMAKE_CUDA_ARCHITECTURES OR
         75 IN_LIST CMAKE_CUDA_ARCHITECTURES OR
         80 IN_LIST CMAKE_CUDA_ARCHITECTURES OR
         86 IN_LIST CMAKE_CUDA_ARCHITECTURES)
        set(ENABLE_CUDNN_HALF "TRUE" CACHE BOOL "Enable CUDNN Half precision" FORCE)
        message(STATUS "Your setup supports half precision (CUDA_ARCHITECTURES >= 70)")
      else()
        set(ENABLE_CUDNN_HALF "FALSE" CACHE BOOL "Enable CUDNN Half precision" FORCE)
        message(STATUS "Your setup does not support half precision (it requires CUDA_ARCHITECTURES >= 70)")
      endif()
    endif()
    if(BUILD_SHARED_LIBS)
      set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    else()
      set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    endif()
  endif()
endif()

if(WIN32 AND ENABLE_CUDA AND CMAKE_MAKE_PROGRAM MATCHES "ninja")
  option(SELECT_OPENCV_MODULES "Use only few selected OpenCV modules to circumvent 8192 char limit when using Ninja on Windows" ON)
else()
  option(SELECT_OPENCV_MODULES "Use only few selected OpenCV modules to circumvent 8192 char limit when using Ninja on Windows" OFF)
endif()

if(USE_INTEGRATED_LIBS)
  set(PThreads4W_ROOT ${CMAKE_CURRENT_LIST_DIR}/3rdparty/pthreads CACHE PATH "Path where pthreads for windows can be located")
  set(Stb_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb CACHE PATH "Path where Stb image library can be located")
endif()

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
if(MSVC)
  find_package(PThreads4W REQUIRED)
endif()
if(ENABLE_OPENCV)
  find_package(OpenCV REQUIRED)
  if(OpenCV_FOUND)
    if(SELECT_OPENCV_MODULES)
      if(TARGET opencv_world)
        list(APPEND OpenCV_LINKED_COMPONENTS "opencv_world")
      else()
        if(TARGET opencv_core)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_core")
        endif()
        if(TARGET opencv_highgui)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_highgui")
        endif()
        if(TARGET opencv_imgproc)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_imgproc")
        endif()
        if(TARGET opencv_video)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_video")
        endif()
        if(TARGET opencv_videoio)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_videoio")
        endif()
        if(TARGET opencv_imgcodecs)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_imgcodecs")
        endif()
        if(TARGET opencv_text)
          list(APPEND OpenCV_LINKED_COMPONENTS "opencv_text")
        endif()
      endif()
    else()
      list(APPEND OpenCV_LINKED_COMPONENTS ${OpenCV_LIBS})
    endif()
  endif()
endif()
find_package(Stb REQUIRED)
find_package(OpenMP)

if(APPLE AND NOT OPENMP_FOUND)
  message(STATUS "  ->  To enable OpenMP on macOS, please install libomp from Homebrew")
endif()

set(ADDITIONAL_CXX_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")
set(ADDITIONAL_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")
if(UNIX AND BUILD_SHARED_LIBS AND NOT CMAKE_COMPILER_IS_CLANG)
  set(SHAREDLIB_CXX_FLAGS "-Wl,-Bsymbolic")
  set(SHAREDLIB_C_FLAGS "-Wl,-Bsymbolic")
endif()

if(MSVC)
  set(ADDITIONAL_CXX_FLAGS " /nologo /wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
  set(ADDITIONAL_C_FLAGS " /nologo /wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
endif()

if(CMAKE_COMPILER_IS_GNUCC_OR_CLANG)
  if(CMAKE_COMPILER_IS_CLANG)
    if(UNIX AND NOT APPLE)
      set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")  #force pthread to avoid bugs in some cmake setups
      set(CMAKE_C_FLAGS "-pthread ${CMAKE_C_FLAGS}")
    endif()
  endif()
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REGEX REPLACE "-O0" "-Og" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  if(ENABLE_SSE_AND_AVX_FLAGS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} ${SHAREDLIB_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${ADDITIONAL_C_FLAGS} ${SHAREDLIB_C_FLAGS} ${CMAKE_C_FLAGS}")

if(OpenCV_FOUND)
  if(ENABLE_CUDA AND OpenCV_CUDA_VERSION)
    if(TARGET opencv_cudaoptflow)
      list(APPEND OpenCV_LINKED_COMPONENTS "opencv_cudaoptflow")
    endif()
    if(TARGET opencv_cudaimgproc)
      list(APPEND OpenCV_LINKED_COMPONENTS "opencv_cudaimgproc")
    endif()
  elseif(ENABLE_CUDA AND NOT OpenCV_CUDA_VERSION)
    set(BUILD_USELIB_TRACK "FALSE" CACHE BOOL "Build uselib_track" FORCE)
    message(STATUS "  ->  darknet is fine for now, but uselib_track has been disabled!")
    message(STATUS "  ->  Please rebuild OpenCV from sources with CUDA support to enable it")
  else()
    set(BUILD_USELIB_TRACK "FALSE" CACHE BOOL "Build uselib_track" FORCE)
  endif()
endif()

if(ENABLE_CUDA AND ENABLE_CUDNN)
  find_package(CUDNN REQUIRED)
endif()

if(ENABLE_CUDA)
  if(MSVC)
    set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DGPU")

    if(ENABLE_CUDA_OPENGL_INTEGRATION)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DCUDA_OPENGL_INTEGRATION")
    endif()

    if(CUDNN_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DCUDNN")
    endif()
    if(OpenCV_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DOPENCV")
    endif()
    string(REPLACE " " "," ADDITIONAL_CXX_FLAGS_COMMA_SEPARATED "${ADDITIONAL_CXX_FLAGS}")
    set(CUDA_HOST_COMPILER_FLAGS "-Wno-deprecated-declarations -Xcompiler=\"${ADDITIONAL_CXX_FLAGS_COMMA_SEPARATED}\"")
  else()
    set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DGPU")

    if(ENABLE_CUDA_OPENGL_INTEGRATION)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DCUDA_OPENGL_INTEGRATION")
    endif()

    if(CUDNN_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DCUDNN")
    endif()
    if(OpenCV_FOUND)
      set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DOPENCV")
    endif()
    if(APPLE)
        set(CUDA_HOST_COMPILER_FLAGS "--compiler-options \" ${ADDITIONAL_CXX_FLAGS} -fPIC -Xpreprocessor -fopenmp -Ofast \"")
    else()
        set(CUDA_HOST_COMPILER_FLAGS "--compiler-options \" ${ADDITIONAL_CXX_FLAGS} -fPIC -fopenmp -Ofast \"")
    endif()
  endif()

  string (REPLACE ";" " " CUDA_ARCH_FLAGS_SPACE_SEPARATED "${CUDA_ARCH_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS_SPACE_SEPARATED} ${CUDA_HOST_COMPILER_FLAGS} ${CMAKE_CUDA_FLAGS}")
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

if(ENABLE_CUDA AND ENABLE_ZED_CAMERA)
  find_package(ZED 2 QUIET)
  if(ZED_FOUND)
    include_directories(${ZED_INCLUDE_DIRS})
    link_directories(${ZED_LIBRARY_DIR})
    message(STATUS "ZED SDK enabled")
  else()
    message(STATUS "ZED SDK not found")
    set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
  endif()
else()
  if(ENABLE_ZED_CAMERA)
    message(STATUS "ZED SDK not enabled, since it requires CUDA")
  endif()
  set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
endif()

foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(FULLPATH_${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/src/version.h.in"
  "${CMAKE_CURRENT_LIST_DIR}/src/version.h"
)

#look for all *.h files in src folder
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/src/*.h")
#add also files in the include folder
list(APPEND headers
  ${CMAKE_CURRENT_LIST_DIR}/include/darknet.h
)
#remove windows only files
if(NOT MSVC)
  list(REMOVE_ITEM headers
    ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.h
  )
endif()
#set(exported_headers ${headers})

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
#remove windows only files
if(NOT MSVC)
  list(REMOVE_ITEM sources
    ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.c
  )
endif()

if(USE_INTEGRATED_LIBS AND MSVC)
  list(APPEND sources
    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/getopt/getopt.c
    ${CMAKE_CURRENT_LIST_DIR}/3rdparty/getopt/getopt.h
  )
endif()
if((NOT USE_INTEGRATED_LIBS) AND MSVC)
  find_package(unofficial-getopt-win32 REQUIRED)
endif()

if(ENABLE_CUDA)
  file(GLOB cuda_sources "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
endif()

if(BUILD_AS_CPP)
  set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
endif()

add_library(dark ${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp ${CMAKE_CURRENT_LIST_DIR}/src/yolo_v2_class.cpp ${sources} ${headers} ${cuda_sources})
set_target_properties(dark PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  set_target_properties(dark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
if(BUILD_SHARED_LIBS)
  target_compile_definitions(dark PRIVATE LIB_EXPORTS=1)
endif()
if(BUILD_AS_CPP)
  set_target_properties(dark PROPERTIES LINKER_LANGUAGE CXX)
endif()
set_target_properties(dark PROPERTIES OUTPUT_NAME "darknet")

if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
  add_executable(uselib_track ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
endif()

add_executable(uselib ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
if(BUILD_AS_CPP)
  set_target_properties(uselib PROPERTIES LINKER_LANGUAGE CXX)
endif()

add_executable(darknet ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c ${sources} ${headers} ${cuda_sources})
if(BUILD_AS_CPP)
  set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/darknet.c PROPERTIES LANGUAGE CXX)
  set_target_properties(darknet PROPERTIES LINKER_LANGUAGE CXX)
endif()
if(MSVC)
  target_sources(darknet PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/darknet.rc)
endif()

add_executable(kmeansiou ${CMAKE_CURRENT_LIST_DIR}/scripts/kmeansiou.c)
if(MATH_LIBRARY)
  target_link_libraries(kmeansiou PRIVATE ${MATH_LIBRARY})
endif()

target_include_directories(darknet PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
target_include_directories(dark PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
target_include_directories(uselib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

target_compile_definitions(darknet PRIVATE -DUSE_CMAKE_LIBS)
target_compile_definitions(dark PRIVATE -DUSE_CMAKE_LIBS)
target_compile_definitions(uselib PRIVATE -DUSE_CMAKE_LIBS)

if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK AND NOT MANUALLY_EXPORT_TRACK_OPTFLOW)
  target_compile_definitions(dark PUBLIC TRACK_OPTFLOW=1)
endif()

if(CUDNN_FOUND)
  target_link_libraries(darknet PRIVATE CuDNN::CuDNN)
  target_link_libraries(dark PRIVATE CuDNN::CuDNN)
  target_compile_definitions(darknet PRIVATE -DCUDNN)
  target_compile_definitions(dark PUBLIC -DCUDNN)
  if(ENABLE_CUDNN_HALF)
    target_compile_definitions(darknet PRIVATE -DCUDNN_HALF)
    target_compile_definitions(dark PUBLIC -DCUDNN_HALF)
  endif()
endif()

if(OpenCV_FOUND)
  target_link_libraries(darknet PRIVATE ${OpenCV_LINKED_COMPONENTS})
  target_link_libraries(uselib PRIVATE ${OpenCV_LINKED_COMPONENTS})
  target_link_libraries(dark PUBLIC ${OpenCV_LINKED_COMPONENTS})
  target_include_directories(dark PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(darknet PRIVATE -DOPENCV)
  target_compile_definitions(dark PUBLIC -DOPENCV)
endif()

if(OPENMP_FOUND)
  target_link_libraries(darknet PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(darknet PRIVATE OpenMP::OpenMP_C)
  target_link_libraries(dark PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(dark PUBLIC OpenMP::OpenMP_C)
endif()

if(CMAKE_COMPILER_IS_GNUCC AND MATH_LIBRARY)
  target_link_libraries(darknet PRIVATE ${MATH_LIBRARY})
  target_link_libraries(dark PUBLIC ${MATH_LIBRARY})
endif()

if(MSVC)
  target_link_libraries(darknet PRIVATE PThreads4W::PThreads4W)
  target_link_libraries(darknet PRIVATE wsock32)
  target_link_libraries(dark PUBLIC PThreads4W::PThreads4W)
  target_link_libraries(dark PUBLIC wsock32)
  target_link_libraries(uselib PRIVATE PThreads4W::PThreads4W)
  if(USE_INTEGRATED_LIBS)
    target_include_directories(dark PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdparty/getopt)
    target_include_directories(darknet PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdparty/getopt)
  else()
    target_link_libraries(dark PRIVATE unofficial::getopt-win32::getopt)
    target_link_libraries(darknet PRIVATE unofficial::getopt-win32::getopt)
  endif()
  target_compile_definitions(darknet PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
  target_compile_definitions(dark PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
  target_compile_definitions(dark PUBLIC -D_CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(uselib PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
endif()

if(MSVC OR MINGW)
  target_link_libraries(darknet PRIVATE ws2_32)
  target_link_libraries(dark PUBLIC ws2_32)
endif()

target_link_libraries(darknet PRIVATE Threads::Threads)
target_link_libraries(dark PUBLIC Threads::Threads)
target_link_libraries(uselib PRIVATE Threads::Threads)

if(ENABLE_ZED_CAMERA)
  target_link_libraries(darknet PRIVATE ${ZED_LIBRARIES})
  target_link_libraries(dark PUBLIC ${ZED_LIBRARIES})
  target_link_libraries(uselib PRIVATE ${ZED_LIBRARIES})
  target_compile_definitions(darknet PRIVATE -DZED_STEREO)
  target_compile_definitions(uselib PRIVATE -DZED_STEREO)
  target_compile_definitions(dark PUBLIC -DZED_STEREO)
endif()

if(ENABLE_CUDA)
  target_include_directories(darknet PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(dark PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(darknet PRIVATE curand cublas cuda)
  target_link_libraries(dark PRIVATE curand cublas cuda)
  set_target_properties(dark PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_definitions(darknet PRIVATE -DGPU)
  target_compile_definitions(dark PUBLIC -DGPU)
endif()

if(ENABLE_CUDA_OPENGL_INTEGRATION)
  target_compile_definitions(darknet PRIVATE -DCUDA_OPENGL_INTEGRATION)
  target_compile_definitions(dark PUBLIC -DCUDA_OPENGL_INTEGRATION)
endif()

if(USE_INTEGRATED_LIBS AND WIN32)
  target_compile_definitions(darknet PRIVATE -D_TIMESPEC_DEFINED)
  target_compile_definitions(dark PRIVATE -D_TIMESPEC_DEFINED)
endif()

target_link_libraries(uselib PRIVATE dark)
if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
  target_link_libraries(uselib_track PRIVATE dark)
  target_compile_definitions(uselib_track PRIVATE TRACK_OPTFLOW=1)
  target_compile_definitions(uselib_track PRIVATE -DUSE_CMAKE_LIBS)
  if(BUILD_AS_CPP)
    set_target_properties(uselib_track PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  target_include_directories(uselib_track PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
  target_link_libraries(uselib_track PRIVATE ${OpenCV_LINKED_COMPONENTS})
  if(ENABLE_ZED_CAMERA)
    target_link_libraries(uselib_track PRIVATE ${ZED_LIBRARIES})
    target_compile_definitions(uselib_track PRIVATE -DZED_STEREO)
  endif()
  if(MSVC)
    target_link_libraries(uselib_track PRIVATE PThreads4W::PThreads4W)
    target_compile_definitions(uselib_track PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
  endif()
  target_link_libraries(uselib_track PRIVATE Threads::Threads)
endif()

#set_target_properties(dark PROPERTIES PUBLIC_HEADER "${exported_headers};${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp")
set_target_properties(dark PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_LIST_DIR}/include/darknet.h;${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp")

set_target_properties(dark PROPERTIES CXX_VISIBILITY_PRESET hidden)

install(TARGETS dark EXPORT DarknetTargets
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
  COMPONENT dev
)
install(TARGETS uselib darknet kmeansiou
  DESTINATION "${INSTALL_BIN_DIR}"
)
if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
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
file(RELATIVE_PATH REL_INCLUDE_DIR "${FULLPATH_INSTALL_CMAKE_DIR}" "${FULLPATH_INSTALL_INCLUDE_DIR}")
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

if (ENABLE_DEPLOY_CUSTOM_CMAKE_MODULES)
  install(FILES
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/FindCUDNN.cmake"
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/FindPThreads4W.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}"
  )
endif()

if(ENABLE_CSHARP_WRAPPER)
  add_subdirectory(src/csharp)
endif()

set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
include(InstallRequiredSystemLibraries)

install(
    PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
    DESTINATION ${INSTALL_BIN_DIR}
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json VCPKG_JSON_STRING)
  string(JSON CPACK_PACKAGE_NAME GET ${VCPKG_JSON_STRING} name)
  string(JSON CPACK_PACKAGE_HOMEPAGE_URL GET ${VCPKG_JSON_STRING} homepage)
  string(JSON CPACK_PACKAGE_DESCRIPTION GET ${VCPKG_JSON_STRING} description)
  set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)

  if(UNIX AND NOT APPLE)
    find_program(LSB_RELEASE_EXEC lsb_release)
    execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
        OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(LSB_RELEASE_ID_SHORT STREQUAL "Ubuntu")
      set(CPACK_GENERATOR "DEB")
      set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Darknet")
    else()
      set(CPACK_GENERATOR "RPM")
    endif()
  elseif(APPLE)
      set(CPACK_GENERATOR "DragNDrop")
  elseif(WIN32)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
    if(USE_NSIS)
      set(CPACK_GENERATOR "NSIS")
      string(JSON CPACK_NSIS_PACKAGE_NAME GET ${VCPKG_JSON_STRING} name)
      string(JSON CPACK_NSIS_DISPLAY_NAME GET ${VCPKG_JSON_STRING} name)
      set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL "ON")
      set(CPACK_NSIS_MODIFY_PATH OFF) #disable extra page for adding to PATH, because it's broken on Win10+ due to NSIS not supporting MAX_PATH
      set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/src/darknet.ico")
      set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/src/darknet.ico")
    else()
      set(CPACK_GENERATOR "WIX")
      #set(CPACK_WIX_UPGRADE_GUID "") # IMPORTANT! It has to be unique for every project!!
    endif()
  endif()

  include(CPack)
endif()
