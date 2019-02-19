# To use your libs, launch cmake with -DUSE_INTEGRATED_LIBS:BOOL=FALSE
# To enable CUDA, launch cmake with -DENABLE_CUDA. Also it is advisable to set the highest compute model with -DCOMPUTE_MODEL=..., depending on your specific card capabilities. If CUDA is not required, this setting is completely ignored.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#set(CMAKE_VERBOSE_MAKEFILE 1)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(darknet)

if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(USE_INTEGRATED_LIBS "TRUE" CACHE BOOL "Use libs distributed with this repo")
else()
  set(USE_INTEGRATED_LIBS "FALSE" CACHE BOOL "Use libs distributed with this repo")
endif()

set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable building with CUDA")

enable_language(C)
enable_language(CXX)
if(ENABLE_CUDA)
  enable_language(CUDA) # do not enable after searching for OpenCV package!
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

set(COMPUTE_MODEL_75 "75") #Compute capability for Tesla T4, RTX 2080
set(COMPUTE_MODEL_72 "72") #Compute capability for Jetson Xavier
set(COMPUTE_MODEL_70 "70") #Compute capability for Tesla V100
set(COMPUTE_MODEL_62 "62") #Compute capability for Jetson TX2
set(COMPUTE_MODEL_61 "61") #Compute capability for Tesla P40
set(COMPUTE_MODEL_60 "60") #Compute capability for Tesla P100
set(COMPUTE_MODEL_53 "53") #Compute capability for Jetson TX1
set(COMPUTE_MODEL_52 "52") #Compute capability for Tesla M40/M60
set(COMPUTE_MODEL_37 "37") #Compute capability for Tesla K80
set(COMPUTE_MODEL_35 "35") #Compute capability for Tesla K20/K40
set(COMPUTE_MODEL_30 "30") #Compute capability for Tesla K10, Quadro K4000

set(default_compute_model ${COMPUTE_MODEL_30})
if(NOT COMPUTE_MODEL)
  message(STATUS "Setting the compute model to '${default_compute_model}' as none was specified.")
  set(COMPUTE_MODEL "${default_compute_model}" CACHE
      STRING "Choose the best compute model supported by your GPU" FORCE)
  set_property(CACHE COMPUTE_MODEL PROPERTY STRINGS ${COMPUTE_MODEL_75} ${COMPUTE_MODEL_72} ${COMPUTE_MODEL_70} ${COMPUTE_MODEL_62} ${COMPUTE_MODEL_61} ${COMPUTE_MODEL_60} ${COMPUTE_MODEL_53} ${COMPUTE_MODEL_52} ${COMPUTE_MODEL_37} ${COMPUTE_MODEL_35} ${COMPUTE_MODEL_30})
endif()

if(USE_INTEGRATED_LIBS)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty/include)
  set(PTHREADS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/include)
  set(PTHREADS_LIBRARY ${CMAKE_CURRENT_LIST_DIR}/3rdparty/lib/x64/pthreadVC2.lib)
  add_definitions(-D_TIMESPEC_DEFINED)
endif()

if(NOT APPLE)
  set(CMAKE_SKIP_BUILD_RPATH  FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}")

add_definitions(-DUSE_CMAKE_LIBS)

if(MSVC)
  set(CMAKE_C_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /fp:fast ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /fp:fast ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  add_definitions(-D_CRT_RAND_S)
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_C_FLAGS "-Wno-unused-result -Wno-discarded-qualifiers -Wno-incompatible-pointer-types ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Wno-unused-result -Wno-write-strings ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-O0" "-Og" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "-Wno-deprecated -Wno-writable-strings ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-O0" "-Og" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
endif()

if(NOT USE_INTEGRATED_LIBS)
  if(APPLE)
    find_package(OpenCV PATHS "/usr/local/Cellar/opencv@3/3.4.5" NO_DEFAULT_PATH)
  else()
    find_package(OpenCV)
  endif()
endif()

if(OpenCV_FOUND)
  add_definitions(-DOPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(ENABLE_CUDA)
  add_definitions(-DGPU)
  if(COMPUTE_MODEL VERSION_GREATER_EQUAL 75 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL 9)
    message(STATUS "CUDA 10 is required for Compute Model 7.5 - Lowering settings to 7.0")
    set(COMPUTE_MODEL "70")
  endif()
  find_package(CUDNN)
  if(CUDNN_FOUND)
    add_definitions(-DCUDNN)
    include_directories(${CUDNN_INCLUDE_DIR})
    if(COMPUTE_MODEL VERSION_GREATER_EQUAL 53)
      add_definitions(-DCUDNN_HALF)
      message(STATUS "Enabled CUDNN_HALF")
    endif()
  endif()
  if (MSVC)
    if(CUDNN_FOUND)
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/DGPU,/DCUDNN,/DOPENCV\" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/DGPU,/DCUDNN\" ${CMAKE_CUDA_FLAGS}")
      endif()
    else()
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/DGPU,/DOPENCV\" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] -Wno-deprecated-declarations -Xcompiler=\"/wd4028,/wd4190,/wd4244,/wd4267,/wd4305,/wd4477,/wd4996,/DGPU\" ${CMAKE_CUDA_FLAGS}")
      endif()
    endif()
  else()
    if(CUDNN_FOUND)
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DOPENCV -DGPU -DCUDNN \" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DGPU -DCUDNN \" ${CMAKE_CUDA_FLAGS}")
      endif()
    else()
      if(OpenCV_FOUND)
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DOPENCV -DGPU \" ${CMAKE_CUDA_FLAGS}")
      else()
        set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${COMPUTE_MODEL},code=[sm_${COMPUTE_MODEL},compute_${COMPUTE_MODEL}] --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -fPIC -fopenmp -Ofast -DGPU \" ${CMAKE_CUDA_FLAGS}")
      endif()
    endif()
  endif()
endif()

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
if(MSVC)
  find_package(PThreads_windows REQUIRED)
  include_directories(${PTHREADS_INCLUDE_DIR})
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#look for all *.h files in src folder
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/src/*.h")
#add also files in the include folder
list(APPEND sources
  ${CMAKE_CURRENT_LIST_DIR}/include/darknet.h
)
#remove windows only files, will be re-added if necessary later
list(REMOVE_ITEM headers
  ${CMAKE_CURRENT_LIST_DIR}/src/gettimeofday.h
  ${CMAKE_CURRENT_LIST_DIR}/src/getopt.h
)

#look for all *.c files in src folder
file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
#add also .cpp files
list(APPEND sources
  ${CMAKE_CURRENT_LIST_DIR}/src/http_stream.cpp
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

add_library(darklib SHARED ${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp ${CMAKE_CURRENT_LIST_DIR}/src/yolo_v2_class.cpp ${sources} ${headers} ${cuda_sources})
set_target_properties(darklib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ENABLE_CUDA)
  set_target_properties(darklib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_compile_definitions(darklib PRIVATE LIB_EXPORTS=1)
set_target_properties(darklib PROPERTIES LINKER_LANGUAGE CXX)

if(OpenCV_VERSION_MAJOR GREATER_EQUAL 3)
  add_executable(uselib_track ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
  target_compile_definitions(uselib_track PRIVATE TRACK_OPTFLOW=1)
  set_target_properties(uselib_track PROPERTIES LINKER_LANGUAGE CXX)
endif()

add_executable(uselib ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
set_target_properties(uselib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(darknet ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c ${sources} ${headers} ${cuda_sources})
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/darknet.c PROPERTIES LANGUAGE CXX)
set_target_properties(darknet PROPERTIES LINKER_LANGUAGE CXX)

if(CUDNN_FOUND)
  target_link_libraries(darknet ${CUDNN_LIBRARY})
  target_link_libraries(darklib ${CUDNN_LIBRARY})
endif()

if(OpenCV_FOUND)
  target_link_libraries(darknet ${OpenCV_LIBS})
  target_link_libraries(darklib ${OpenCV_LIBS})
endif()

if(OPENMP_FOUND)
  target_link_libraries(darknet OpenMP::OpenMP_C)
  target_link_libraries(darklib OpenMP::OpenMP_C)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(darknet m)
  target_link_libraries(darklib m)
endif()

if(MSVC)
  target_link_libraries(darknet ${PTHREADS_LIBRARY})
  target_link_libraries(darknet wsock32 ws2_32)
  target_link_libraries(darklib ${PTHREADS_LIBRARY})
  target_link_libraries(darklib wsock32 ws2_32)
endif()

target_link_libraries(darknet Threads::Threads)
target_link_libraries(darklib Threads::Threads)

if(ENABLE_CUDA)
  target_link_libraries(darknet curand cublas)
  target_link_libraries(darklib curand cublas)
endif()

target_link_libraries(uselib darklib)
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 3)
  target_link_libraries(uselib_track darklib)
  install(TARGETS uselib_track DESTINATION ${CMAKE_SOURCE_DIR}/)
endif()

install(TARGETS darklib uselib darknet DESTINATION ${CMAKE_SOURCE_DIR}/)
