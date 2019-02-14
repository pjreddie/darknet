cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#set(CMAKE_VERBOSE_MAKEFILE 1)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(darknet)

enable_language(C)
enable_language(CXX)
if(ENABLE_CUDA)
  enable_language(CUDA) # do not enable after searching for OpenCV package!
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

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

if(APPLE)
  find_package(OpenCV PATHS "/usr/local/Cellar/opencv@3/3.4.5" NO_DEFAULT_PATH)
else()
  find_package(OpenCV)
endif()

if(OpenCV_FOUND)
  add_definitions(-DOPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(ENABLE_CUDA)
  add_definitions(-DGPU)
  #set(COMPUTE_MODEL "75")  #Tesla T4, RTX 2080
  #set(COMPUTE_MODEL "72")  #Jetson Xavier
  #set(COMPUTE_MODEL "70")  #Tesla V100
  #set(COMPUTE_MODEL "62")  #Jetson TX2
  #set(COMPUTE_MODEL "61")  #Telsa P40
  #set(COMPUTE_MODEL "60")  #Telsa P100
  #set(COMPUTE_MODEL "53")  #Jetson TX1
  #set(COMPUTE_MODEL "52")  #Tesla M40/M60
  #set(COMPUTE_MODEL "37")  #Tesla K80
  #set(COMPUTE_MODEL "35")  #Tesla K20/K40
  set(COMPUTE_MODEL "30")  #Tesla K10, Quadro K4000
  if(COMPUTE_MODEL VERSION_GREATER_EQUAL 53)
    add_definitions(-DCUDNN_HALF)
  endif()
  find_package(CUDNN)
  if(CUDNN_FOUND)
    add_definitions(-DCUDNN)
    include_directories(${CUDNN_INCLUDE_DIR})
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
