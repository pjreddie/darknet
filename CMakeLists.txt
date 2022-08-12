cmake_minimum_required(VERSION 3.5)
project(darknet C)

set(GPU 1)
set(CUDNN 0)
set(OPENCV 0)
set(OPENMP 0)
set(DEBUG 1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(DEBUG 1)

include_directories(src include)

set (OPTS -Ofast)

if ( DEBUG )
   set(OPTS ${OPTS} -O0 -g )
endif()

set ( NVCC nvcc )
set ( AR ar )
set ( ARFLAGS rcs )
set ( LDFLAGS -lm -pthread )
set ( CFLAGS ${OPTS} -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC)

if ( OPENMP )
   set(CFLAGS ${CFLAGS} -fopenmp )
endif()

if ( OPENCV )
   set(COMMON ${COMMON} -DOPENCV )
   set(CFLAGS ${CFLAGS} -DOPENCV )
   set(LDFLAGS ${LDFLAGS} -L${env.OPENCV_HOME}/lib -lopencv_core  -lstdc++)
endif()

if ( GPU )
   set(COMMON ${COMMON} -DGPU -I/usr/local/cuda/include/ )
   set(CFLAGS ${CFLAGS} -DGPU )
   set(LDFLAGS ${LDFLAGS} -lstdc++ -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand)
endif()

if ( CUDNN )
   set(COMMON ${COMMON} -DCUDNN )
   set(CFLAGS ${CFLAGS} -DCUDNN )
   set(LDFLAGS ${LDFLAGS} -lcudnn)
endif()

add_subdirectory(src)
add_subdirectory(examples)

