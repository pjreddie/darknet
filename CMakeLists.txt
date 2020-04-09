CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(darknet)
# C++和CUDA的编译参数，可选。
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -std=c++11")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=[sm_50,compute_50] -gencode arch=compute_52,code=[sm_52,compute_52];-std=c++11;)
# 头文件路径，按需
include_directories(
        ./src
        ./include
)
# 库文件路径，按需
link_directories(/usr/lib
        /usr/local/lib)
# 主要就是这个，教cmake去找nvcc来编译这些东西

set(darknet_lib libDarkNet)
option(BUILD_SHARED_LIBS "BUILD_SHARED_LIBS" ON)
if (BUILD_SHARED_LIBS)
    set(darknet_LIB_TYPE SHARED)
else ()
    set(darknet_LIB_TYPE STATIC)
endif ()

FILE(GLOB C_SrcSource "src/*.c")
list(FILTER C_SrcSource EXCLUDE REGEX ".*compare.c")

FILE(GLOB CU_SrcSource "src/*.cu")

option(ENABLE_OPENCV "option for OpenCV" OFF)
if (ENABLE_OPENCV)
    find_package(OpenCV)
    if (OpenCV_FOUND)
        set(ENABLE_OPENCV ON)
        add_definitions(-DENABLE_OPENCV -DOPENCV)
        message(STATUS "OpenCV library status:")
        message(STATUS "    version: ${OpenCV_VERSION}")
        message(STATUS "    libraries: ${OpenCV_LIBS}")
        message(STATUS "    libraries: ${OpenCV_LIBRARIES}")
        message(STATUS "    lib_dir: ${OpenCV_LIB_DIR}")
        message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

        link_directories(${OpenCV_DIR})
        include_directories(
                ${OpenCV_INCLUDE_DIRS}
        )
    endif ()
endif ()


find_package(CUDA)
option(ENABLE_CUDA "option for CUDA" OFF)
if (ENABLE_CUDA AND CUDA_FOUND)
    add_definitions(-DENABLE_CUDA -DCUDNN -DGPU)

    include_directories(${CUDA_INCLUDE_DIRS})
    if ("${Tools_Other_Project}" STREQUAL "ON")
        message(STATUS "CUDA library status:")
        message(STATUS "    ${CUDA_VERSION}")
        message(STATUS "    libraries: ${CUDA_LIBS}")
        message(STATUS "    libraries: ${CUDA_LIBRARIES}")
        message(STATUS "    lib_dir: ${CUDA_LIBRARY_DIRS}")
        message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
    endif ()
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})

    cuda_add_library(${darknet_lib} ${darknet_LIB_TYPE}
            ${C_SrcSource}
            ${CU_SrcSource}
            )
    # 链接外部库，按需
    target_link_libraries(${darknet_lib} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
else ()
    add_library(${darknet_lib} ${darknet_LIB_TYPE}
            ${C_SrcSource}
            )
    # 链接外部库，按需
    target_link_libraries(${darknet_lib} ${OpenCV_LIBRARIES})
endif ()


FILE(GLOB example_SrcSource "examples/*.c")
list(FILTER example_SrcSource EXCLUDE REGEX ".*attention.c")
list(FILTER example_SrcSource EXCLUDE REGEX ".*dice.c")
list(FILTER example_SrcSource EXCLUDE REGEX ".*swag.c")
list(FILTER example_SrcSource EXCLUDE REGEX ".*writing.c")
list(FILTER example_SrcSource EXCLUDE REGEX ".*voxel.c")
add_executable(darknet ${example_SrcSource})
target_link_libraries(darknet ${darknet_lib})


