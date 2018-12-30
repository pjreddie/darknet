cmake_minimum_required(VERSION 3.8)

set(GPU ON)
set(CUDNN ON)
set(OPENCV ON)

#add_definitions(-DOPENMP)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0071 NEW)
cmake_policy(SET CMP0046 NEW)

project(darknet)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green       "${Esc}[32m")

message("${Green}-- Project build-type: ${CMAKE_BUILD_TYPE}${ColourReset}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_COMPILER /usr/bin/gcc)
# enables all -O3 optimizations

add_definitions(-Ofast)
add_definitions(-Wno-unused-result)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_SOURCE_DIR}/install)

include(CMakeParseArguments)

# gather_flags
# Gathers all lists of flags for printing or manipulation
macro (gather_flags with_linker result)
    set(${result} "")
    # add the main flags without a config
    list(APPEND ${result} CMAKE_C_FLAGS)
    list(APPEND ${result} CMAKE_CXX_FLAGS)
    if(${with_linker})
        list(APPEND ${result} CMAKE_EXE_LINKER_FLAGS)
        list(APPEND ${result} CMAKE_MODULE_LINKER_FLAGS)
        list(APPEND ${result} CMAKE_SHARED_LINKER_FLAGS)
        list(APPEND ${result} CMAKE_STATIC_LINKER_FLAGS)
    endif()

    if("${CMAKE_CONFIGURATION_TYPES}" STREQUAL "" AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
        # handle single config generators - like makefiles/ninja - when CMAKE_BUILD_TYPE is set
        string(TOUPPER ${CMAKE_BUILD_TYPE} config)
        list(APPEND ${result} CMAKE_C_FLAGS_${config})
        list(APPEND ${result} CMAKE_CXX_FLAGS_${config})
        if(${with_linker})
            list(APPEND ${result} CMAKE_EXE_LINKER_FLAGS_${config})
            list(APPEND ${result} CMAKE_MODULE_LINKER_FLAGS_${config})
            list(APPEND ${result} CMAKE_SHARED_LINKER_FLAGS_${config})
            list(APPEND ${result} CMAKE_STATIC_LINKER_FLAGS_${config})
        endif()
    else()
        # handle multi config generators (like msvc, xcode)
        foreach(config ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${config} config)
            list(APPEND ${result} CMAKE_C_FLAGS_${config})
            list(APPEND ${result} CMAKE_CXX_FLAGS_${config})
            if(${with_linker})
                list(APPEND ${result} CMAKE_EXE_LINKER_FLAGS_${config})
                list(APPEND ${result} CMAKE_MODULE_LINKER_FLAGS_${config})
                list(APPEND ${result} CMAKE_SHARED_LINKER_FLAGS_${config})
                list(APPEND ${result} CMAKE_STATIC_LINKER_FLAGS_${config})
            endif()
        endforeach()
    endif()
endmacro()

# set_runtime
# Sets the runtime (static/dynamic) for msvc/gcc
macro (set_runtime)
    cmake_parse_arguments(ARG "STATIC;DYNAMIC" "" "" ${ARGN})

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" STREQUAL "")
        message(AUTHOR_WARNING "set_runtime() does not support clang yet!")
    endif()

    gather_flags(0 flags_configs)

    # add/replace the flags
    # note that if the user has messed with the flags directly this function might fail
    # - for example if with MSVC and the user has removed the flags - here we just switch/replace them
    if("${ARG_STATIC}")
        foreach(flags ${flags_configs})
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                if(NOT ${flags} MATCHES "-static-libstdc\\+\\+")
                    set(${flags} "${${flags}} -static-libstdc++")
                endif()
                if(NOT ${flags} MATCHES "-static-libgcc")
                    set(${flags} "${${flags}} -static-libgcc")
                endif()
            elseif(MSVC)
                if(${flags} MATCHES "/MD")
                    string(REGEX REPLACE "/MD" "/MT" ${flags} "${${flags}}")
                endif()
            endif()
        endforeach()
    elseif("${ARG_DYNAMIC}")
        foreach(flags ${flags_configs})
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                if(${flags} MATCHES "-static-libstdc\\+\\+")
                    string(REGEX REPLACE "-static-libstdc\\+\\+" "" ${flags} "${${flags}}")
                endif()
                if(${flags} MATCHES "-static-libgcc")
                    string(REGEX REPLACE "-static-libgcc" "" ${flags} "${${flags}}")
                endif()
            elseif(MSVC)
                if(${flags} MATCHES "/MT")
                    string(REGEX REPLACE "/MT" "/MD" ${flags} "${${flags}}")
                endif()
            endif()
        endforeach()
    endif()

    #    foreach(flags ${flags_configs})
    #        message(STATUS "${flags}: ${${flags}}")
    #    endforeach()
endmacro()

set(CompilerFlags
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
)

foreach(CompilerFlag ${CompilerFlags})
    string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
endforeach()

file(GLOB C_SOURCES "src/*.c" "src/*.h")
file(GLOB CPP_SOURCES "src/*.cpp")
file(GLOB APP_SOURCES "examples/*.c")
file(GLOB CUDA_SOURCES "src/*.cu")

list(REMOVE_ITEM C_SOURCES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/compare.c)

list(REMOVE_ITEM APP_SOURCES 
	${CMAKE_CURRENT_SOURCE_DIR}/examples/attention.c
	${CMAKE_CURRENT_SOURCE_DIR}/examples/rnn_vid.c
	${CMAKE_CURRENT_SOURCE_DIR}/examples/swag.c
	${CMAKE_CURRENT_SOURCE_DIR}/examples/dice.c
	${CMAKE_CURRENT_SOURCE_DIR}/examples/voxel.c
	${CMAKE_CURRENT_SOURCE_DIR}/examples/writing.c
	${CMAKE_CURRENT_SOURCE_DIR}/examples/darknet.c
	)

if (GPU AND CUDNN)
    add_definitions(-DCUDNN)
    include(FindCUDNN.cmake)
    add_definitions(${CUDNN_DEFINITIONS})
    include_directories(${CUDNN_INCLUDE_DIRS})
endif()

if (GPU)
    add_definitions(-DGPU)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (OPENCV)
    add_definitions(-DOPENCV)
    set(OpenCV_STATIC ON)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if (OPENCV AND GPU)
    set(OpenCV_CUDA ON)
endif()

set_runtime(STATIC)

add_library(darknet_cpp STATIC ${CPP_SOURCES})
add_library(darknet_app STATIC ${APP_SOURCES})

add_library(${PROJECT_NAME} SHARED ${C_SOURCES})

target_link_libraries(${PROJECT_NAME}
    darknet_cpp
    darknet_app
    pthread
)

if (OPENCV)
    target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
    )
endif()

if (GPU)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler="-fPIC")
    cuda_add_library(darknet_cuda STATIC ${CUDA_SOURCES})
    target_link_libraries(${PROJECT_NAME}
	darknet_cuda
	${CUDA_curand_LIBRARY}
	${CUDA_LIBRARIES}
    )
    CUDA_ADD_CUBLAS_TO_TARGET(${PROJECT_NAME})
endif()

if (CUDNN)
    target_link_libraries(${PROJECT_NAME}
	${CUDNN_LIBRARIES}
    )
endif()

set(APP_NAME executable)
add_executable(${APP_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/examples/darknet.c)
target_link_libraries(${APP_NAME} ${PROJECT_NAME})
set_target_properties(${APP_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

add_custom_target(link_target ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    $<TARGET_FILE:${PROJECT_NAME}>
    ${PROJECT_SOURCE_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
)

