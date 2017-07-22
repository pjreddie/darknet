

message("DIR: ${CUDA_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/../ ${CMAKE_CURRENT_LIST_DIR}/../include ${CMAKE_CURRENT_LIST_DIR}/../examples ${CMAKE_CURRENT_LIST_DIR}/../src ${CMAKE_CURRENT_LIST_DIR} ${OPENCV_INCLUDE_DIR} ${RAPIDJSON_INCLUDE_DIR})
cuda_add_executable(darknetTest main.cpp)

target_link_libraries(darknetTest
        darknetAPILib
        darknetExamplesLib
        darknetLib
        pthread
        m
        ${Darknet_LINKER_LIBS}
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        cudart
        cublas
        curand
    )
