
if(UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC; -O3; -ccbin ${CMAKE_CXX_COMPILER})
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -ccbin "\"${CMAKE_CXX_COMPILER}\"")
endif()


set (darknetAPI_SOURCES
	DarknetAPI
	DarknetAPIConversions
	DarknetDetection
)



include_directories(${CMAKE_CURRENT_LIST_DIR}/../ ${CMAKE_CURRENT_LIST_DIR}/../include ${CMAKE_CURRENT_LIST_DIR}/../examples ${CMAKE_CURRENT_LIST_DIR}/../src ${CMAKE_CURRENT_LIST_DIR} ${OPENCV_INCLUDE_DIR} ${RAPIDJSON_INCLUDE_DIR})
cuda_add_library (darknetAPILib ${darknetAPI_SOURCES} ${gpu_SRCS} SHARED)
install(TARGETS darknetAPILib LIBRARY DESTINATION ${LIB_DESTINATION} ARCHIVE DESTINATION ${LIB_DESTINATION} RUNTIME DESTINATION ${LIB_DESTINATION})

file(GLOB filesB *.h)
file(GLOB filesC ../include/*.h)


install(FILES  ${filesB} DESTINATION  ${HEADER_DESTINATION}/DarknetAPI)
install(FILES  ${filesC} DESTINATION  ${HEADER_DESTINATION})







