cmake_minimum_required(VERSION 2.8)

PROJECT(DARKNET C CXX)

####################################################################
####################################################################
######           DEFINE DARKNET CMAKE OPTIONS                 ######
####################################################################
####################################################################
OPTION(BUILD_python "Build Python wrapper" ON)
OPTION(USE_GPU "Compile with GPU support" ON)

SET(DARKNET_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/examples ${CMAKE_CURRENT_LIST_DIR}/)

IF (USE_GPU)
	add_definitions(-DGPU)
ENDIF()


####################################################################
####################################################################
######           DEFINE LOCAL CMAKE REPO AND TOOLS            ######
####################################################################
####################################################################


IF (NOT DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
ELSEIF (CMAKE_BUILD_TYPE STREQUAL "" )
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
ENDIF()


####################################################################
####################################################################



####################################################################
######           DEPENDENCIES                                 ######
####################################################################
# ---[ Python


#Cuda
FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(CUDA REQUIRED)




####################################################################
####################################################################
######           SET COMPILLER OPTIONS                        ######
####################################################################
####################################################################
add_definitions(-std=c++0x)
add_definitions(-fPIC)
####################################################################
####################################################################

set(LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" CACHE STRING "Library directory name" FORCE)
set(HEADER_DESTINATION "${CMAKE_INSTALL_PREFIX}/include" CACHE STRING "include directory name" FORCE)



add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(DarknetAPI)
#add_subdirectory(python)
add_subdirectory(test)




SET(RUNME_SCRIPT ${PROJECT_SOURCE_DIR}/activate.sh)
IF (EXISTS ${RUNME_SCRIPT})
    file(REMOVE ${RUNME_SCRIPT})
ENDIF()
file(APPEND ${RUNME_SCRIPT} "export PYTHONPATH=${NUMPY-OPENCV-CONVERTER_PATH}:${PROJECT_SOURCE_DIR}/python\n")



