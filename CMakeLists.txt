#
#
#

project( darknet )

cmake_minimum_required(VERSION 3.0)

add_subdirectory( src )

if( WIN32 )
  add_subdirectory( 3rdparty )
endif()

# configure the package description file.
configure_file( darknetConfig.cmake.in
  ${CMAKE_BINARY_DIR}/darknetConfig.cmake @ONLY )
