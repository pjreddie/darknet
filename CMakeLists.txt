#
#
#

project( darknet )

cmake_minimum_required(VERSION 3.0)

add_subdirectory( src )

# configure the package description file.
configure_file( darknetConfig.cmake.in
  ${CMAKE_BINARY_DIR}/darknetConfig.cmake @ONLY )

install( FILES ${CMAKE_BINARY_DIR}/darknetConfig.cmake
	 DESTINATION .)
