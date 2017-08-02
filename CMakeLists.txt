project( Darknet )

cmake_minimum_required(VERSION 3.0)

add_subdirectory( src )

OPTION( INSTALL_HEADER_FILES "Install header files needed for development" FALSE )

if( WIN32 )
  add_subdirectory( 3rdparty )
endif()

# configure the package description file.
configure_file( DarknetConfig.cmake.in
  ${CMAKE_BINARY_DIR}/DarknetConfig.cmake @ONLY )
