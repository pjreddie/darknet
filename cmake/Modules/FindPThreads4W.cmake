# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindPThreads4W
# ------------
#
# Find the PThread4W includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This script defines the following variables:
#
# ``PThreads4W_FOUND``
#   True if PThreads4W library found
#
# ``PThreads4W_VERSION``
#   Containing the PThreads4W version tag (manually defined)
#
# ``PThreads4W_INCLUDE_DIR``
#   Location of PThreads4W headers
#
# ``PThreads4W_LIBRARY``
#   List of libraries to link with when using PThreads4W (no exception handling)
#
# Result Targets
# ^^^^^^^^^^^^^^^^
#
# This script defines the following targets:
#
# ``PThreads4W::PThreads4W``
#   Target to use PThreads4W (no exception handling)
#

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)

if(NOT PThreads4W_INCLUDE_DIR)
  find_path(PThreads4W_INCLUDE_DIR NAMES pthread.h)
endif()

set(PThreads4W_MAJOR_VERSION 2)
set(PThreads4W_MINOR_VERSION 0)
set(PThreads4W_PATCH_VERSION 0)
set(PThreads4W_VERSION "${PThreads4W_MAJOR_VERSION}.${PThreads4W_MINOR_VERSION}.${PThreads4W_PATCH_VERSION}")

# Allow libraries to be set manually
if(NOT PThreads4W_LIBRARY)
  find_library(PThreads4W_LIBRARY NAMES pthreadVC${PThreads4W_MAJOR_VERSION})
endif()

find_package_handle_standard_args(PThreads4W DEFAULT_MSG PThreads4W_LIBRARY PThreads4W_INCLUDE_DIR)
mark_as_advanced(PThreads4W_INCLUDE_DIR PThreads4W_LIBRARY )

set(PThreads4W_DLL_DIR ${PThreads4W_INCLUDE_DIR})
list(TRANSFORM PThreads4W_DLL_DIR APPEND "/../bin")
message(STATUS "PThreads4W_DLL_DIR: ${PThreads4W_DLL_DIR}")

find_file(PThreads4W_LIBRARY_DLL NAMES pthreadVC${PThreads4W_MAJOR_VERSION}.dll PATHS ${PThreads4W_DLL_DIR})

if( PThreads4W_FOUND AND NOT TARGET PThreads4W::PThreads4W )
  if( EXISTS "${PThreads4W_LIBRARY_RELEASE_DLL}" )
    add_library( PThreads4W::PThreads4W      SHARED IMPORTED )
    set_target_properties( PThreads4W::PThreads4W PROPERTIES
      IMPORTED_LOCATION_RELEASE         "${PThreads4W_LIBRARY_DLL}"
      IMPORTED_IMPLIB                   "${PThreads4W_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${PThreads4W_INCLUDE_DIR}"
      IMPORTED_CONFIGURATIONS           Release
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
  else()
    add_library( PThreads4W::PThreads4W      UNKNOWN IMPORTED )
    set_target_properties( PThreads4W::PThreads4W PROPERTIES
      IMPORTED_LOCATION_RELEASE         "${PThreads4W_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${PThreads4W_INCLUDE_DIR}"
      IMPORTED_CONFIGURATIONS           Release
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
  endif()
endif()
