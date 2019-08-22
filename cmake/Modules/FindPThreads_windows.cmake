# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindPThreads
# ------------
#
# Find the PThreads includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``PThreads_windows_FOUND``
#   True if PThreads_windows library found
#
# ``PThreads_windows_INCLUDE_DIR``
#   Location of PThreads_windows headers
#
# ``PThreads_windows_LIBRARY``
#   List of libraries to link with when using PThreads_windows
#

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)

if(NOT PThreads_windows_INCLUDE_DIR)
  find_path(PThreads_windows_INCLUDE_DIR NAMES pthread.h PATHS ${PThreads_windows_DIR} PATH_SUFFIXES include)
endif()

# Allow libraries to be set manually
if(NOT PThreads_windows_LIBRARY)
  find_library(PThreads_windows_LIBRARY_RELEASE NAMES pthreadsVC2 pthreadVC2 PATHS ${PThreads_windows_DIR} PATH_SUFFIXES lib)
  find_library(PThreads_windows_LIBRARY_DEBUG NAMES pthreadsVC2d pthreadVC2d PATHS ${PThreads_windows_DIR} PATH_SUFFIXES lib)
  select_library_configurations(PThreads_windows)
endif()

find_package_handle_standard_args(PThreads_windows DEFAULT_MSG PThreads_windows_LIBRARY PThreads_windows_INCLUDE_DIR)
mark_as_advanced(PThreads_windows_INCLUDE_DIR PThreads_windows_LIBRARY)

set(PThreads_windows_DLL_DIR ${PThreads_windows_INCLUDE_DIR})
list(TRANSFORM PThreads_windows_DLL_DIR APPEND "/../bin")
message(STATUS "PThreads_windows_DLL_DIR: ${PThreads_windows_DLL_DIR}")

find_file(PThreads_windows_LIBRARY_RELEASE_DLL NAMES pthreadVC2.dll PATHS ${PThreads_windows_DLL_DIR})
find_file(PThreads_windows_LIBRARY_DEBUG_DLL NAMES pthreadVC2d.dll PATHS ${PThreads_windows_DLL_DIR})

# Register imported libraries:
# 1. If we can find a Windows .dll file (or if we can find both Debug and
#    Release libraries), we will set appropriate target properties for these.
# 2. However, for most systems, we will only register the import location and
#    include directory.

if( PThreads_windows_FOUND AND NOT TARGET PThreads_windows::PThreads_windows )
  if( EXISTS "${PThreads_windows_LIBRARY_RELEASE_DLL}" )
    add_library( PThreads_windows::PThreads_windows      SHARED IMPORTED )
    set_target_properties( PThreads_windows::PThreads_windows PROPERTIES
      IMPORTED_LOCATION_RELEASE         "${PThreads_windows_LIBRARY_RELEASE_DLL}"
      IMPORTED_IMPLIB                   "${PThreads_windows_LIBRARY_RELEASE}"
      INTERFACE_INCLUDE_DIRECTORIES     "${PThreads_windows_INCLUDE_DIR}"
      IMPORTED_CONFIGURATIONS           Release
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
    if( EXISTS "${PThreads_windows_LIBRARY_DEBUG_DLL}" )
      set_property( TARGET PThreads_windows::PThreads_windows APPEND PROPERTY IMPORTED_CONFIGURATIONS Debug )
      set_target_properties( PThreads_windows::PThreads_windows PROPERTIES
        IMPORTED_LOCATION_DEBUG           "${PThreads_windows_LIBRARY_DEBUG_DLL}"
        IMPORTED_IMPLIB_DEBUG             "${PThreads_windows_LIBRARY_DEBUG}" )
    endif()
  else()
    add_library( PThreads_windows::PThreads_windows      UNKNOWN IMPORTED )
    set_target_properties( PThreads_windows::PThreads_windows PROPERTIES
      IMPORTED_LOCATION_RELEASE         "${PThreads_windows_LIBRARY_RELEASE}"
      INTERFACE_INCLUDE_DIRECTORIES     "${PThreads_windows_INCLUDE_DIR}"
      IMPORTED_CONFIGURATIONS           Release
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
    if( EXISTS "${PThreads_windows_LIBRARY_DEBUG}" )
      set_property( TARGET PThreads_windows::PThreads_windows APPEND PROPERTY IMPORTED_CONFIGURATIONS Debug )
      set_target_properties( PThreads_windows::PThreads_windows PROPERTIES
        IMPORTED_LOCATION_DEBUG           "${PThreads_windows_LIBRARY_DEBUG}" )
    endif()
  endif()
endif()
