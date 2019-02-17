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
# ``PTHREADS_FOUND``
#   True if PThreads library found
#
# ``PTHREADS_INCLUDE_DIR``
#   Location of PThreads headers
#
# ``PTHREADS_LIBRARY``
#   List of libraries to link with when using PThreads
#

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)

if(NOT PTHREADS_INCLUDE_DIR)
  find_path(PTHREADS_INCLUDE_DIR NAMES pthread.h)
endif()

# Allow libraries to be set manually
if(NOT PTHREADS_LIBRARY)
  find_library(PTHREADS_LIBRARY_RELEASE NAMES pthreadsVC2)
  find_library(PTHREADS_LIBRARY_DEBUG NAMES pthreadsVC2d)
  select_library_configurations(PTHREADS)
endif()

find_package_handle_standard_args(PTHREADS DEFAULT_MSG PTHREADS_LIBRARY PTHREADS_INCLUDE_DIR)
mark_as_advanced(PTHREADS_INCLUDE_DIR PTHREADS_LIBRARY)


# Register imported libraries:
# 1. If we can find a Windows .dll file (or if we can find both Debug and
#    Release libraries), we will set appropriate target properties for these.
# 2. However, for most systems, we will only register the import location and
#    include directory.

# Look for dlls, for Release and Debug libraries.
if(WIN32)
  string( REPLACE ".lib" ".dll" PTHREADS_LIBRARY_RELEASE_DLL "${PTHREADS_LIBRARY_RELEASE}" )
  string( REPLACE ".lib" ".dll" PTHREADS_LIBRARY_DEBUG_DLL   "${PTHREADS_LIBRARY_DEBUG}" )
endif()

if( PTHREADS_FOUND AND NOT TARGET PThreads::PThreads )
  if( EXISTS "${PTHREADS_LIBRARY_RELEASE_DLL}" )
    add_library( PThreads::PThreads      SHARED IMPORTED )
    set_target_properties( PThreads::PThreads PROPERTIES
      IMPORTED_LOCATION_RELEASE         "${PTHREADS_LIBRARY_RELEASE_DLL}"
      IMPORTED_IMPLIB                   "${PTHREADS_LIBRARY_RELEASE}"
      INTERFACE_INCLUDE_DIRECTORIES     "${PTHREADS_INCLUDE_DIR}"
      IMPORTED_CONFIGURATIONS           Release
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
    if( EXISTS "${PTHREADS_LIBRARY_DEBUG_DLL}" )
      set_property( TARGET PThreads::PThreads APPEND PROPERTY IMPORTED_CONFIGURATIONS Debug )
      set_target_properties( FFTW::fftw3 PROPERTIES
        IMPORTED_LOCATION_DEBUG           "${PTHREADS_LIBRARY_DEBUG_DLL}"
        IMPORTED_IMPLIB_DEBUG             "${PTHREADS_LIBRARY_DEBUG}" )
    endif()
  else()
    add_library( PThreads::PThreads      UNKNOWN IMPORTED )
    set_target_properties( PThreads::PThreads PROPERTIES
      IMPORTED_LOCATION                 "${PTHREADS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${PTHREADS_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
  endif()
endif()
