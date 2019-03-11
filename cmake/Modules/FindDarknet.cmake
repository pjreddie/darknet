# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindDarknet
# ------------
#
# Find the Darknet includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``Darknet_FOUND``
#   True if Darknet library found
#
# ``Darknet_INCLUDE_DIR``
#   Location of Darknet headers
#
# ``Darknet_LIBRARY``
#   List of libraries to link with when using Darknet
#

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)

if(NOT Darknet_INCLUDE_DIR)
  find_path(Darknet_INCLUDE_DIR NAMES darknet.h PATHS darknet darknet/include)
endif()

# Allow libraries to be set manually
if(NOT Darknet_LIBRARY)
  find_library(Darknet_LIBRARY_RELEASE NAMES darklib PATHS darknet darknet/lib)
  find_library(Darknet_LIBRARY_DEBUG NAMES darklibd PATHS darknet darknet/lib)
  select_library_configurations(Darknet)
endif()

set(Darknet_INCLUDE_DIRS "${Darknet_INCLUDE_DIR}")
set(Darknet_LIBRARIES "${Darknet_LIBRARY}")

find_package_handle_standard_args(Darknet DEFAULT_MSG Darknet_LIBRARY Darknet_INCLUDE_DIR)
mark_as_advanced(Darknet_INCLUDE_DIR Darknet_INCLUDE_DIRS Darknet_LIBRARY Darknet_LIBRARIES)
