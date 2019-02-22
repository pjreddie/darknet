# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindCUDNN
# --------
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  ``CUDNN_FOUND``
#    True if CUDNN found on the local system
#
#  ``CUDNN_INCLUDE_DIRS``
#    Location of CUDNN header files.
#
#  ``CUDNN_LIBRARIES``
#    The CUDNN libraries.
#

include(FindPackageHandleStandardArgs)

if(NOT CUDNN_INCLUDE_DIR)
  find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN}
    PATH_SUFFIXES cuda/include include)
endif()

if(NOT CUDNN_LIBRARY)
  find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
endif()

find_package_handle_standard_args(
    CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(CUDNN_FOUND)
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                 CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                 CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                 CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                 CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                 CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                 CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()

  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  message(STATUS "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARY})")
  mark_as_advanced(CUDNN_ROOT_DIR CUDNN_LIBRARY CUDNN_INCLUDE_DIR)
endif()
