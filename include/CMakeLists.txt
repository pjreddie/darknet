##
# Build file for Darknet include directory
##
set( headers
  darknet.h
  )

if( INSTALL_HEADER_FILES )
  install( FILES ${headers}
    DESTINATION "include/darknet"
    )
endif()
