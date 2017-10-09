##
# Build file for Darknet include directory
##
set( headers
  darknet.h
  )

install( FILES ${headers}
  DESTINATION "include/darknet"
  )
