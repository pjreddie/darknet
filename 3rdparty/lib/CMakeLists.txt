
if( NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)" )
  add_subdirectory( x86 )
else()
  add_subdirectory( x64 )
endif()