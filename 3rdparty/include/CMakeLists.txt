
if( INSTALL_HEADER_FILES )
  install( FILES pthread.h sched.h semaphore.h unistd.h
    DESTINATION "include/darknet"
    )
endif()
