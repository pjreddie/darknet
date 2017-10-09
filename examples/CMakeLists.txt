
add_executable( darknet
  art.c
  captcha.c
  cifar.c
  classifier.c
  coco.c
  darknet.c
  detector.c
  dice.c
  go.c
  lsd.c
  nightmare.c
  regressor.c
  rnn.c
  rnn_vid.c
  segmenter.c
  super.c
  swag.c
  tag.c
  voxel.c
  writing.c
  yolo.c )

target_link_libraries( darknet
  darknet_lib
  ${CMAKE_THREAD_LIBS_INIT} ${DARKNET_LINKED_LIBS}
  )
target_include_directories(darknet PRIVATE ${CMAKE_SOURCE_DIR}/src)

set_target_properties( darknet
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

install( TARGETS darknet
  ARCHIVE DESTINATION     lib
  LIBRARY DESTINATION     lib
  RUNTIME DESTINATION     bin
  COMPONENT    runtime
  )
