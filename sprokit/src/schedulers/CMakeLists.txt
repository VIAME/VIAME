project(sprokit_schedulers)

add_subdirectory(examples)

sprokit_create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  schedulers
  pipeline)
