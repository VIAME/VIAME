project(sprokit_schedulers)

add_subdirectory(examples)

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  schedulers
  pipeline)
