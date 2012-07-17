project(vistk_schedulers)

add_subdirectory(examples)

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  schedulers
  pipeline)
