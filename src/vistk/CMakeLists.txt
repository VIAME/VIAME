include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden __VISTK_HAVE_GCC_VISIBILITY)
configure_file(config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)

add_subdirectory(pipeline)
add_subdirectory(pipeline_types)
add_subdirectory(pipeline_util)

install(
  FILES       ${CMAKE_CURRENT_BINARY_DIR}/config.h
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk"
  COMPONENT   runtime)
