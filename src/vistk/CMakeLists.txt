include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden __VISTK_HAVE_GCC_VISIBILITY)
vistk_configure_file(config.h
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
  __VISTK_HAVE_GCC_VISIBILITY)

add_subdirectory(pipeline)
add_subdirectory(utilities)
add_subdirectory(pipeline_util)
add_subdirectory(image)
add_subdirectory(scoring)

if (VISTK_ENABLE_PYTHON)
  add_subdirectory(python)
endif (VISTK_ENABLE_PYTHON)

install(
  FILES       "${CMAKE_CURRENT_BINARY_DIR}/config.h"
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk"
  COMPONENT   runtime)
