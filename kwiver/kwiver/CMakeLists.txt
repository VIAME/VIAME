#
# Top level build for kwiver core and processes
#
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden KWIVER_HAVE_GCC_VISIBILITY)

# ==================================================================
# Configure config.h.in
kwiver_configure_file(exim_config.h
  "${CMAKE_CURRENT_SOURCE_DIR}/exim_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/exim_config.h"
  BUILD_SHARED_LIBS
  KWIVER_HAVE_GCC_VISIBILITY
  )
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/exim_config.h"
  DESTINATION include/kwiver
  )

include_directories("${KWIVER_SOURCE_DIR}/kwiver/src")
include_directories("${KWIVER_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Adding module subdirectories
kwiver_add_module(core "${CMAKE_CURRENT_LIST_DIR}/core")

kwiver_add_module(processes "${CMAKE_CURRENT_LIST_DIR}/processes")

# other language bindings
add_subdirectory( bindings )

#
# Build documentation
#
# - run doxygen on core, processes
