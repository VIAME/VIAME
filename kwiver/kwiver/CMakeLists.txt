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

##
#
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

###
#
set( kwiver_public_headers
  geo_lat_lon.h
  homography.h
  image.h
  image_container.h
  matrix.h
  timestamp.h
  vector.h

  exceptions/algorithm.h
  exceptions/base.h
  exceptions/image.h
  exceptions/io.h
  exceptions/math.h

  sprokit_type_traits.h
)

set( kwiver_sources
  geo_lat_lon.cxx
  homography.cxx
  image.cxx
  timestamp.cxx
  vector.cxx

  exceptions/algorithm.cxx
  exceptions/base.cxx
  exceptions/image.cxx
  exceptions/io.cxx
  exceptions/math.cxx

  kwiver.cxx
)

kwiver_install_headers(
  ${kwiver_public_headers}
  )

kwiver_add_library( kwiver
  ${kwiver_public_headers}
  ${kwiver_sources}
  )

target_link_libraries( kwiver
  PRIVATE       #  ${Boost_LIBRARIES} # be more specific
                  kwiver_config
                  kwiver_logger
  )

# Adding module subdirectories
kwiver_add_module(config "${CMAKE_CURRENT_LIST_DIR}/config")
kwiver_add_module(logger "${CMAKE_CURRENT_LIST_DIR}/logger")
