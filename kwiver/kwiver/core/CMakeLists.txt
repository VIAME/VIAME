#
# kwiver/core library
#

set( kwiver_core_public_headers
  config_util.h
  exceptions.h
  geo_lat_lon.h
  homography.h
  image.h
  image_container.h
  matrix.h
  timestamp.h
  vector.h

  exceptions/algorithm.h
  exceptions/base.h
  exceptions/config_block.h
  exceptions/config_block_io.h
  exceptions/image.h
  exceptions/io.h
  exceptions/math.h

  types/kwiver.h
)

set( kwiver_core_sources
  config_util.cxx
  geo_lat_lon.cxx
  homography.cxx
  image.cxx
  matrix.cxx
  timestamp.cxx
  vector.cxx

  exceptions/algorithm.cxx
  exceptions/base.cxx
  exceptions/config_block.cxx
  exceptions/config_block_io.cxx
  exceptions/image.cxx
  exceptions/io.cxx
  exceptions/math.cxx

  types/kwiver.cxx
)

kwiver_install_headers(
  ${kwiver_core_public_headers}
  SUBDIR core
  )

kwiver_add_library( kwiver_core
  ${kwiver_core_public_headers}
  ${kwiver_core_sources}
  )

#set_target_properties( kwiver_core PROPERTIES
#  VERSION ${KWIVER_VERSION} SOVERSION ${KWIVER_VERSION}
#  LIBRARY_OUTPUT_DIRECTORY ${KWIVER_BUILD_INSTALL_PREFIX}/lib
#)

add_dependencies( kwiver_core sprokit maptk VXL )

target_link_libraries( kwiver_core
  ${Boost_LIBRARIES}
  maptk_core
  sprokit_pipeline
  )
