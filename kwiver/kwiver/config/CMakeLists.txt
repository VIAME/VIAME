#
# kwiver/config library
#
# The config library must built as a separate library because it is
# used by other components that do not need all of the rest of kwiver
#

set( kwiver_config_public_headers
  config-config.h
  config_block.h
  config_block_types.h
  config_block_io.h
  config_block_exception.h
  config_block_io_exception.h
)

set( kwiver_config_sources
  config_block.cxx
  config_block_io.cxx
  config_block_exception.cxx
  config_block_io_exception.cxx
)

kwiver_install_headers(
  ${kwiver_config_public_headers}
  SUBDIR config
  )

kwiver_add_library( kwiver_config
  ${kwiver_config_public_headers}
  ${kwiver_config_sources}
  )

set_target_properties( kwiver_config PROPERTIES
  VERSION                  ${KWIVER_VERSION} SOVERSION ${KWIVER_VERSION}
  LIBRARY_OUTPUT_DIRECTORY ${KWIVER_BUILD_INSTALL_PREFIX}/lib
)

target_link_libraries( kwiver_config
  PRIVATE                 ${Boost_LIBRARIES}
  )
