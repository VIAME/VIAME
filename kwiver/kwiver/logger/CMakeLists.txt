#
# kwiver/core library
#

set( kwiver_logger_public_headers
  logger.h

)

set( kwiver_logger_sources
  logger.cxx
)

kwiver_install_headers(
  ${kwiver_logger_public_headers}
  SUBDIR logger
  )

kwiver_add_library( kwiver_logger
  ${kwiver_config_public_headers}
  ${kwiver_config_sources}
  )

set_target_properties( kwiver_logger PROPERTIES
  VERSION ${KWIVER_VERSION} SOVERSION ${KWIVER_VERSION}
  LIBRARY_OUTPUT_DIRECTORY ${KWIVER_BUILD_INSTALL_PREFIX}/lib
)

target_link_libraries( kwiver_logger
  ${Boost_LIBRARIES} # ???
  )
