#
# kwiver/core library
#

set( kwiver_logger_public_headers
  logger-config.h
  logger.h
  kwiver_logger.h
  location_info.h
)

set( kwiver_logger_private_headers
  kwiver_logger_factory.h
  )

set( kwiver_logger_sources
  kwiver_logger.cxx
  location_info.cxx
  kwiver_logger_factory.cxx
)

kwiver_install_headers(
  ${kwiver_logger_public_headers}
  SUBDIR logger
  )

kwiver_add_library( kwiver_logger
  ${kwiver_logger_public_headers}
  ${kwiver_logger_private_headers}
  ${kwiver_logger_sources}
  )

target_link_libraries( kwiver_logger
  PRIVATE              kwiversys
                     # ${Boost_LIBRARIES}
  )
