#
# kwiver/logger library
#
include_directories("${KWIVER_BINARY_DIR}")

set( kwiver_logger_public_headers
  logger.h
  kwiver_logger.h
  location_info.h
)

set( kwiver_logger_private_headers
  kwiver_logger_factory.h
  kwiver_logger_manager.cxx
  default_logger.h
  )

set( kwiver_logger_sources
  kwiver_logger.cxx
  location_info.cxx
  kwiver_logger_factory.cxx
  kwiver_logger_manager.cxx
  default_logger.cxx
)

kwiver_install_headers(
  ${kwiver_logger_public_headers}
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_logger_export.h
  SUBDIR logger
  )

kwiver_add_library( kwiver_logger
  ${kwiver_logger_public_headers}
  ${kwiver_logger_private_headers}
  ${kwiver_logger_sources}
  )

target_link_libraries( kwiver_logger
  PRIVATE              kwiversys
                       ${Boost_LIBRARIES}
  )

###
# sources that all plugins need
set( plugin_base_src
    kwiver_logger_factory.h     kwiver_logger_factory.cxx
    kwiver_logger.h             kwiver_logger.cxx
    location_info.h             location_info.cxx
  )

###
# Build log4cxx plug-in if configured
if (KWIVER_ENABLE_LOG4CXX)

  set ( log4cxx_lib     ${Log4cxx_LIBRARY} ${ApacheRunTime_LIBRARY} )
  include_directories ( SYSTEM ${Log4cxx_INCLUDE_DIR} ${ApacheRunTime_INCLUDE_DIR} )

  kwiver_add_plugin( kwiver_log4cxx_logger
    ${plugin_base_src} log4cxx_factory.cxx )

  target_link_libraries( kwiver_log4cxx_logger
    ${log4cxx_lib} )

endif(KWIVER_ENABLE_LOG4CXX)

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
