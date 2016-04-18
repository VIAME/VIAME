#
# vital/logger library
#

include_directories( ${KWSYS_INCLUDE_DIR} )

set( vital_logger_public_headers
  logger.h
  kwiver_logger.h
  location_info.h
)

set( vital_logger_private_headers
  kwiver_logger_factory.h
  kwiver_logger_manager.h
  default_logger.h
  ${CMAKE_CURRENT_BINARY_DIR}/vital_logger_export.h
  )

set( vital_logger_sources
  kwiver_logger.cxx
  location_info.cxx
  kwiver_logger_factory.cxx
  kwiver_logger_manager.cxx
  default_logger.cxx
)

kwiver_install_headers(
  ${vital_logger_public_headers}
  SUBDIR      vital/logger
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_logger_export.h
  NOPATH
  SUBDIR      vital/logger
  )

kwiver_add_library( vital_logger
  ${vital_logger_public_headers}
  ${vital_logger_private_headers}
  ${vital_logger_sources}
  )

target_link_libraries( vital_logger
  PRIVATE              kwiversys
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

  kwiver_add_plugin( vital_log4cxx_logger
    SOURCES          ${plugin_base_src}
                     log4cxx_factory.cxx
    PRIVATE          ${log4cxx_lib}
                     kwiversys
    SUBDIR           modules
    )

endif()

###
# Add tests if enbled
if (VITAL_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
