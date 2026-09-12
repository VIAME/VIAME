#
# The logger
#
# `logger.h` is the interface -- the `LOG_*` macros and `get_logger` -- and
# everything else here is what they call. P8-T04 removed the factory, the
# manager and the loadable back end they existed to select between; see
# `kwiver_logger.h` for why.
#

set( vital_logger_public_headers
  logger.h
  kwiver_logger.h
  location_info.h
)

set( vital_logger_private_headers
  ${CMAKE_CURRENT_BINARY_DIR}/vital_logger_export.h
  )

set( vital_logger_sources
  kwiver_logger.cxx
  location_info.cxx
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_logger_export.h
  NOPATH
  SUBDIR      viame/algorithm_framework/logger
  )

kwiver_add_library( vital_logger
  ${vital_logger_public_headers}
  ${vital_logger_private_headers}
  ${vital_logger_sources}
  )

# Republished under the prefix the imported sources use.
viame_lite_generated( vital_logger_export.h algorithm_framework/logger )
