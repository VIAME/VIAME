#
# vital/config library
#
# The config library must built as a separate library because it is
# used by other components that do not need all of the rest of kwiver
#


set( vital_config_public_headers
  config_block.h
  config_block_exception.h
  config_block_formatter.h
  config_block_io.h
  config_block_types.h
  config_difference.h
  token_type_config.h
)

set( vital_config_private_headers
  config_parser.h
  format_config_block.h
  )

set( vital_config_sources
  config_block.cxx
  config_block_exception.cxx
  config_block_formatter.cxx
  config_block_io.cxx
  config_difference.cxx
  config_parser.cxx
  token_type_config.cxx
)

kwiver_install_headers(
  ${vital_config_public_headers}
  SUBDIR     vital/config
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config_export.h
  SUBDIR     vital/config
  NOPATH
)

kwiver_add_library( vital_config
  ${vital_config_public_headers}
  ${vital_config_private_headers}
  ${vital_config_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config_export.h
  )

add_dependencies (vital_config kwiver_configure)

target_link_libraries( vital_config
  PUBLIC               vital_exceptions
  PRIVATE              kwiversys
                       vital_logger
                       vital_util
  )

kwiver_add_plugin( format_config
  SOURCES          format_config_block_plugin.cxx
  PRIVATE          vital_config
                   vital_vpm
                   vital_util
                   vital_exceptions
  SUBDIR           ${kwiver_plugin_module_subdir}
  )

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
