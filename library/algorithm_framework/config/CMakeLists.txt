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
  config_camera_helpers.txx
  config_helpers.txx
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
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config_export.h
  SUBDIR     viame/algorithm_framework/config
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
                       vital_vpm  # uses pluggable
  )

###
# Add tests if enbled

# Republished under the prefix the imported sources use.
viame_lite_generated( vital_config_export.h algorithm_framework/config )
