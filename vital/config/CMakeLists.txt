#
# vital/config library
#
# The config library must built as a separate library because it is
# used by other components that do not need all of the rest of kwiver
#


set( vital_config_public_headers
  config_block.h
  config_block_types.h
  config_block_io.h
  config_block_exception.h
)

set( vital_config_private_headers
  config_parser.h
  token_expander.h
  token_type_env.h
  token_type.h
  token_type_config.h
  token_type_symtab.h
  token_type_sysenv.h
  )

set( vital_config_sources
  config_block.cxx
  config_block_io.cxx
  config_block_exception.cxx
  config_parser.cxx
  token_expander.cxx
  token_type.cxx
  token_type_config.cxx
  token_type_env.cxx
  token_type_symtab.cxx
  token_type_sysenv.cxx
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
  ${vital_config_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config_export.h
  )

target_link_libraries( vital_config
  PUBLIC             ${Boost_SYSTEM_LIBRARY}
                     ${Boost_REGEX_LIBRARY}
                     vital_logger
  PRIVATE            kwiversys
  )

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
