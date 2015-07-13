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
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config_export.h
)

set( vital_config_sources
  config_block.cxx
  config_block_io.cxx
  config_block_exception.cxx
)

kwiver_install_headers(
  ${vital_config_public_headers}
  SUBDIR config
  )

kwiver_install_header_files(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_config_export.h
  SUBDIR config
)

kwiver_add_library( vital_config
  ${vital_config_public_headers}
  ${vital_config_private_headers}
  ${vital_config_sources}
  )

target_link_libraries( vital_config
  PRIVATE            ${Boost_LIBRARIES}
  )

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
