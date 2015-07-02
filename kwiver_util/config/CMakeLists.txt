#
# kwiver/config library
#
# The config library must built as a separate library because it is
# used by other components that do not need all of the rest of kwiver
#

include( GenerateExportHeader )

set( kwiver_config_public_headers
  config_block.h
  config_block_types.h
  config_block_io.h
  config_block_exception.h
)

set( kwiver_config_sources
  config_block.cxx
  config_block_io.cxx
  config_block_exception.cxx
)

kwiver_install_headers(
  ${kwiver_config_public_headers}
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_config_export.h
  SUBDIR config
  )

kwiver_add_library( kwiver_config SHARED
  ${kwiver_config_public_headers}
  ${kwiver_config_sources}
  )

target_link_libraries( kwiver_config
  PRIVATE            ${Boost_LIBRARIES}
  )

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

###
# Make appropriate export header file
generate_export_header( kwiver_config )
