###
# plugin loader

###
# vital plugin manager
set(vital_vpm_headers
  plugin_loader.h
  plugin_factory.h
  plugin_manager.h
  )

set(vital_vpm_sources
  plugin_loader.cxx
  plugin_factory.cxx
  plugin_manager.cxx
  )

set(vital_vpm_headers_private
  ${CMAKE_CURRENT_BINARY_DIR}/vital_vpm_export.h
  )

kwiver_add_library(vital_vpm
  ${vital_vpm_headers}
  ${vital_vpm_headers_private}
  ${vital_vpm_sources}
  )

kwiver_install_headers(
  ${vital_vpm_headers}
  SUBDIR      vital
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_vpm_export.h
  SUBDIR      vital
  NOPATH
  )

target_link_libraries(vital_vpm
  PUBLIC              vital
  PRIVATE             vital_logger
                      kwiversys
  )

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
