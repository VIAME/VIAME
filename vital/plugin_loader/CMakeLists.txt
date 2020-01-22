###
# plugin loader

###
# vital plugin manager
set(vital_vpm_headers
  plugin_loader.h
  plugin_factory.h
  plugin_manager.h
  plugin_info.h
  plugin_registrar.h
  )

set(vital_vpm_sources
  plugin_loader.cxx
  plugin_factory.cxx
  plugin_manager.cxx
  plugin_filter_default.cxx
  plugin_filter_category.cxx
  )

set(vital_vpm_headers_private
  plugin_loader_filter.h
  plugin_filter_default.h
  plugin_filter_category.h
  ${CMAKE_CURRENT_BINARY_DIR}/vital_vpm_export.h
  )

kwiver_add_library(vital_vpm
  ${vital_vpm_headers}
  ${vital_vpm_headers_private}
  ${vital_vpm_sources}
  )

kwiver_install_headers(
  ${vital_vpm_headers}
  SUBDIR      vital/plugin_loader
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_vpm_export.h
  SUBDIR      vital/plugin_loader
  NOPATH
  )

target_link_libraries(vital_vpm
  PUBLIC              vital_util
                      kwiversys
  PRIVATE             vital_logger
                      vital_exceptions
                      ${VITAL_BOOST_REGEX}
  )

###
# Add tests if enbled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()
