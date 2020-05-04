#
# CMake file for the vital tools
#

###
#
kwiver_add_executable( plugin_explorer
  plugin_explorer.cxx
  )

target_link_libraries( plugin_explorer
  PRIVATE        vital
                 vital_vpm
                 vital_logger
                 kwiversys
                 explorer_plugin
  )


set( public_headers
  explorer_plugin.h
  )

set_source_files_properties(plugin_explorer.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "PLUGIN_EXPLORER_VERSION=\"1.2\"")

# public headers for explorer
kwiver_install_headers(
         ${public_headers}
  SUBDIR vital/tools
  )

kwiver_install_headers(
         ${CMAKE_CURRENT_BINARY_DIR}/explorer_plugin_export.h
  NOPATH
  SUBDIR vital/tools
  )

kwiver_add_library( explorer_plugin
  ${CMAKE_CURRENT_BINARY_DIR}/explorer_plugin_export.h
  ${public_headers}
  explorer_context.cxx
  )

target_link_libraries( explorer_plugin
  PUBLIC   vital
           ${VITAL_BOOST_REGEX}
  PRIVATE  kwiversys
           vital_util
  )
