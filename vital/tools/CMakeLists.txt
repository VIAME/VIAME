#
# CMake file fot the vital tools
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
  PRIVATE  kwiversys
           vital_util
  )

###
#
kwiver_add_executable( dump_klv
  dump_klv.cxx
  )

target_link_libraries( dump_klv
  PRIVATE     vital
              vital_config
              vital_vpm
              vital_klv
              vital_algo
              vital_logger
              vital_video_metadata
              kwiversys
  )

###
#
kwiver_add_executable( config_explorer
  config_explorer.cxx
  )

target_link_libraries( config_explorer
  PRIVATE      vital
               vital_vpm
               vital_config
               vital_logger
               kwiversys
  )
