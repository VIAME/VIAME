#
# CMake file fot the vital tools
#

###
#
##++ There is a problem here in that plugin_explorer needs sprokit includes
##++ But they are not readily available. Generally speaking, vital shopuld not depend
##++ on any other package.
include_directories( ${sprokit_source_dir}/src )

kwiver_add_executable( plugin_explorer
  plugin_explorer.cxx
  )

target_link_libraries( plugin_explorer
  PRIVATE        vital
                 vital_vpm
                 vital_logger
                 kwiversys
                 explorer_plugin
                 sprokit_tools
                 sprokit_pipeline
  )


set( public_headers
  explorer_plugin.h
  )

kwiver_install_headers(
         ${CMAKE_CURRENT_BINARY_DIR}/explorer_plugin_export.h
         ${public_headers}
  SUBDIR vital/tools   # is this the best place?
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
