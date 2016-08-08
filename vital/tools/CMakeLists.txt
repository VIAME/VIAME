#
# CMake file fot the vital tools
#

kwiver_add_executable( plugin_explorer
  plugin_explorer.cxx
  )

target_link_libraries( plugin_explorer
  PRIVATE        vital
                 vital_apm
                 vital_logger
                 kwiversys
  )

###
#
kwiver_add_executable( dump_klv
  dump_klv.cxx
  )

target_link_libraries( dump_klv
  PRIVATE     vital
              vital_apm
              vital_klv
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
               vital_apm
               vital_logger
               kwiversys
  )
