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
                 sprokit_tools
                 sprokit_pipeline
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
