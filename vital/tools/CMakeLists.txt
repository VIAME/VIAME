#
# CMake file fot the vital tools
#

kwiver_add_executable( plugin_manager
  plugin_manager.cxx
  )

target_link_libraries( plugin_manager
  vital
  vital_apm
  vital_logger
  boost_program_options
  boost_filesystem
  boost_system
  )
