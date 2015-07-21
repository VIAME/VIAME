#
# CMake file fot the vital tools
#

kwiver_add_executable( plugin_explorer
  plugin_explorer.cxx
  )

target_link_libraries( plugin_explorer
  vital
  vital_apm
  vital_logger
  boost_program_options
  boost_filesystem
  boost_system
  )
