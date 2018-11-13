#
# CMake file for the vital tools
#

set( sources
  register_tools.cxx

  config_explorer.cxx
  dump_klv.cxx
  )

set( headers
  config_explorer.h
  dump_klv.h
  cxxopts.hpp
  )

###
# Add our plugin path to the default list
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

###
# Add path to applets so plugin loader can find them
kwiver_make_module_path( ${KWIVER_BINARY_DIR} ${kwiver_plugin_applets} )
kwiver_add_module_path( ${kwiver_module_path_result} )

###
#
kwiver_add_library( kwiver_tools_applet
               kwiver_applet.h
               kwiver_applet.cxx
)

target_link_libraries( kwiver_tools_applet
              vital_util
  )


###
# built the tool runner
kwiver_add_executable( kwiver
           kwiver_tool_runner.cxx
)

target_link_libraries( kwiver
  PRIVATE        vital
                 vital_vpm
                 vital_logger
                 kwiversys
                 kwiver_tools_applet
)

###
# Add applet for tools plugin
kwiver_add_plugin( kwiver_tools
  SUBDIR       ${kwiver_plugin_applets}
  SOURCES      ${sources} ${headers}
  PRIVATE      vital
               vital_vpm
               vital_klv
               vital_algo
               vital_logger
               kwiversys
               kwiver_tools_applet
)

kwiver_install_headers(
  kwiver_applet.h
  cxxopts.hpp
  SUBDIR      tools
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_tools_applet_export.h
  NOPATH
  SUBDIR      tools
  )

if( KWIVER_ENABLE_ARROWS AND KWIVER_ENABLE_QT AND KWIVER_ENABLE_PROCESSES )
  add_subdirectory( pipeline_viewer )
endif()
