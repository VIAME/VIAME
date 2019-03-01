#
# CMake file for the arrows.core.applets
#

set( sources
  register_applets.cxx

  render_mesh.cxx
  )

set( headers
  render_mesh.h
  )

###
# Add our plugin path to the default list
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

###
# Add path to applets so plugin loader can find them
kwiver_make_module_path( ${KWIVER_BINARY_DIR} ${kwiver_plugin_applets} )
kwiver_add_module_path( ${kwiver_module_path_result} )


###
# Add applet plugin
kwiver_add_plugin( kwiver_algo_core_applets
  SUBDIR       ${kwiver_plugin_applets}
  SOURCES      ${sources} ${headers}
  PRIVATE      kwiver_algo_core
               kwiver_tools_applet
               kwiversys
)

