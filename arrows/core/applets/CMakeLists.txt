#
# CMake file for the arrows.core.applets
#

set( sources
  register_applets.cxx

  dump_klv.cxx
  render_mesh.cxx
  track_features.cxx
  )

set( headers
  dump_klv.h
  render_mesh.h
  track_features.h
  )

###
# Add our plugin path to the default list
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

###
# Add applet plugin
kwiver_add_plugin( kwiver_algo_core_applets
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${sources} ${headers}
  PRIVATE      kwiver_algo_core
               kwiver_tools_applet
               kwiversys
)
