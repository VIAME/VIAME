#
# CMake file for the arrows.mvg.applets
#

set( sources
  register_applets.cxx
  color_mesh.cxx
  )

set( headers
  color_mesh.h
  )

###
# Add our plugin path to the default list
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

# Add applet plugin
kwiver_add_plugin( kwiver_algo_vtk_applets
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${sources} ${headers}
  PRIVATE      kwiver_algo_core
               kwiver_algo_vtk
               kwiver_tools_applet
               kwiversys
               vtkIOGeometry
               vtkIOPLY
               vtkIOXML

)
