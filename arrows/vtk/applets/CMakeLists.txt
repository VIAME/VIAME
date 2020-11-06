#
# CMake file for the arrows.vtk.applets
#

set( sources
  register_applets.cxx
  color_mesh.cxx
  )

set( headers
  color_mesh.h
  )

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
