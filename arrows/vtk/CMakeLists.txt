# Build / Install VTK arrow

set(CMAKE_FOLDER "Arrows/VTK")

set( plugin_vtk_headers
  depth_utils.h
  mesh_coloration.h
  vtkKwiverCamera.h
  )

kwiver_install_headers(
  SUBDIR     arrows/vtk
  ${plugin_vtk_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_vtk_export.h
  NOPATH   SUBDIR     arrows/vtk
  )

set( plugin_vtk_sources
  depth_utils.cxx
  mesh_coloration.cxx
  vtkKwiverCamera.cxx
  )

kwiver_add_library( kwiver_algo_vtk
  ${plugin_vtk_headers}
  ${plugin_vtk_sources}
  )

if(VTK_VERSION VERSION_LESS 9.0)
  set(VTK_public_targets
      vtkRenderingCore
      vtkRenderingOpenGL2)
  set(VTK_private_targets
      vtkCommonCore
      vtkCommonDataModel
      vtkIOImage
      vtkIOXML)
else()
  set(VTK_public_targets
      VTK::RenderingCore
      VTK::RenderingOpenGL2)
  set(VTK_private_targets
      VTK::CommonCore
      VTK::CommonDataModel
      VTK::IOImage
      VTK::IOXML)
endif()

target_link_libraries( kwiver_algo_vtk
  PUBLIC               vital
                       ${VTK_public_targets}
  PRIVATE              vital_algo
                       ${VTK_private_targets}
  )

#algorithms_create_plugin( kwiver_algo_vtk
#  register_algorithms.cxx
#  )

if (KWIVER_ENABLE_TESTS)
#  add_subdirectory(tests)
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
