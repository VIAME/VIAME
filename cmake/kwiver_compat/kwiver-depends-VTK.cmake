# Optional find and confgure VTK dependency

kwiver_package_option(VTK
  DESCRIPTION "Enable VTK dependent code and plugins (Arrows)"
)

if( kwiver_enabled_vtk )
  find_package(VTK)
  if(VTK_VERSION VERSION_LESS 9.0)
    find_package(VTK REQUIRED
        COMPONENTS
        vtkCommonCore
        vtkCommonDataModel
        vtkIOXML
        vtkIOPLY
        vtkIOGeometry
        vtkRenderingCore
        vtkRenderingOpenGL2
        )
    if(VTK_VERSION VERSION_LESS 8.2)
        message(FATAL_ERROR "${PROJECT_NAME} supports VTK >= v8.2 "
            "(Found ${VTK_VERSION})")
    endif()

    include(${VTK_USE_FILE})
  else()
    find_package(VTK REQUIRED
        COMPONENTS
        CommonCore
        CommonDataModel
        IOXML
        IOPLY
        IOGeometry
        RenderingCore
        RenderingOpenGL2
        )
  endif()

endif()
