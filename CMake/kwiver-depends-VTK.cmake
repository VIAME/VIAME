# Optional find and confgure VTK dependency

option( KWIVER_ENABLE_VTK
  "Enable VTK dependent code and plugins (Arrows)"
  ${fletch_ENABLED_VTK}
  )

if( KWIVER_ENABLE_VTK )
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

endif( KWIVER_ENABLE_VTK )
