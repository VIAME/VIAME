
###
# Add all plugin subdirectories below
##

set( viame_plugin_subdir           viame )
set( viame_plugin_process_subdir   ${viame_plugin_subdir}/processes )

add_subdirectory( core )
add_subdirectory( examples )

if( VIAME_ENABLE_PYTHON )
  kwiver_create_python_init( arrows )
  kwiver_create_python_init( processes )
endif()

if( VIAME_ENABLE_OPENCV )
  add_subdirectory( opencv )
endif()

if( VIAME_ENABLE_ITK )
  add_subdirectory( itk )
endif()

if( VIAME_ENABLE_SEAGIS )
  add_subdirectory( seagis )
endif()

if( VIAME_ENABLE_MATLAB )
  add_subdirectory( matlab )
endif()

if( VIAME_ENABLE_TENSORFLOW )
  add_subdirectory( tensorflow )
endif()

if( VIAME_ENABLE_PYTORCH )
  add_subdirectory( pytorch )
endif()

if( VIAME_ENABLE_DARKNET )
  add_subdirectory( darknet )
endif()

if( VIAME_ENABLE_VXL )
  add_subdirectory( vxl )
endif()

if( VIAME_ENABLE_POSTGRESQL )
  add_subdirectory( cppdb )
endif()

if( VIAME_ENABLE_SVM )
  add_subdirectory( svm )
endif()

if( VIAME_ENABLE_VERTEX_AI AND VIAME_ENABLE_PYTHON )
  add_subdirectory( vertex-ai )
endif()

###
# Add all code that's a part of the simplified python API here
##

if( VIAME_ENABLE_PYTHON AND VIAME_ENABLE_KWIVER )
  kwiver_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/types.py
    .
    types )
endif()
