
###
# Add all plugin subdirectories below
##

set( viame_plugin_subdir           viame )
set( viame_plugin_process_subdir   ${viame_plugin_subdir}/processes )

add_subdirectory( core )
add_subdirectory( claude )
add_subdirectory( examples )

if( VIAME_ENABLE_PYTHON )
  viame_create_python_init( arrows )
  viame_create_python_init( processes )
endif()

if( VIAME_ENABLE_OPENCV )
  add_subdirectory( opencv )
endif()

if( VIAME_ENABLE_COLMAP )
  add_subdirectory( colmap )
endif()

if( VIAME_ENABLE_SEAGIS )
  add_subdirectory( seagis )
endif()

if( VIAME_ENABLE_MATLAB )
  add_subdirectory( matlab )
endif()

# `plugins/tensorflow` stood here; `VIAME_ENABLE_TENSORFLOW` went in P1.

if( VIAME_ENABLE_PYTORCH )
  add_subdirectory( pytorch )
endif()

if( VIAME_ENABLE_ONNX AND VIAME_ENABLE_PYTHON )
  add_subdirectory( onnx )
endif()

if( VIAME_ENABLE_SVM )
  add_subdirectory( svm )
endif()

if( VIAME_ENABLE_VERTEX_AI )
  add_subdirectory( vertex-ai )
endif()

###
# Add all code that's a part of the simplified python API here
##

if( VIAME_ENABLE_PYTHON )
  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/types.py
    .
    types )

  viame_add_python_module(
    ${CMAKE_CURRENT_SOURCE_DIR}/compat.py
    .
    compat )
endif()
