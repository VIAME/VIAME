
###
# Add all plugin subdirectories below
##

set( viame_plugin_subdir           viame )
set( viame_plugin_process_subdir   ${viame_plugin_subdir}/processes )

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

# `plugins/tensorflow` stood here; `VIAME_ENABLE_TENSORFLOW` went in P1.

if( VIAME_ENABLE_ONNX AND VIAME_ENABLE_PYTHON )
  add_subdirectory( onnx )
endif()

if( VIAME_ENABLE_SVM )
  add_subdirectory( svm )
endif()

if( VIAME_ENABLE_VERTEX_AI )
  add_subdirectory( vertex-ai )
endif()
