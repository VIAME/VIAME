
###
# add all plugin subdirectories below

set( viame_plugin_subdir           viame )
set( viame_plugin_process_subdir   ${viame_plugin_subdir}/processes )

add_subdirectory( core )
add_subdirectory( hello_world )

if( VIAME_ENABLE_OPENCV )
  add_subdirectory( opencv )
endif()

if( VIAME_ENABLE_ITK )
  add_subdirectory( itk )
endif()

if( VIAME_ENABLE_SCALLOP_TK )
  add_subdirectory( scallop_tk )
endif()

if( VIAME_ENABLE_FASTER_RCNN )
  add_subdirectory( faster_rcnn )
endif()

if( VIAME_ENABLE_CAMTRAWL )
  add_subdirectory( camtrawl )
endif()

if( VIAME_ENABLE_MATLAB )
  add_subdirectory( matlab )
endif()

if( VIAME_ENABLE_UW_PREDICTOR )
  add_subdirectory( uw_predictor )
endif()

if( VIAME_ENABLE_SMQTK )
  add_subdirectory( smqtk )
endif()

if( VIAME_ENABLE_TENSORFLOW )
  add_subdirectory( tensorflow )
endif()
