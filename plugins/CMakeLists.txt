
###
# add all plugin subdirectories below

add_subdirectory( core )

###
# OpenCV plugin
##
if( VIAME_ENABLE_OPENCV )
  add_subdirectory( opencv )
endif()

###
# Scallop-TK plugin
##
option( VIAME_ENABLE_SCALLOP_TK  "Enable ScallopTK plugin"   OFF )

if( VIAME_ENABLE_SCALLOP_TK )
  add_subdirectory( scallop_tk )

  set( VIAME_ENABLE_OPENCV ON CACHE BOOL "OpenCV required for other projects" FORCE )
endif()

###
# Faster RCNN plugin
##
option( VIAME_ENABLE_FASTER_RCNN "Enable Faster RCNN plugin" OFF )

if( VIAME_ENABLE_FASTER_RCNN )
  add_subdirectory( faster_rcnn )

  set( VIAME_ENABLE_PYTHON ON CACHE BOOL "Python required for other projects" FORCE )
  set( VIAME_ENABLE_CAFFE  ON CACHE BOOL "Caffe required for other projects"  FORCE )
endif()
