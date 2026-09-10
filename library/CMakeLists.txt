###
# Functional libraries
#
# Code that has been taken off a third-party dependency lands here, in a
# directory named for what it does rather than for what it used to depend on.
# See design/lite-library-layout.md for the target tree; `plugins/` holds what
# has not been converted yet.
##

set( viame_plugin_subdir           viame )
set( viame_plugin_process_subdir   ${viame_plugin_subdir}/processes )

add_subdirectory( image_ops )
add_subdirectory( image_processing )
add_subdirectory( object_detectors )
add_subdirectory( video_io )
add_subdirectory( file_io )
add_subdirectory( descriptors )
