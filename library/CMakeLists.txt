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

# Vital first: everything below links it, and `core_types` is a subdirectory
# of `algorithm_framework` in the build even though it is a sibling here.
add_subdirectory( algorithm_framework )
add_subdirectory( pipeline_framework )
add_subdirectory( image_ops )
add_subdirectory( classifiers )
add_subdirectory( image_processing )
add_subdirectory( object_detectors )
add_subdirectory( object_trackers )
add_subdirectory( video_io )
add_subdirectory( file_io )
add_subdirectory( descriptors )
add_subdirectory( measurement )
add_subdirectory( examples )
