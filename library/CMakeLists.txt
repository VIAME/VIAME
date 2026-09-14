###
# Functional libraries
#
# Code that has been taken off a third-party dependency lands here, in a
# directory named for what it does rather than for what it used to depend on.
# See design/lite-library-layout.md for the tree; phase 2 emptied the
# `plugins` directory that held the rest into these.
##

set( viame_plugin_subdir           viame )
set( viame_plugin_process_subdir   ${viame_plugin_subdir}/processes )

# Vital first: everything below links it, and `core_types` is a subdirectory
# of `algorithm_framework` in the build even though it is a sibling here.
add_subdirectory( algorithm_framework )
add_subdirectory( pipeline_framework )

# File, path and image helpers that no dependency owns and everything above
# here uses. `core_types <- algorithm_framework <- pipeline_framework <-
# utilities <- image_ops`, which is the order in lite-library-layout.md.
add_subdirectory( utilities )

add_subdirectory( image_ops )
add_subdirectory( classifiers )
add_subdirectory( segmentation )
add_subdirectory( image_processing )
add_subdirectory( object_detectors )
add_subdirectory( object_trackers )
add_subdirectory( video_io )
add_subdirectory( file_io )
add_subdirectory( descriptors )
add_subdirectory( measurement )
add_subdirectory( training )
add_subdirectory( evaluation )
add_subdirectory( examples )
