# The pipeline applet: `runner`
#
# Imported from kwiver's `sprokit/src/applets` in P5-T03 and built from here
# since P5-T05. `pipe-config` and `pipe-to-dot` went in P8-T07: VIAME's own
# `pipeline` applet does what `pipe-config` did -- info, get, set, flatten,
# check -- and `pipe-to-dot`'s graph output had no caller in the tree, no
# shipped pipeline, and 900 lines of exporter behind it.

set( sources
  register_tools.cxx
  pipeline_runner.cxx
  )

set( headers
  pipeline_runner.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

viame_add_plugin( viame_applets_plugin
  KIND        APPLETS
  SOURCES      ${sources} ${headers}
  PRIVATE      viame_algorithm_framework
               viame_plugin
               viame_logger
               viame_applets
               viame_pipeline_framework
               viame_pipeline_util
)

viame_lite_generated( viame_applets_plugin_export.h pipeline_framework/applets )
