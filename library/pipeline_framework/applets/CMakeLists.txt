# The pipeline applets: `runner`, `pipe-config` and `pipe-to-dot`
#
# Imported from kwiver's `sprokit/src/applets` in P5-T03 and built from here
# since P5-T05.

set( sources
  register_tools.cxx
  pipe_config.cxx
  pipeline_runner.cxx
  pipe_to_dot.cxx
  tool_io.cxx
  )

set( headers
  pipe_config.h
  pipeline_runner.h
  pipe_to_dot.h
  tool_io.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin( sprokit_applets
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${sources} ${headers}
  PRIVATE      vital
               vital_vpm
               vital_logger
               kwiversys
               vital_applets
               sprokit_pipeline
               sprokit_pipeline_util
)

viame_lite_generated( sprokit_applets_export.h pipeline_framework/applets )
