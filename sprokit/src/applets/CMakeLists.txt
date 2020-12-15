project(sprokit_tools)

###
# support for applets

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

###
# Add our plugin path to the default list
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

###
# Add applet for tools plugin
kwiver_add_plugin( sprokit_applets
  SUBDIR       ${kwiver_plugin_applets_subdir}
  SOURCES      ${sources} ${headers}
  PRIVATE      vital
               vital_vpm
               vital_logger
               kwiversys
               kwiver_tools_applet
               sprokit_pipeline
               sprokit_pipeline_util
)
