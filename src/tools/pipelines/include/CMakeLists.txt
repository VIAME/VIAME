project(sprokit_tools_pipelines_include)

set(pipeline_includes)

install(
  FILES       ${pipeline_includes}
  DESTINATION share/sprokit/pipelines/include
  COMPONENT   pipelines)
