project(vistk_tools_pipelines_include)

set(pipeline_includes)

install(
  FILES       ${pipeline_includes}
  DESTINATION share/vistk/pipelines/include
  COMPONENT   pipelines)
