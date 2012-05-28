project(vistk_tools_pipelines_include)

set(pipeline_includes
  _warp1.pipe
  _warp2.pipe
  _warp3.pipe
  _warp4.pipe
  _warp5.pipe
  _warp6.pipe
  _warp7.pipe
  _warp8.pipe
  warp.pipe)

install(
  FILES       ${pipeline_includes}
  DESTINATION share/vistk/pipelines/include
  COMPONENT   pipelines)
