project(vistk_tools_pipelines)

add_subdirectory(include)

set(pipelines
  grayscale.pipe
  warp_image.pipe)

install(
  FILES       ${pipelines}
  DESTINATION share/vistk/pipelines
  COMPONENT   pipelines)
