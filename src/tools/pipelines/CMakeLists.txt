project(vistk_tools_pipelines)

add_subdirectory(include)

set(pipelines
  grayscale.pipe
  mask_scoring.pipe
  warp_image.pipe)

install(
  FILES       ${pipelines}
  DESTINATION "${PIPE_INSTALL_PATH}"
  COMPONENT   pipelines)
