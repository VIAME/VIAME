project(sprokit_tools_pipelines)

add_subdirectory(clusters)
add_subdirectory(include)

set(pipelines)

install(
  FILES       ${pipelines}
  DESTINATION share/sprokit/pipelines
  COMPONENT   pipelines)
