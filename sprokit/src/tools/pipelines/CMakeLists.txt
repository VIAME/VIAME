project(sprokit_tools_pipelines)

add_subdirectory(clusters)
add_subdirectory(include)

set(pipelines)

sprokit_install_pipelines(${pipelines})
