project(sprokit_tools_pipelines_clusters)

set(pipeline_clusters)

install(
  FILES       ${pipeline_clusters}
  DESTINATION share/sprokit/pipelines/clusters
  COMPONENT   pipelines)
