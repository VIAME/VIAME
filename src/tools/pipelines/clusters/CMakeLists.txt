project(vistk_tools_pipelines_clusters)

set(pipeline_clusters)

install(
  FILES       ${pipeline_clusters}
  DESTINATION share/vistk/pipelines/clusters
  COMPONENT   pipelines)
