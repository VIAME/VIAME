project(sprokit_processes_clusters)

set(clusters_srcs
  registration.cxx
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_plugin(processes_clusters
  SUBDIR       ${kwiver_plugin_process_subdir}
  SOURCES      ${clusters_srcs}
  PUBLIC       vital_logger
  PRIVATE      sprokit_pipeline_util
               sprokit_pipeline
               vital_vpm vital_util kwiversys
  )
