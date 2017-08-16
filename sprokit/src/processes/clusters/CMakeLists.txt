project(sprokit_processes_clusters)

set(clusters_srcs
  registration.cxx
  )

set(clusters_private_headers
  cluster-paths.h.in
  )

set(sprokit_default_cluster_paths
  "${CMAKE_INSTALL_PREFIX}/share/sprokit/pipelines/clusters")

set(path_prefix)

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

if (KWIVER_USE_BUILD_TREE)
  set(sprokit_default_cluster_paths
    "${sprokit_default_cluster_paths}${path_sep}${sprokit_source_dir}/src/tools/pipelines/clusters")
endif ()

set(SPROKIT_DEFAULT_CLUSTER_PATHS "${sprokit_default_cluster_paths}"
  CACHE STRING "The default paths to search for clusters in" FORCE)

sprokit_configure_file(cluster-paths.h
  "${CMAKE_CURRENT_SOURCE_DIR}/cluster-paths.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cluster-paths.h"
  SPROKIT_DEFAULT_CLUSTER_PATHS
  path_prefix
  path_sep)

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

sprokit_private_header_group(${clusters_private_headers})
kwiver_add_plugin(processes_clusters
  SUBDIR       sprokit
  SOURCES      ${clusters_srcs}
               ${clusters_private_headers}
  PUBLIC       vital_logger
  PRIVATE      sprokit_pipeline_util
               sprokit_pipeline
               ${Boost_FILESYSTEM_LIBRARY}
               ${Boost_SYSTEM_LIBRARY}
               vital_vpm vital_util kwiversys
  )
