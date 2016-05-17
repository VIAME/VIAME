project(sprokit_processes_clusters)

set(clusters_srcs
  registration.cxx)

set(clusters_private_headers
  clusters-config.h
  registration.h)

set(sprokit_default_cluster_paths
  "${CMAKE_INSTALL_PREFIX}/share/sprokit/pipelines/clusters")

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

option(SPROKIT_USE_BUILD_CLUSTER_PATHS "Whether to use the build tree for the default cluster path" ${SPROKIT_USE_BUILD_TREE})
mark_as_advanced(SPROKIT_USE_BUILD_CLUSTER_PATHS)
if (SPROKIT_USE_BUILD_CLUSTER_PATHS)
  set(sprokit_default_cluster_paths
    "${sprokit_default_cluster_paths}${path_sep}${sprokit_source_dir}/src/tools/pipelines/clusters")
endif ()

set(path_prefix)

if (WIN32)
  set(path_prefix "L")
endif ()

set(SPROKIT_DEFAULT_CLUSTER_PATHS "${sprokit_default_cluster_paths}"
  CACHE STRING "The default paths to search for clusters in" FORCE)
set(cluster_build_options
  "DEFAULT_CLUSTER_PATHS=${path_prefix}\"${SPROKIT_DEFAULT_CLUSTER_PATHS}\"")

set(registration_build_options)

if (WIN32)
  sprokit_configure_file(cluster-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/cluster-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cluster-paths.h"
    SPROKIT_DEFAULT_CLUSTER_PATHS
    path_prefix)

  list(APPEND clusters_private_headers
    cluster-paths.h.in)
else ()
  list(APPEND registration_build_options
    ${cluster_build_options})
endif ()

set_source_files_properties(registration.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${registration_build_options}")

sprokit_private_header_group(${clusters_private_headers})
sprokit_add_plugin(processes_clusters
  MAKE_SPROKIT_PROCESSES_CLUSTERS_LIB
  ${clusters_srcs}
  ${clusters_private_headers})
target_link_libraries(processes_clusters
  PUBLIC              vital_logger

  PRIVATE             sprokit_pipeline_util
                      sprokit_pipeline
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY})
