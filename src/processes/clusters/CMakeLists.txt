project(vistk_processes_clusters)

set(clusters_srcs
  registration.cxx)

set(clusters_private_headers
  clusters-config.h
  registration.h)

set(vistk_default_cluster_paths
  "${CMAKE_INSTALL_PREFIX}/share/vistk/pipelines/clusters")

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

option(VISTK_USE_BUILD_CLUSTER_PATHS "Whether to use the build tree for the default cluster path" ${VISTK_USE_BUILD_TREE})
mark_as_advanced(VISTK_USE_BUILD_CLUSTER_PATHS)
if (VISTK_USE_BUILD_CLUSTER_PATHS)
  set(vistk_default_cluster_paths
    "${vistk_default_cluster_paths}${path_sep}${vistk_source_dir}/src/tools/pipelines/clusters")
endif ()

set(path_prefix)

if (WIN32)
  set(path_prefix "L")
endif ()

set(VISTK_DEFAULT_CLUSTER_PATHS "${vistk_default_cluster_paths}"
  CACHE STRING "The default paths to search for clusters in")
set(cluster_build_options
  "DEFAULT_CLUSTER_PATHS=${path_prefix}\"${VISTK_DEFAULT_CLUSTER_PATHS}\"")

set(registration_build_options)

if (WIN32)
  vistk_configure_file(cluster-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/cluster-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cluster-paths.h"
    VISTK_DEFAULT_CLUSTER_PATHS
    path_prefix)

  set(clusters_private_headers
    ${clusters_private_headers}
    cluster-paths.h.in)
else ()
  set(registration_build_options
    ${registration_build_options}
    ${cluster_build_options})
endif ()

set_source_files_properties(registration.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${registration_build_options}")

vistk_private_header_group(${clusters_private_headers})
add_vistk_plugin(processes_clusters
  MAKE_VISTK_PROCESSES_CLUSTERS_LIB
  ${clusters_srcs}
  ${clusters_private_headers})
target_link_libraries(processes_clusters
  vistk_pipeline_util
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
