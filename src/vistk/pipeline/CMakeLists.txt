project(vistk_pipeline)

set(pipeline_srcs
  config.cxx
  datum.cxx
  edge.cxx
  edge_exception.cxx
  modules.cxx
  pipeline.cxx
  pipeline_exception.cxx
  process.cxx
  process_exception.cxx
  process_cluster.cxx
  process_registry.cxx
  process_registry_exception.cxx
  scheduler.cxx
  scheduler_exception.cxx
  scheduler_registry.cxx
  scheduler_registry_exception.cxx
  stamp.cxx
  types.cxx
  utils.cxx
  version.cxx)

set(pipeline_headers
  config.h
  datum.h
  edge.h
  edge_exception.h
  modules.h
  pipeline-config.h
  pipeline.h
  pipeline_exception.h
  process.h
  process_exception.h
  process_cluster.h
  process_registry.h
  process_registry_exception.h
  scheduler.h
  scheduler_exception.h
  scheduler_registry.h
  scheduler_registry_exception.h
  stamp.h
  types.h
  utils.h
  version.h)

if (WIN32)
  set(libdir bin)
else ()
  set(libdir lib${LIB_SUFFIX})
endif ()

set(vistk_default_module_paths
  "${CMAKE_INSTALL_PREFIX}/${libdir}/vistk")

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

set(vistk_default_module_paths
  "${vistk_default_module_paths}${path_sep}${module_output_path}")

set(vistk_path_prefix)

if (WIN32)
  set(vistk_path_prefix "L")
endif ()

set(VISTK_DEFAULT_MODULE_PATHS "${vistk_default_module_paths}"
  CACHE STRING "The default paths to search for modules in")
set(vistk_library_options
  "VISTK_DEFAULT_MODULE_PATHS=${vistk_path_prefix}\"${VISTK_DEFAULT_MODULE_PATHS}\"")

set(module_build_options
  "LIBRARY_SUFFIX=\"${CMAKE_SHARED_MODULE_SUFFIX}\"")

if (WIN32)
  vistk_configure_file(module-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/module-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/module-paths.h"
    VISTK_DEFAULT_MODULE_PATHS
    vistk_path_prefix)
else ()
  set(module_build_options
    ${module_build_options}
    ${vistk_library_options})
endif ()

option(VISTK_USE_CONFIGURATION_SUBDIRECTORY "Whether to look in the configuration's subdirectory for each module path" ${vistk_use_config_subdir})
if (VISTK_USE_CONFIGURATION_SUBDIRECTORY)
  set(module_build_options
    ${module_build_options}
    VISTK_USE_CONFIGURATION_SUBDIRECTORY)
endif ()

set_source_files_properties(modules.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${module_build_options}")

set(utils_build_options)

include(CheckFunctionExists)
check_function_exists(setproctitle HAVE_SETPROCTITLE)

if (HAVE_SETPROCTITLE)
  set(utils_build_options
    "HAVE_SETPROCTITLE")
endif ()

set_source_files_properties(utils.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${utils_build_options}")

set(scheduler_build_options)

if (${Boost_VERSION} LESS 105000)
  set(scheduler_build_options
    "BOOST_NO_HAVE_REVERSE_LOCK")
endif ()

set_source_files_properties(scheduler.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${scheduler_build_options}")

vistk_add_library(vistk_pipeline
  ${pipeline_srcs})
target_link_libraries(vistk_pipeline
  ${Boost_CHRONO_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${CMAKE_DL_LIBS}
  ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(vistk_pipeline LINK_INTERFACE_LIBRARIES)
set_target_properties(vistk_pipeline
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_LIB)

add_dependencies(vistk_pipeline
  configure-version.h)

if (vistk_use_config_subdir)
  add_dependencies(vistk_pipeline
    configure-module-paths.h)
endif ()

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  pipeline)

vistk_install_headers(vistk/pipeline ${pipeline_headers})

vistk_configure_pkgconfig(vistk-pipeline)
