project(sprokit_pipeline)

set(pipeline_srcs
  datum.cxx
  edge.cxx
  edge_exception.cxx
  modules.cxx
  pipeline.cxx
  pipeline_exception.cxx
  process.cxx
  process_exception.cxx
  process_cluster.cxx
  process_cluster_exception.cxx
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
  process_cluster_exception.h
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

set(pipeline_private_headers)

if (WIN32)
  set(libdir bin)
  set(destdir bin)
else ()
  set(libdir lib${LIB_SUFFIX})
  set(destdir lib)
endif ()

set(sprokit_default_module_paths  "${CMAKE_INSTALL_PREFIX}/${libdir}/sprokit")
set(module_output_path            "${sprokit_output_dir}/${destdir}/sprokit")

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

if (KWIVER_USE_BUILD_TREE)
  set(sprokit_default_module_paths "${sprokit_default_module_paths}${path_sep}${module_output_path}")
endif ()

set(path_prefix)

if (WIN32)
  set(path_prefix "L")
endif ()

set(SPROKIT_DEFAULT_MODULE_PATHS "${sprokit_default_module_paths}"
  CACHE STRING "The default paths to search for modules in" FORCE)
set(sprokit_library_options
  "DEFAULT_MODULE_PATHS=${path_prefix}\"${SPROKIT_DEFAULT_MODULE_PATHS}\"")

set(module_build_options
  "LIBRARY_SUFFIX=\"${CMAKE_SHARED_MODULE_SUFFIX}\"")

if (WIN32)
  sprokit_configure_file(module-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/module-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/module-paths.h"
    SPROKIT_DEFAULT_MODULE_PATHS
    path_prefix)

  list(APPEND pipeline_private_headers
    module-paths.h.in)
else ()
  list(APPEND module_build_options
    ${sprokit_library_options})
endif ()

set(default OFF)

if (CMAKE_CONFIGURATION_TYPES)
  set(default ON)
endif ()

##? should this be KWIVER
option(SPROKIT_USE_CONFIGURATION_SUBDIRECTORY
  "Whether to look in the configuration's subdirectory for each module path (e.g. debug, release, ...)" ${default})

if (SPROKIT_USE_CONFIGURATION_SUBDIRECTORY)
  list(APPEND module_build_options
    USE_CONFIGURATION_SUBDIRECTORY)
endif ()

set_source_files_properties(modules.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${module_build_options}")

set(utils_build_options)

include("${CMAKE_CURRENT_SOURCE_DIR}/thread_naming.cmake")

list(APPEND utils_build_options
  ${thread_naming_defines})

set_source_files_properties(utils.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${utils_build_options}")

set(scheduler_build_options)

# XXX(Boost): 1.50.0
if (Boost_VERSION LESS 105000)
  list(APPEND scheduler_build_options
    BOOST_NO_HAVE_REVERSE_LOCK)
endif ()

set_source_files_properties(scheduler.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${scheduler_build_options}")

# XXX: The core library must be build shared because the plugins will otherwise
# each have their own instance of the registries and then nothing else can get
# access to the loaded processes. Splitting out the registries into a separate
# library might be something for the future.
sprokit_private_header_group(${pipeline_private_headers})

sprokit_add_library(sprokit_pipeline SHARED
  ${pipeline_srcs}
  ${pipeline_headers}
  ${pipeline_private_headers})

target_link_libraries(sprokit_pipeline
  PUBLIC              vital_config
                      vital_logger
                    ${Boost_CHRONO_LIBRARY}
                    ${Boost_SYSTEM_LIBRARY}
  PRIVATE           ${Boost_DATE_TIME_LIBRARY}
                    ${Boost_FILESYSTEM_LIBRARY}
                    ${Boost_THREAD_LIBRARY}
                    ${CMAKE_DL_LIBS}
                    ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(sprokit_pipeline
  PROPERTIES
    VERSION       ${KWIVER_VERSION}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_SPROKIT_PIPELINE_LIB)

if(APPLE AND KWIVER_ENABLE_PYTHON)
  set_target_properties(sprokit_pipeline PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

add_dependencies(sprokit_pipeline     configure-version.h)

if (WIN32)
  add_dependencies(sprokit_pipeline   configure-module-paths.h)
endif ()

sprokit_install_headers(sprokit/pipeline ${pipeline_headers})

sprokit_configure_pkgconfig(sprokit-pipeline)
