project(vistk_pipeline)

set(pipeline_srcs
  config.cxx
  datum.cxx
  dtor_registry.cxx
  dtor_registry_exception.cxx
  edge.cxx
  edge_exception.cxx
  modules.cxx
  pipeline.cxx
  pipeline_exception.cxx
  process.cxx
  process_exception.cxx
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
  dtor_registry.h
  dtor_registry_exception.h
  edge.h
  edge_exception.h
  modules.h
  pipeline-config.h
  pipeline.h
  pipeline_exception.h
  process.h
  process_exception.h
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
else (WIN32)
  set(libdir lib${LIB_SUFFIX})
endif (WIN32)

set(vistk_default_module_paths
  "${CMAKE_INSTALL_PREFIX}/${libdir}/vistk")

if (WIN32)
  set(vistk_default_module_paths
    "${vistk_default_module_paths};${module_output_path}/$<CONFIGURATION>")
else (WIN32)
  set(vistk_default_module_paths
    "${vistk_default_module_paths}:${module_output_path}")
endif (WIN32)

set(vistk_path_prefix)

if (WIN32)
  set(vistk_path_prefix "L")
endif (WIN32)

set(VISTK_DEFAULT_MODULE_PATHS "${vistk_default_module_paths}"
  CACHE PATH "The default paths to search for modules in")
set(vistk_library_options
  "VISTK_DEFAULT_MODULE_PATHS=${vistk_path_prefix}\"${VISTK_DEFAULT_MODULE_PATHS}\"")

set(module_build_options
  ${vistk_library_options})

set_source_files_properties(modules.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${module_build_options}")

set(utils_build_options)

include(CheckFunctionExists)
check_function_exists(setproctitle HAVE_SETPROCTITLE)

if (HAVE_SETPROCTITLE)
  set(utils_build_options
    "HAVE_SETPROCTITLE")
endif (HAVE_SETPROCTITLE)

set_source_files_properties(utils.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${utils_build_options}")

vistk_add_library(vistk_pipeline
  ${pipeline_srcs})
target_link_libraries(vistk_pipeline
  ${Boost_DATE_TIME_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY})
if (NOT WIN32)
  target_link_libraries(vistk_pipeline
    ${CMAKE_DL_LIBS})
endif (NOT WIN32)
target_link_libraries(vistk_pipeline LINK_INTERFACE_LIBRARIES)
set_target_properties(vistk_pipeline
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)

vistk_install_headers(vistk/pipeline ${pipeline_headers})

vistk_configure_pkgconfig(vistk-pipeline)
