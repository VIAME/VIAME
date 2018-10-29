project(sprokit_pipeline)

set(pipeline_srcs
  datum.cxx
  edge.cxx
  edge_exception.cxx
  pipeline.cxx
  pipeline_exception.cxx
  process.cxx
  process_factory.cxx
  process_instrumentation.cxx
  process_exception.cxx
  process_cluster.cxx
  process_cluster_exception.cxx
  process_registry_exception.cxx
  scheduler.cxx
  scheduler_factory.cxx
  scheduler_exception.cxx
  scheduler_registry_exception.cxx
  stamp.cxx
  types.cxx
  utils.cxx
  version.cxx
  )

set(pipeline_headers
  datum.h
  edge.h
  edge_exception.h
  pipeline.h
  pipeline_exception.h
  process.h
  process_factory.h
  process_instrumentation.h
  process_exception.h
  process_cluster.h
  process_cluster_exception.h
  process_registry_exception.h
  scheduler.h
  scheduler_factory.h
  scheduler_exception.h
  scheduler_registry_exception.h
  stamp.h
  types.h
  utils.h
  version.h
  )

set(pipeline_private_headers)

kwiver_make_module_path( ${CMAKE_INSTALL_PREFIX} sprokit )
set( sprokit_default_module_paths ${kwiver_module_path_result} )


if (KWIVER_USE_BUILD_TREE)
  kwiver_make_module_path( ${sprokit_output_dir} sprokit )
  list( INSERT sprokit_default_module_paths 0 "${kwiver_module_path_result}" )
endif ()

set(SPROKIT_DEFAULT_MODULE_PATHS "${sprokit_default_module_paths}"
  CACHE STRING "The default paths to search for modules in" FORCE)

# add all paths to the property
foreach( p IN LISTS   SPROKIT_DEFAULT_MODULE_PATHS )
  kwiver_add_module_path( ${p} )
endforeach(p)

##
set(utils_build_options)

include("${CMAKE_CURRENT_SOURCE_DIR}/thread_naming.cmake")

list(APPEND utils_build_options       ${thread_naming_defines})

set_source_files_properties(utils.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${utils_build_options}")

###
set(scheduler_build_options)

# XXX(Boost): 1.50.0
if (Boost_VERSION LESS 105000)
  list(APPEND scheduler_build_options        BOOST_NO_HAVE_REVERSE_LOCK)
endif ()

set_source_files_properties(scheduler.cxx
  PROPERTIES
  COMPILE_DEFINITIONS "${scheduler_build_options}")

###
set(SPROKIT_DEFAULT_EDGE_CAPACITY "40"
  CACHE STRING "The default capacity for pipeline edges. O is unlmited buffering." )
mark_as_advanced( SPROKIT_DEFAULT_EDGE_CAPACITY )

set_source_files_properties(edge.cxx
  PROPERTIES
  COMPILE_DEFINITIONS "SPROKIT_DEFAULT_EDGE_CAPACITY=${SPROKIT_DEFAULT_EDGE_CAPACITY}" )

sprokit_private_header_group(${pipeline_private_headers})

kwiver_add_library(sprokit_pipeline
  ${pipeline_srcs}
  ${pipeline_headers}
  ${pipeline_private_headers}
  )

target_link_libraries(sprokit_pipeline
  PUBLIC              vital_config
                      vital_logger
                      ${Boost_CHRONO_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY}
  PRIVATE             ${Boost_THREAD_LIBRARY}
                      ${CMAKE_DL_LIBS}
                      ${CMAKE_THREAD_LIBS_INIT}
                      vital_vpm kwiversys
  )

if(APPLE AND KWIVER_ENABLE_PYTHON)
  set_target_properties(sprokit_pipeline PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

kwiver_install_headers(
  ${pipeline_headers}
  SUBDIR sprokit/pipeline
  )

# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/sprokit_pipeline_export.h
  SUBDIR sprokit/pipeline
  NOPATH
  )

sprokit_configure_pkgconfig(sprokit-pipeline)
