# The pipeline framework
#
# Sprokit, imported from kwiver in P5-T03 and built from here since P5-T05.
# `sprokit_pipeline` and `sprokit_pipeline_util` are two libraries out of one
# directory: kwiver kept them in `pipeline/` and `pipeline_util/`, the import
# flattened the two into this one because no file name is shared, and which
# cuts are worth keeping is phase 8's question.

set( sprokit_source_dir "${CMAKE_CURRENT_SOURCE_DIR}" )
set( sprokit_binary_dir "${CMAKE_CURRENT_BINARY_DIR}" )
set( SPROKIT_CMAKE_DIR  "${KWIVER_CMAKE_DIR}/sprokit" )

set( SPROKIT_ENABLE_PYTHON ${KWIVER_ENABLE_PYTHON} )
set( sprokit_output_dir "${CMAKE_BINARY_DIR}" )

if( SPROKIT_ENABLE_PYTHON )
  add_definitions( "-DSPROKIT_ENABLE_PYTHON" )
endif()

check_cxx_compiler_flag( -fvisibility=hidden SPROKIT_HAVE_GCC_VISIBILITY )

# Sprokit's own macro layer, which its CMakeLists still use. P5-T05 moved it
# to cmake/kwiver_compat/sprokit; phase 8 is where it goes for good.
include( "${SPROKIT_CMAKE_DIR}/sprokit-macro-configure.cmake" )
include( "${SPROKIT_CMAKE_DIR}/sprokit-macro-targets.cmake" )
include( "${SPROKIT_CMAKE_DIR}/sprokit-macro-python.cmake" )
include( "${SPROKIT_CMAKE_DIR}/pkgconfig.cmake" )
include( "${SPROKIT_CMAKE_DIR}/snippet-configuration.cmake" )
include( "${SPROKIT_CMAKE_DIR}/snippet-groups.cmake" )

include_directories( "${CMAKE_CURRENT_BINARY_DIR}" )

###
# The pipeline
##
set( pipeline_srcs
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

set( pipeline_headers
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
  trait_utils.h
  type_traits.h
  )

set( VITAL_VERSION_H "${VIAME_LITE_GENERATED_DIR}/viame/algorithm_framework/version.h" )

set_source_files_properties( ${VITAL_VERSION_H}
  PROPERTIES GENERATED TRUE
  )

include( "${CMAKE_CURRENT_SOURCE_DIR}/thread_naming.cmake" )

set_source_files_properties( utils.cxx
  PROPERTIES COMPILE_DEFINITIONS "${thread_naming_defines}"
  )

set( SPROKIT_DEFAULT_EDGE_CAPACITY "40"
  CACHE STRING "The default capacity for pipeline edges. O is unlmited buffering." )
mark_as_advanced( SPROKIT_DEFAULT_EDGE_CAPACITY )

set_source_files_properties( edge.cxx
  PROPERTIES
  COMPILE_DEFINITIONS "SPROKIT_DEFAULT_EDGE_CAPACITY=${SPROKIT_DEFAULT_EDGE_CAPACITY}"
  )

kwiver_add_library( sprokit_pipeline
  ${pipeline_srcs}
  ${pipeline_headers}
  ${VITAL_VERSION_H}
  )

add_dependencies( sprokit_pipeline configure-version.h configure-viame_version.h )

target_link_libraries( sprokit_pipeline
  PUBLIC              vital_config
                      vital_logger
  PRIVATE             ${CMAKE_DL_LIBS}
                      ${CMAKE_THREAD_LIBS_INIT}
                      vital_vpm kwiversys
  )

if( APPLE AND KWIVER_ENABLE_PYTHON )
  set_target_properties( sprokit_pipeline PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup" )
endif()

viame_lite_generated( sprokit_pipeline_export.h pipeline_framework )

# The namespaced name VIAME's CMakeLists use, as cmake/kwiver_aliases.cmake
# gives the targets `packages/kwiver` still builds.
add_library( kwiver::sprokit_pipeline ALIAS sprokit_pipeline )

###
# The .pipe parser and bakery
##
set( pipeline_util_srcs
  bakery_base.cxx
  bakery_display.cxx
  cluster_bakery.cxx
  cluster_creator.cxx
  cluster_info.cxx
  cluster_splitter.cxx
  export_dot.cxx
  export_dot_exception.cxx
  export_pipe.cxx
  lex_processor.cxx
  load_pipe_exception.cxx
  loaded_cluster.cxx
  pipe_bakery.cxx
  pipe_bakery_exception.cxx
  pipe_display.cxx
  pipe_parser.cxx
  pipeline_builder.cxx
  provided_by_cluster.cxx
  token.cxx
  )

set( pipeline_util_headers
  export_dot.h
  export_dot_exception.h
  export_pipe.h
  literal_pipeline.h
  load_pipe_exception.h
  path.h
  pipe_bakery.h
  pipe_bakery_exception.h
  pipe_declaration_types.h
  bakery_base.h
  bakery_display.h
  cluster_bakery.h
  cluster_creator.h
  cluster_info.h
  cluster_splitter.h
  lex_processor.h
  loaded_cluster.h
  pipe_display.h
  pipe_parser.h
  pipeline_builder.h
  provided_by_cluster.h
  token.h
  )

kwiver_add_library( sprokit_pipeline_util
  ${pipeline_util_srcs}
  ${pipeline_util_headers}
  )

target_link_libraries( sprokit_pipeline_util
  PUBLIC      sprokit_pipeline
  PRIVATE     vital_config
              vital_util
              vital_exceptions
              ${CMAKE_THREAD_LIBS_INIT}
              kwiversys
 )

viame_lite_generated( sprokit_pipeline_util_export.h pipeline_framework )

add_library( kwiver::sprokit_pipeline_util ALIAS sprokit_pipeline_util )

###
# The headers install as a tree, as core_types and algorithm_framework do:
# the set is fixed by the import, not chosen here.
##
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "include/viame/pipeline_framework"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.txx"
  PATTERN "CMakeLists.txt" EXCLUDE )

add_subdirectory( applets )
add_subdirectory( schedulers )
add_subdirectory( adapters )
add_subdirectory( processes )
