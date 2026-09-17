# The pipeline framework
#
# Sprokit, imported from kwiver in P5-T03 and built from here since P5-T05.
# `viame_pipeline_framework` and `viame_pipeline_util` are two libraries out of one
# directory: kwiver kept them in `pipeline/` and `pipeline_util/`, the import
# flattened the two into this one because no file name is shared, and which
# cuts are worth keeping is phase 8's question.

set( sprokit_source_dir "${CMAKE_CURRENT_SOURCE_DIR}" )
set( sprokit_binary_dir "${CMAKE_CURRENT_BINARY_DIR}" )
set( SPROKIT_ENABLE_PYTHON ${KWIVER_ENABLE_PYTHON} )
set( sprokit_output_dir "${CMAKE_BINARY_DIR}" )

if( SPROKIT_ENABLE_PYTHON )
  add_definitions( "-DSPROKIT_ENABLE_PYTHON" )
endif()

check_cxx_compiler_flag( -fvisibility=hidden SPROKIT_HAVE_GCC_VISIBILITY )

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

viame_add_library( viame_pipeline_framework
  ${pipeline_srcs}
  ${pipeline_headers}
  ${VITAL_VERSION_H}
  )

add_dependencies( viame_pipeline_framework configure-version.h configure-viame_version.h )

target_link_libraries( viame_pipeline_framework
  PUBLIC              viame_config
                      viame_logger
  PRIVATE             ${CMAKE_DL_LIBS}
                      ${CMAKE_THREAD_LIBS_INIT}
                      viame_plugin
  )

# Its macOS `-undefined dynamic_lookup` link flag is set on `viame_library`
# in `library/algorithm_framework/registry`: an OBJECT library is not linked.

viame_lite_generated( viame_pipeline_framework_export.h pipeline_framework )

# The namespaced name VIAME's CMakeLists use, as cmake/kwiver_aliases.cmake
# gives the targets `packages/kwiver` still builds.
add_library( kwiver::sprokit_pipeline ALIAS viame_pipeline_framework )

###
# The .pipe parser and bakery
##
set( pipeline_util_srcs
  bakery_base.cxx
  lex_processor.cxx
  load_pipe_exception.cxx
  pipe_bakery.cxx
  pipe_bakery_exception.cxx
  pipe_display.cxx
  pipe_parser.cxx
  pipeline_builder.cxx
  token.cxx
  )

set( pipeline_util_headers
  literal_pipeline.h
  load_pipe_exception.h
  path.h
  pipe_bakery.h
  pipe_bakery_exception.h
  pipe_declaration_types.h
  bakery_base.h
  lex_processor.h
  pipe_display.h
  pipe_parser.h
  pipeline_builder.h
  token.h
  )

viame_add_library( viame_pipeline_util
  ${pipeline_util_srcs}
  ${pipeline_util_headers}
  )

target_link_libraries( viame_pipeline_util
  PUBLIC      viame_pipeline_framework
  PRIVATE     viame_config
              viame_util
              viame_exceptions
              ${CMAKE_THREAD_LIBS_INIT}
 )

viame_lite_generated( viame_pipeline_util_export.h pipeline_framework )

add_library( kwiver::sprokit_pipeline_util ALIAS viame_pipeline_util )

###
# The headers install as a tree, as core_types and algorithm_framework do:
# the set is fixed by the import, not chosen here.
##
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "include/viame/pipeline_framework"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.txx"
  PATTERN "CMakeLists.txt" EXCLUDE
  # the unit tests beside the code since P2-T09
  PATTERN "tests" EXCLUDE
  # `.py` sits beside `.h` since P11-T02b, so importing one of them
  # leaves a `__pycache__` in the source tree that this would sweep
  # into `include/` -- and `FILES_MATCHING` still creates the empty
  # directory, as the note above says.
  PATTERN "__pycache__" EXCLUDE )

add_subdirectory( applets )
add_subdirectory( schedulers )
add_subdirectory( adapters )
add_subdirectory( processes )

###
# Python bindings
#
# `viame.pipeline` and the packages around it, moved here from
# `python/kwiver/sprokit` by P11-T02b. In its own file because it is twenty
# module definitions and a library.
##
if( VIAME_ENABLE_PYTHON )
  include( "${CMAKE_CURRENT_SOURCE_DIR}/python.cmake" )
endif()

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
