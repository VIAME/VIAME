###
#
project(sprokit_pipeline_util)

set(pipeline_util_srcs
  bakery_base.cxx
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
  pipe_parser.cxx
  pipeline_builder.cxx
  provided_by_cluster.cxx
  token.cxx
  pipe_display.cxx
  )

set(pipeline_util_headers
  export_dot.h
  export_dot_exception.h
  export_pipe.h
  literal_pipeline.h
  load_pipe_exception.h
  path.h
  pipe_bakery.h
  pipe_bakery_exception.h
  pipe_declaration_types.h
  )

set(pipeline_util_private_headers
  bakery_base.h
  cluster_bakery.h
  cluster_creator.h
  cluster_info.h
  cluster_splitter.h
  lex_processor.h
  loaded_cluster.h
  pipe_parser.h
  pipeline_builder.h
  provided_by_cluster.h
  token.h
  pipe_display.h
  )

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

set(sprokit_default_pipe_include_paths   "${CMAKE_INSTALL_PREFIX}/share/sprokit/pipelines/include")

if (KWIVER_USE_BUILD_TREE)
  set(sprokit_default_pipe_include_paths
    "${sprokit_default_pipe_include_paths}${path_sep}${sprokit_source_dir}/src/tools/pipelines/include")
endif ()

set(SPROKIT_DEFAULT_PIPE_INCLUDE_PATHS "${sprokit_default_pipe_include_paths}"
  CACHE STRING "The default paths to search for pipe includes in")

set(sprokit_pipe_include_options  "DEFAULT_PIPE_INCLUDE_PATHS=\"${SPROKIT_DEFAULT_PIPE_INCLUDE_PATHS}\"")

set(load_pipe_build_options)

if (WIN32)
  sprokit_configure_file(include-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/include-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include-paths.h"
    SPROKIT_DEFAULT_PIPE_INCLUDE_PATHS)

  list(APPEND pipeline_util_private_headers
    include-paths.h.in)
else ()
  list(APPEND load_pipe_build_options
    ${sprokit_pipe_include_options})
endif ()

set_source_files_properties(pipeline_builder.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${load_pipe_build_options}")

kwiver_private_header_group(${pipeline_util_private_headers})

kwiver_add_library(sprokit_pipeline_util
  ${pipeline_util_srcs}
  ${pipeline_util_headers}
  ${pipeline_util_private_headers}
  )

target_link_libraries(sprokit_pipeline_util
  PUBLIC      sprokit_pipeline
            ${Boost_SYSTEM_LIBRARY}
  PRIVATE     vital_config
              vital_util
              vital_exceptions
            ${Boost_CHRONO_LIBRARY}
            ${Boost_THREAD_LIBRARY}
            ${CMAKE_THREAD_LIBS_INIT}
             kwiversys
 )

if (WIN32)
  add_dependencies(sprokit_pipeline_util
    configure-include-paths.h)
endif ()

kwiver_install_headers(
  ${pipeline_util_headers}
  SUBDIR sprokit/pipeline_util
  )

# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/sprokit_pipeline_util_export.h
  SUBDIR   sprokit/pipeline_util
  NOPATH
  )

sprokit_configure_pkgconfig(sprokit-pipeline_util)
