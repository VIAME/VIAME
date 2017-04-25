###
#
project(sprokit_pipeline_util)

set(pipeline_util_srcs
  bakery_base.cxx
  check_provider.cxx
  cluster_bakery.cxx
  cluster_creator.cxx
  cluster_info.cxx
  cluster_splitter.cxx
  config_provider_sorter.cxx
  ensure_provided.cxx
  export_dot.cxx
  export_dot_exception.cxx
  export_pipe.cxx
  extract_literal_value.cxx
  load_pipe.cxx
  load_pipe_exception.cxx
  loaded_cluster.cxx
  pipe_bakery.cxx
  pipe_bakery_exception.cxx
  pipe_grammar.cxx
  pipeline_builder.cxx
  provided_by_cluster.cxx
  provider_dereferencer.cxx
  providers.cxx
  )

set(pipeline_util_headers
  export_dot.h
  export_dot_exception.h
  export_pipe.h
  literal_pipeline.h
  load_pipe.h
  load_pipe_exception.h
  path.h
  pipe_bakery.h
  pipe_bakery_exception.h
  pipe_declaration_types.h
  pipe_grammar.h
  pipeline_builder.h
  pipeline_util-config.h
  )

set(pipeline_util_private_headers
  bakery_base.h
  check_provider.h
  cluster_bakery.h
  cluster_creator.h
  cluster_info.h
  cluster_splitter.h
  config_provider_sorter.h
  ensure_provided.h
  extract_literal_value.h
  loaded_cluster.h
  provided_by_cluster.h
  provider_dereferencer.h
  providers.h
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

set_source_files_properties(load_pipe.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${load_pipe_build_options}")

sprokit_private_header_group(${pipeline_util_private_headers})

sprokit_add_library(sprokit_pipeline_util
  ${pipeline_util_srcs}
  ${pipeline_util_headers}
  ${pipeline_util_private_headers}
  )


target_link_libraries(sprokit_pipeline_util
  PUBLIC      sprokit_pipeline
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_SYSTEM_LIBRARY}
  PRIVATE     vital_config
              vital_util
            ${Boost_CHRONO_LIBRARY}
            ${Boost_THREAD_LIBRARY}
            ${CMAKE_THREAD_LIBS_INIT}
 )

set_target_properties(sprokit_pipeline_util
  PROPERTIES
    VERSION       ${KWIVER_VERSION}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_SPROKIT_PIPELINE_UTIL_LIB)

if (WIN32)
  add_dependencies(sprokit_pipeline_util
    configure-include-paths.h)
endif ()

sprokit_install_headers(sprokit/pipeline_util ${pipeline_util_headers})

sprokit_configure_pkgconfig(sprokit-pipeline_util)
