###
#
project(sprokit_pipeline_util)

set(pipeline_util_srcs
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
