project(vistk_pipeline_util)

set(pipeline_util_srcs
  export_dot.cxx
  export_dot_exception.cxx
  load_pipe.cxx
  load_pipe_exception.cxx
  pipe_bakery.cxx
  pipe_bakery_exception.cxx
  pipe_grammar.cxx
  providers.cxx)

set(pipeline_util_headers
  export_dot.h
  export_dot_exception.h
  load_pipe.h
  load_pipe_exception.h
  pipe_bakery.h
  pipe_bakery_exception.h
  pipe_declaration_types.h
  pipe_grammar.h
  pipeline_util-config.h
  types.h)

set(vistk_pipe_include_options)

# TODO: Set system pipe include directories

set(load_pipe_build_options
  ${boost_filesystem_options}
  ${vistk_pipe_include_options})

set_source_files_properties(load_pipe.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${load_pipe_build_options}")

set(providers_build_options
  ${boost_filesystem_options})

set_source_files_properties(providers.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${providers_build_options}")

add_library(vistk_pipeline_util
  ${pipeline_util_srcs})
target_link_libraries(vistk_pipeline_util
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY})
target_link_libraries(vistk_pipeline_util LINK_INTERFACE_LIBRARIES
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
set_target_properties(vistk_pipeline_util
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_UTIL_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    pipeline_util
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${pipeline_util_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/pipeline_util"
  COMPONENT   runtime)
install(
  TARGETS     vistk_pipeline_util
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
