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
  pipeline_util-config.h)

set(vistk_default_pipe_include_paths
  "${CMAKE_INSTALL_PREFIX}/share/vistk/pipelines")

if (WIN32)
  set(vistk_default_pipe_include_paths
    "${vistk_default_pipe_include_paths};${CMAKE_INSTALL_PREFIX}/share/vistk/pipelines/include")
  set(vistk_default_pipe_include_paths
    "${vistk_default_pipe_include_paths};${vistk_source_dir}/src/tools/pipelines/include")
else (WIN32)
  set(vistk_default_pipe_include_paths
    "${vistk_default_pipe_include_paths}:${CMAKE_INSTALL_PREFIX}/share/vistk/pipelines/include")
  set(vistk_default_pipe_include_paths
    "${vistk_default_pipe_include_paths}:${vistk_source_dir}/src/tools/pipelines/include")
endif (WIN32)

set(VISTK_DEFAULT_PIPE_INCLUDE_PATHS "${vistk_default_pipe_include_paths}"
  CACHE STRING "The default paths to search for pipe includes in")
set(vistk_pipe_include_options
  "VISTK_DEFAULT_PIPE_INCLUDE_PATHS=\"${VISTK_DEFAULT_PIPE_INCLUDE_PATHS}\"")

if (WIN32)
  vistk_configure_file(include-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/include-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include-paths.h"
    VISTK_DEFAULT_PIPE_INCLUDE_PATHS)
else (WIN32)
  set(load_pipe_build_options
    ${vistk_pipe_include_options})
endif (WIN32)

set_source_files_properties(load_pipe.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${load_pipe_build_options}")

vistk_add_library(vistk_pipeline_util
  ${pipeline_util_srcs})
target_link_libraries(vistk_pipeline_util
  vistk_pipeline
  ${Boost_CHRONO_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(vistk_pipeline_util LINK_INTERFACE_LIBRARIES
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
set_target_properties(vistk_pipeline_util
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_UTIL_LIB)

if (WIN32)
  add_dependencies(vistk_pipeline_util
    configure-include-paths.h)
endif (WIN32)

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  pipeline_util
  pipeline)

vistk_install_headers(vistk/pipeline_util ${pipeline_util_headers})

vistk_configure_pkgconfig(vistk-pipeline_util)
