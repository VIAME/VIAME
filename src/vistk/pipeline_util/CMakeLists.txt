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
  path.h
  pipe_bakery.h
  pipe_bakery_exception.h
  pipe_declaration_types.h
  pipe_grammar.h
  pipeline_util-config.h)

set(pipeline_util_private_headers
  providers.h)

if (WIN32)
  set(path_sep ";")
else ()
  set(path_sep ":")
endif ()

set(vistk_default_pipe_include_paths
  "${CMAKE_INSTALL_PREFIX}/share/vistk/pipelines/include")

cmake_dependent_option(VISTK_USE_BUILD_INCLUDE_PATHS "Whether to use the build tree for the default include path" ON
  VISTK_USE_BUILD_TREE OFF)
mark_as_advanced(VISTK_USE_BUILD_INCLUDE_PATHS)
if (VISTK_USE_BUILD_INCLUDE_PATHS)
  set(vistk_default_pipe_include_paths
    "${vistk_default_pipe_include_paths}${path_sep}${vistk_source_dir}/src/tools/pipelines/include")
endif ()

set(VISTK_DEFAULT_PIPE_INCLUDE_PATHS "${vistk_default_pipe_include_paths}"
  CACHE STRING "The default paths to search for pipe includes in")
set(vistk_pipe_include_options
  "DEFAULT_PIPE_INCLUDE_PATHS=\"${VISTK_DEFAULT_PIPE_INCLUDE_PATHS}\"")

set(load_pipe_build_options)

if (WIN32)
  vistk_configure_file(include-paths.h
    "${CMAKE_CURRENT_SOURCE_DIR}/include-paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include-paths.h"
    VISTK_DEFAULT_PIPE_INCLUDE_PATHS)

  set(pipeline_util_private_headers
    ${pipeline_util_private_headers}
    include-paths.h.in)
else ()
  set(load_pipe_build_options
    ${load_pipe_build_options}
    ${vistk_pipe_include_options})
endif ()

set_source_files_properties(load_pipe.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${load_pipe_build_options}")

vistk_private_header_group(${pipeline_util_private_headers})
vistk_add_library(vistk_pipeline_util
  ${pipeline_util_srcs}
  ${pipeline_util_headers}
  ${pipeline_util_private_headers})
target_link_libraries(vistk_pipeline_util
  LINK_PUBLIC
    vistk_pipeline
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
  LINK_PRIVATE
    ${Boost_CHRONO_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(vistk_pipeline_util
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_UTIL_LIB)

if (WIN32)
  add_dependencies(vistk_pipeline_util
    configure-include-paths.h)
endif ()

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  pipeline_util
  pipeline)

vistk_install_headers(vistk/pipeline_util ${pipeline_util_headers})

vistk_configure_pkgconfig(vistk-pipeline_util)
