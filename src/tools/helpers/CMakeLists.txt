project(sprokit_tools_helpers)

function (add_tool_helper_library name)
  set(tool_sources
    ${name}.cxx
    ${name}.h)
  sprokit_private_header_group(${name}.h)
  sprokit_add_helper_library(${name} tool_sources
    ${ARGN})
  sprokit_require_tools_fixes(${name})
endfunction ()

add_tool_helper_library(tool_usage
  ${Boost_PROGRAM_OPTIONS_LIBRARY})

add_dependencies(tool_usage
  configure-version.h)

add_tool_helper_library(tool_io
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool_helper_library(pipeline_builder
  tool_io
  tool_usage
  sprokit_pipeline_util
  sprokit_pipeline
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
