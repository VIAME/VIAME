project(sprokit_tools)

add_subdirectory(pipelines)

find_package(Boost ${sprokit_boost_version} REQUIRED
  COMPONENTS
    program_options)

sprokit_use_tools_fixes()

add_subdirectory(helpers)

function (add_tool name)
  set(component tool)

  sprokit_add_executable(${name}
    ${name}.cxx)
  target_link_libraries(${name}
    LINK_PRIVATE
      ${ARGN})
  sprokit_require_tools_fixes(${name})
endfunction ()

function (add_python_tool name)
  if (SPROKIT_ENABLE_PYTHON)
    sprokit_configure_file(tool-python-${name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
      "${sprokit_binary_dir}/bin/${name}"
      PYTHON_EXECUTABLE)

    sprokit_install(
      PROGRAMS    "${sprokit_binary_dir}/bin/${name}"
      DESTINATION bin
      COMPONENT   tool)
  endif ()
endfunction ()

add_tool(pipe_config
  pipeline_builder
  tool_usage
  sprokit_tools
  sprokit_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipe_to_dot
  pipeline_builder
  tool_usage
  sprokit_tools
  sprokit_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipeline_runner
  pipeline_builder
  tool_usage
  sprokit_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(processopedia
  tool_usage
  sprokit_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
