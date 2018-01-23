project(sprokit_tools)

sprokit_use_tools_fixes()

function (add_tool       name)
  set(component tool)

  kwiver_add_executable(${name}
    ${name}.cxx)
  target_link_libraries(${name}
    LINK_PRIVATE
      ${ARGN})
  sprokit_require_tools_fixes(${name})
endfunction ()

function (add_python_tool     name)
  if (KWIVER_ENABLE_PYTHON)
    sprokit_configure_file(tool-python-${name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
      "${sprokit_output_dir}/bin/${name}"
      PYTHON_EXECUTABLE)

    sprokit_install(
      PROGRAMS    "${sprokit_output_dir}/bin/${name}"
      DESTINATION bin
      COMPONENT   tool)
  endif ()
endfunction ()

add_tool(pipe_config
  sprokit_tools
  sprokit_pipeline
  sprokit_pipeline_util
  vital_config vital_vpm
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipe_to_dot
  sprokit_tools
  sprokit_pipeline
  vital_config vital_vpm
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipeline_runner
  sprokit_tools
  sprokit_pipeline
  vital_config vital_vpm
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
