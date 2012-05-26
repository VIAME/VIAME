project(vistk_tools)

add_subdirectory(pipelines)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    program_options)

if (WIN32)
  add_definitions(-DBOOST_PROGRAM_OPTIONS_DYN_LINK)
endif (WIN32)

add_subdirectory(helpers)

function (add_tool name)
  add_executable(${name}
    ${name}.cxx)
  target_link_libraries(${name}
    ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      RUNTIME_OUTPUT_PATH "${vistk_binary_dir}/bin")

  install(
    TARGETS     ${name}
    DESTINATION "${EXECUTABLE_INSTALL_PATH}"
    COMPONENT   tools)
endfunction (add_tool)

function (add_python_tool name)
  if (VISTK_ENABLE_PYTHON)
    vistk_configure_file(tool-python-${name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
      "${vistk_binary_dir}/bin/${name}"
      PYTHON_EXECUTABLE)

    install(
      PROGRAMS    "${vistk_binary_dir}/bin/${name}"
      DESTINATION "${EXECUTABLE_INSTALL_PATH}"
      COMPONENT   tools)
  endif (VISTK_ENABLE_PYTHON)
endfunction (add_python_tool)

add_tool(pipe_to_dot
  pipeline_builder
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipeline_runner
  pipeline_builder
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(processopedia
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(mask_scoring
  pipeline_builder
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
