project(vistk_tools)

add_subdirectory(pipelines)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    program_options)

set(boost_header_targets)

option(VISTK_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES "Enables fixes for boost::program_options bugs (recommended)" ON)
mark_as_advanced(VISTK_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
if (VISTK_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
  if (Boost_VERSION LESS 105000)
    # Override Boost's value_semantic.hpp
    set(reldir "helpers/include/value_name")
    set(target_name "boost-program_options-value_semantic.hpp")
    vistk_configure_directory(${target_name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${reldir}/boost"
      "${CMAKE_CURRENT_BINARY_DIR}/${reldir}/${BOOST_MANGLE_NAMESPACE}")
    if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
      include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/${reldir}")
    endif ()
    include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/${reldir}")
    set(boost_header_targets
      ${boost_header_targets}
      ${target_name})
  endif ()
endif ()

add_subdirectory(helpers)

function (add_tool name)
  set(component tool)

  vistk_add_executable(${name}
    ${name}.cxx)
  target_link_libraries(${name}
    LINK_PRIVATE
      ${ARGN})
  if (boost_header_targets)
    add_dependencies(${name}
      ${boost_header_targets})
  endif (boost_header_targets)
endfunction ()

function (add_python_tool name)
  if (VISTK_ENABLE_PYTHON)
    vistk_configure_file(tool-python-${name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py"
      "${vistk_binary_dir}/bin/${name}"
      PYTHON_EXECUTABLE)

    vistk_install(
      PROGRAMS    "${vistk_binary_dir}/bin/${name}"
      DESTINATION bin
      COMPONENT   tool)
  endif ()
endfunction ()

add_tool(pipe_config
  pipeline_builder
  tool_usage
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipe_to_dot
  pipeline_builder
  tool_usage
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(pipeline_runner
  pipeline_builder
  tool_usage
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(processopedia
  tool_usage
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

add_tool(mask_scoring
  pipeline_builder
  tool_usage
  vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
