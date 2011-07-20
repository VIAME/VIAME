project(vistk_pipeline)

set(pipeline_srcs
  config.cxx
  datum.cxx
  edge.cxx
  edge_exception.cxx
  modules.cxx
  pipeline.cxx
  pipeline_exception.cxx
  pipeline_registry.cxx
  pipeline_registry_exception.cxx
  process.cxx
  process_exception.cxx
  process_registry.cxx
  process_registry_exception.cxx
  schedule.cxx
  stamp.cxx
  utils.cxx)

set(pipeline_headers
  config.h
  datum.h
  edge.h
  edge_exception.h
  modules.h
  pipeline-config.h
  pipeline.h
  pipeline_exception.h
  pipeline_registry.h
  pipeline_registry_exception.h
  process.h
  process_exception.h
  process_registry.h
  process_registry_exception.h
  schedule.h
  stamp.h
  types.h
  utils.h)

set(vistk_library_options)

option(VISTK_MODULE_USE_BUILD_PATH "Turn on to search the build path for modules" OFF)
if (VISTK_MODULE_USE_BUILD_PATH)
  set(vistk_library_options
    ${vistk_library_options}
    "VISTK_LIBRARY_OUTPUT_PATH=\"${MODULE_OUTPUT_PATH}\"")
endif (VISTK_MODULE_USE_BUILD_PATH)

option(VISTK_MODULE_USE_INSTALL_PATH "Turn on to search the install path for modules" OFF)
if (VISTK_MODULE_USE_INSTALL_PATH)
  set(vistk_library_options
    ${vistk_library_options}
    "VISTK_MODULE_INSTALL_PATH=\"${MODULE_INSTALL_PATH}\"")
endif (VISTK_MODULE_USE_INSTALL_PATH)

set(module_build_options
  ${boost_filesystem_options}
  ${vistk_library_options})

set_source_files_properties(modules.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${module_build_options}")

set(utils_build_options)

include(CheckFunctionExists)
check_function_exists(setproctitle HAVE_SETPROCTITLE)

if (HAVE_SETPROCTITLE)
  set(utils_build_options
    "HAVE_SETPROCTITLE")
endif (HAVE_SETPROCTITLE)

set_source_files_properties(utils.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${utils_build_options}")

add_library(vistk_pipeline
  ${pipeline_srcs})
target_link_libraries(vistk_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
if (NOT WIN32)
  target_link_libraries(vistk_pipeline
    ${CMAKE_DL_LIBS})
endif (NOT WIN32)
target_link_libraries(vistk_pipeline LINK_INTERFACE_LIBRARIES)
set_target_properties(vistk_pipeline
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${pipeline_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/pipeline"
  COMPONENT   runtime)
install(
  TARGETS     vistk_pipeline
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
