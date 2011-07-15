project(vistk_pipeline)

set(pipeline_srcs
  config.cxx
  datum.cxx
  edge.cxx
  edge_exception.cxx
  edge_registry.cxx
  edge_registry_exception.cxx
  modules.cxx
  pipeline.cxx
  pipeline_exception.cxx
  pipeline_registry.cxx
  pipeline_registry_exception.cxx
  process.cxx
  process_exception.cxx
  process_registry.cxx
  process_registry_exception.cxx
  stamp.cxx)

set(pipeline_headers
  config.h
  datum.h
  edge.h
  edge_exception.h
  edge_registry.h
  edge_registry_exception.h
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
  stamp.h
  types.h)

add_library(vistk_pipeline
  ${pipeline_srcs})
target_link_libraries(vistk_pipeline
  ${CMAKE_DL_LIBS})
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
