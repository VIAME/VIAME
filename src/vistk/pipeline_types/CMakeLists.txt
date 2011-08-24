project(vistk_pipeline_types)

set(pipeline_types_srcs
  basic_types.cxx)

set(pipeline_types_headers
  basic_types.h
  pipeline_types-config.h)

add_library(vistk_pipeline_types
  ${pipeline_types_srcs})
target_link_libraries(vistk_pipeline_types
  vistk_pipeline)
target_link_libraries(vistk_pipeline_types LINK_INTERFACE_LIBRARIES
  vistk_pipeline)
set_target_properties(vistk_pipeline_types
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_TYPES_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    pipeline_types
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${pipeline_types_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/pipeline_types"
  COMPONENT   runtime)
install(
  TARGETS     vistk_pipeline_types
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
