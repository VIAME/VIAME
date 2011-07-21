project(vistk_pipeline_util)

set(pipeline_util_srcs
  load_pipe.cxx)

set(pipeline_util_headers
  load_pipe.h
  pipeline_util-config.h)

add_library(vistk_pipeline_util
  ${pipeline_util_srcs})
target_link_libraries(vistk_pipeline_util
  vistk_pipeline)
target_link_libraries(vistk_pipeline_util LINK_INTERFACE_LIBRARIES
  vistk_pipeline)
set_target_properties(vistk_pipeline_util
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PIPELINE_UTIL_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    pipeline_util
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${pipeline_util_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/pipeline_util"
  COMPONENT   runtime)
install(
  TARGETS     vistk_pipeline_util
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
