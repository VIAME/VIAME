project(vistk_python_util)

set(python_util_srcs
  python_allow_threads.cxx
  python_gil.cxx)

set(python_util_headers
  util-config.h
  python_allow_threads.h
  python_gil.h)

add_library(vistk_python_util
  ${python_util_srcs})
target_link_libraries(vistk_python_util
  ${PYTHON_LIBRARIES})
target_link_libraries(vistk_python_util LINK_INTERFACE_LIBRARIES
  ${PYTHON_LIBRARIES})
set_target_properties(vistk_python_util
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_UTIL_LIB)

install(
  FILES       ${python_util_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/python/util"
  COMPONENT   development)
install(
  TARGETS     vistk_python_util
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)

vistk_configure_pkgconfig(vistk-python_util
  PYTHON_VERSION)
