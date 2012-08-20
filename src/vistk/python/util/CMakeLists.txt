project(vistk_python_util)

set(python_util_srcs
  python_allow_threads.cxx
  python_gil.cxx)

set(python_util_headers
  util-config.h
  python_allow_threads.h
  python_gil.h)

vistk_add_library(vistk_python_util
  ${python_util_srcs}
  ${python_util_headers})
target_link_libraries(vistk_python_util
  ${PYTHON_LIBRARIES})
target_link_libraries(vistk_python_util LINK_INTERFACE_LIBRARIES
  ${PYTHON_LIBRARIES})
set_target_properties(vistk_python_util
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_UTIL_LIB)

vistk_install_headers(vistk/python/util ${python_util_headers})

vistk_configure_pkgconfig(vistk-python_util
  PYTHON_VERSION)
