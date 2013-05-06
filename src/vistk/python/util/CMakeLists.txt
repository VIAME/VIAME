project(sprokit_python_util)

set(python_util_srcs
  python_allow_threads.cxx
  python_gil.cxx)

set(python_util_headers
  util-config.h
  python_allow_threads.h
  python_gil.h)

sprokit_add_library(sprokit_python_util
  ${python_util_srcs}
  ${python_util_headers})
target_link_libraries(sprokit_python_util
  LINK_PUBLIC
    ${PYTHON_LIBRARIES})
set_target_properties(sprokit_python_util
  PROPERTIES
    VERSION       ${sprokit_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_SPROKIT_PYTHON_UTIL_LIB)

sprokit_install_headers(sprokit/python/util ${python_util_headers})

sprokit_configure_pkgconfig(sprokit-python_util
  PYTHON_VERSION)
