project(sprokit_python_util)

set(python_util_srcs
  python_allow_threads.cxx
  python_exceptions.cxx
  python_gil.cxx
  python_threading.cxx)

set(python_util_headers
  python_allow_threads.h
  python_convert_optional.h
  python_exceptions.h
  python_gil.h
  python.h
  python_threading.h
  python_wrap_const_shared_ptr.h
  set_indexing_suite.h
  util-config.h)

if( NOT WIN32 )
  set(python_util_srcs
      ${python_util_srcs}
      pystream.cxx)

  set(python_util_headers
      ${python_util_headers}
      pystream.h)
endif()

sprokit_add_library(sprokit_python_util
  ${python_util_srcs}
  ${python_util_headers})

target_link_libraries(sprokit_python_util
  LINK_PUBLIC     ${${boost_python_library_var}}
                  ${PYTHON_LIBRARIES}
  LINK_PRIVATE    ${Boost_IOSTREAMS_LIBRARY}
                  ${Boost_SYSTEM_LIBRARY})

  set_target_properties(sprokit_python_util
  PROPERTIES
    VERSION       ${KWIVER_VERSION}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_SPROKIT_PYTHON_UTIL_LIB)

sprokit_install_headers(sprokit/python/util ${python_util_headers})

sprokit_configure_pkgconfig(sprokit-python_util      PYTHON_VERSION)
