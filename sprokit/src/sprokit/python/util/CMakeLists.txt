project(sprokit_python_util)

set(python_util_srcs
  python_allow_threads.cxx
  python_exceptions.cxx
  python_gil.cxx
  python_threading.cxx)

set(python_util_headers
  python_allow_threads.h
  python_exceptions.h
  python_gil.h
  python.h
  python_threading.h)

if( NOT WIN32 )
  set(python_util_srcs
      ${python_util_srcs}
      pystream.cxx)

  set(python_util_headers
      ${python_util_headers}
      pystream.h)
endif()

kwiver_add_library(sprokit_python_util
  ${python_util_srcs}
  ${python_util_headers})

target_link_libraries(sprokit_python_util
  LINK_PUBLIC     ${PYTHON_LIBRARIES}
  LINK_PRIVATE    ${Boost_IOSTREAMS_LIBRARY}
                  ${Boost_SYSTEM_LIBRARY})

kwiver_install_headers(
    SUBDIR     sprokit/python/util
    ${python_util_headers}
    )

  # install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/sprokit_python_util_export.h
  SUBDIR   sprokit/python/util
  NOPATH
  )

sprokit_configure_pkgconfig(sprokit-python_util      PYTHON_VERSION)
