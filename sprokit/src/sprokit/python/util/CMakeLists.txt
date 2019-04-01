project(sprokit_python_util)

set(python_util_srcs
    ${python_util_srcs}
    pystream.cxx)

set(python_util_headers
    ${python_util_headers}
    pystream.h)

kwiver_add_library(sprokit_python_util
  ${python_util_srcs}
  ${python_util_headers})

target_link_libraries(sprokit_python_util
  LINK_PUBLIC     ${PYTHON_LIBRARIES}
  LINK_PRIVATE    ${Boost_IOSTREAMS_LIBRARY}
                  ${Boost_SYSTEM_LIBRARY}
                  vital_python_util)

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

