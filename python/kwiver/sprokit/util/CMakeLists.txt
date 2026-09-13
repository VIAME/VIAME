project(sprokit_python_util)

set(python_util_srcs
    ${python_util_srcs}
    pystream.cxx)

set(python_util_headers
    ${python_util_headers}
    pystream.h)

viame_add_library(sprokit_python_util
  ${python_util_srcs}
  ${python_util_headers})

# `${pybind11_INCLUDE_DIRS}` and the two Boost variables stood here. Nothing
# in the tree sets any of the three any more -- pybind11 is vendored and
# carries its includes on `pybind11::pybind11`, and Boost went with fletch --
# so the Boost entries linked nothing and the pybind11 one was putting the
# reference superbuild's whole `include/` prefix on this target's command
# line, which is how a vendored header gets shadowed by the copy it replaced.
target_link_libraries(sprokit_python_util
  LINK_PUBLIC     ${Python_LIBRARIES}
  LINK_PRIVATE    pybind11::pybind11
                  vital_python_util)

viame_install_headers(
    SUBDIR     sprokit/python/util
    ${python_util_headers}
    )

  # install export header
viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/sprokit_python_util_export.h
  SUBDIR   sprokit/python/util
  NOPATH
  )

if(NOT SKBUILD)
  viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                           sprokit/util
                           __init__)
  viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test.py
                            sprokit/util
                            test)
endif()
