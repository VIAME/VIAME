project(vistk_python_helpers)

find_package(PythonInterp ${PYTHON_VERSION})

function (add_python_helper_library name)
  vistk_add_helper_library(${name}
    ${ARGN})
  if (boost_header_targets)
    add_dependencies(${name}
      ${boost_header_targets})
  endif (boost_header_targets)
endfunction ()

add_python_helper_library(python_exceptions
  ${PYTHON_LIBRARIES})

add_python_helper_library(pystream
  vistk_python_util
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_python_helper_library(python_threading
  ${PYTHON_LIBRARIES})
