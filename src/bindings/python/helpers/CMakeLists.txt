project(sprokit_python_helpers)

find_package(PythonInterp ${PYTHON_VERSION})

function (add_python_helper_library name)
  set(helper_sources
    ${name}.cxx)
  sprokit_add_helper_library(${name} helper_sources
    ${ARGN})
  if (boost_header_targets)
    add_dependencies(${name}
      ${boost_header_targets})
  endif (boost_header_targets)
endfunction ()

add_python_helper_library(python_exceptions
  ${PYTHON_LIBRARIES})

add_python_helper_library(pystream
  sprokit_python_util
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_python_helper_library(python_threading
  ${PYTHON_LIBRARIES})
