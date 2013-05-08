project(sprokit_python_helpers)

find_package(PythonInterp ${PYTHON_VERSION})

function (add_python_helper_library name)
  set(helper_sources
    ${name}.cxx
    ${name}.h)
  sprokit_private_header_group(${name}.h)
  sprokit_add_helper_library(${name} helper_sources
    ${ARGN})
  sprokit_require_python_fixes(${name})
endfunction ()

add_python_helper_library(python_exceptions
  ${PYTHON_LIBRARIES})

add_python_helper_library(python_threading
  ${PYTHON_LIBRARIES})
