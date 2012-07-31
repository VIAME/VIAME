project(vistk_python_helpers)

find_package(PythonInterp ${PYTHON_VERSION})

find_package(NumPy)

if (NUMPY_FOUND)
  add_library(numpy_memory_chunk STATIC
    numpy_memory_chunk.cxx)
  target_link_libraries(numpy_memory_chunk
    vistk_python_util
    vil
    ${PYTHON_LIBRARIES})

  if (NOT WIN32)
    set_target_properties(numpy_memory_chunk
      PROPERTIES
        COMPILE_FLAGS "-fPIC")
  endif ()
endif ()

add_library(python_exceptions STATIC
  python_exceptions.cxx)
target_link_libraries(python_exceptions
  ${PYTHON_LIBRARIES})

if (NOT WIN32)
  set_target_properties(python_exceptions
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif ()

add_library(pystream STATIC
  pystream.cxx)
target_link_libraries(pystream
  vistk_python_util
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

if (NOT WIN32)
  set_target_properties(pystream
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif ()

add_library(python_threading STATIC
  python_threading.cxx)
target_link_libraries(python_threading
  ${PYTHON_LIBRARIES})

if (NOT WIN32)
  set_target_properties(python_threading
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif ()
