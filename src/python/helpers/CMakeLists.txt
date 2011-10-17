project(vistk_python_helpers)

add_library(pystream STATIC
  pystream.cxx)
target_link_libraries(pystream
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

if (NOT WIN32)
  set_target_properties(pystream
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)

add_library(python_convert_any STATIC
  python_convert_any.cxx)
target_link_libraries(python_convert_any
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

if (NOT WIN32)
  set_target_properties(python_convert_any
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)

add_library(python_gil STATIC
  python_gil.cxx)
target_link_libraries(python_gil
  ${PYTHON_LIBRARIES})

if (NOT WIN32)
  set_target_properties(python_gil
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)
