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
