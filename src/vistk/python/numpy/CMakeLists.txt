project(vistk_python_numpy)

set(python_numpy_srcs
  import.cxx
  numpy_memory_chunk.cxx
  numpy_to_vil.cxx
  registration.cxx
  vil_to_numpy.cxx)

set(python_numpy_headers
  import.h
  numpy-config.h
  numpy_to_vil.h
  registration.h
  vil_to_numpy.h)

vistk_add_library(vistk_python_numpy
  ${python_numpy_srcs})
target_link_libraries(vistk_python_numpy
  vil
  vistk_python_util
  ${Boost_THREAD_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})
target_link_libraries(vistk_python_numpy LINK_INTERFACE_LIBRARIES
  vil
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})
set_target_properties(vistk_python_numpy
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_NUMPY_LIB)

vistk_install_headers(vistk/python/numpy ${python_numpy_headers})

vistk_configure_pkgconfig(vistk-python_numpy
  PYTHON_VERSION)
