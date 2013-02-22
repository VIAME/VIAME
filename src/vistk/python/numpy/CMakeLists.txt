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

set(python_numpy_private_headers
  numpy_memory_chunk.h
  numpy_support.h
  python_wrap_vil_smart_ptr.h
  type_mappings.h)

if (NUMPY_VERSION VERSION_GREATER "1.7.0" OR
    NUMPY_VERSION VERSION_EQUAL "1.7.0")
  add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endif ()

# NumPy is not imported from this library, so use an external symbol table
# instead of embedding the table here.
add_definitions(-DNO_IMPORT_ARRAY)
add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=vistk_ARRAY_API)

vistk_private_header_group(${python_numpy_private_headers})
vistk_add_library(vistk_python_numpy
  ${python_numpy_srcs}
  ${python_numpy_headers}
  ${python_numpy_private_headers})
target_link_libraries(vistk_python_numpy
  LINK_PUBLIC
    vil
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES}
  LINK_PRIVATE
    vistk_python_util
    ${Boost_THREAD_LIBRARY}
    ${Boost_CHRONO_LIBRARY}
      ${Boost_SYSTEM_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(vistk_python_numpy
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_NUMPY_LIB)

vistk_install_headers(vistk/python/numpy ${python_numpy_headers})

vistk_configure_pkgconfig(vistk-python_numpy
  PYTHON_VERSION)
