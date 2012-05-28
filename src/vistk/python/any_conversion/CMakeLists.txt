project(vistk_python_any_conversion)

set(python_any_conversion_srcs
  registration.cxx)

set(python_any_conversion_headers
  any_conversion-config.h
  prototypes.h
  registration.h)

vistk_add_library(vistk_python_any_conversion
  ${python_any_conversion_srcs})
target_link_libraries(vistk_python_any_conversion
  vistk_python_util
  ${Boost_THREAD_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})
target_link_libraries(vistk_python_any_conversion LINK_INTERFACE_LIBRARIES
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})
set_target_properties(vistk_python_any_conversion
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_ANY_CONVERSION_LIB)

vistk_install_headers(vistk/python/any_conversion ${python_any_conversion_headers})

vistk_configure_pkgconfig(vistk-python_any_conversion
  PYTHON_VERSION)
