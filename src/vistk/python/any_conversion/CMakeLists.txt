project(vistk_python_any_conversion)

set(python_any_conversion_srcs
  registration.cxx)

set(python_any_conversion_headers
  any_conversion-config.h
  prototypes.h
  registration.h)

# XXX: Force building shared (see src/vistk/pipeline/CMakeLists.txt as to why
# this must be done).
vistk_add_library(vistk_python_any_conversion SHARED
  ${python_any_conversion_srcs}
  ${python_any_conversion_headers})
target_link_libraries(vistk_python_any_conversion
  vistk_python_util
  ${Boost_THREAD_LIBRARY}
  ${Boost_CHRONO_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT})
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
