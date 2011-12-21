project(vistk_python_any_conversion)

set(python_any_conversion_srcs
  registration.cxx)

set(python_any_conversion_headers
  any_conversion-config.h
  prototypes.h
  registration.h)

add_library(vistk_python_any_conversion
  ${python_any_conversion_srcs})
target_link_libraries(vistk_python_any_conversion
  ${Boost_THREAD_LIBRARY}
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})
target_link_libraries(vistk_python_any_conversion LINK_INTERFACE_LIBRARIES
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})
set_target_properties(vistk_python_any_conversion
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_ANY_CONVERSION_LIB)

install(
  FILES       ${python_any_conversion_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/python/any_conversion"
  COMPONENT   runtime)
install(
  TARGETS     vistk_python_any_conversion
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
