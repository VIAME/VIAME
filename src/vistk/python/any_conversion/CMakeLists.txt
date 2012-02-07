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
  COMPONENT   development)
install(
  TARGETS     vistk_python_any_conversion
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)

if (UNIX)
  set(vistk_python_any_conversion_pkgconfig_file "${CMAKE_BINARY_DIR}/vistk-python_any_conversion.pc")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/vistk-python_any_conversion.pc.in"
    "${vistk_python_any_conversion_pkgconfig_file}"
    @ONLY)

  install(
    FILES       "${vistk_python_any_conversion_pkgconfig_file}"
    DESTINATION "${LIBRARY_INSTALL_PATH}/pkgconfig"
    COMPONENT   development)
endif (UNIX)
