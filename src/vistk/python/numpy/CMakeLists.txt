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

add_library(vistk_python_numpy
  ${python_numpy_srcs})
target_link_libraries(vistk_python_numpy
  vil
  vistk_python_util
  ${Boost_THREAD_LIBRARY}
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})
target_link_libraries(vistk_python_numpy LINK_INTERFACE_LIBRARIES
  vil
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})
set_target_properties(vistk_python_numpy
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_PYTHON_NUMPY_LIB)

install(
  FILES       ${python_numpy_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/python/numpy"
  COMPONENT   development)
install(
  TARGETS     vistk_python_numpy
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)

if (UNIX)
  set(vistk_python_numpy_pkgconfig_file "${CMAKE_BINARY_DIR}/vistk-python_numpy.pc")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/vistk-python_numpy.pc.in"
    "${vistk_python_numpy_pkgconfig_file}"
    @ONLY)
  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${vistk_python_numpy_pkgconfig_file}")

  install(
    FILES       "${vistk_python_numpy_pkgconfig_file}"
    DESTINATION "${LIBRARY_INSTALL_PATH}/pkgconfig"
    COMPONENT   development)
endif (UNIX)
