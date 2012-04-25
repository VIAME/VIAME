project(vistk_scoring)

set(scoring_srcs
  scoring_result.cxx)

set(scoring_headers
  scoring-config.h
  scoring_result.h)

add_library(vistk_scoring
  ${scoring_srcs})
set_target_properties(vistk_scoring
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_SCORING_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    scoring)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${scoring_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/scoring"
  COMPONENT   development)
install(
  TARGETS     vistk_scoring
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)

if (UNIX)
  set(vistk_scoring_pkgconfig_file "${CMAKE_BINARY_DIR}/vistk-scoring.pc")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/vistk-scoring.pc.in"
    "${vistk_scoring_pkgconfig_file}"
    @ONLY)
  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${vistk_scoring_pkgconfig_file}")

  install(
    FILES       "${vistk_scoring_pkgconfig_file}"
    DESTINATION "${LIBRARY_INSTALL_PATH}/pkgconfig"
    COMPONENT   development)
endif (UNIX)
