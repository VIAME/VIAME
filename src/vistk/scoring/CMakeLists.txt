project(vistk_scoring)

set(scoring_srcs
  score_mask.cxx
  scoring_result.cxx
  scoring_statistics.cxx
  statistics.cxx)

set(scoring_headers
  score_mask.h
  scoring-config.h
  scoring_result.h
  scoring_statistics.h
  statistics.h)

add_library(vistk_scoring
  ${scoring_srcs})
target_link_libraries(vistk_scoring
  vil)
target_link_libraries(vistk_scoring LINK_INTERFACE_LIBRARIES
  vil)
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

vistk_configure_pkgconfig(vistk-scoring)
