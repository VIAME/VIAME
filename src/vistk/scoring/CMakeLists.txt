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

vistk_add_library(vistk_scoring
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

vistk_install_headers(vistk/scoring ${scoring_headers})

vistk_configure_pkgconfig(vistk-scoring)
