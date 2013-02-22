project(vistk_scoring)

set(scoring_srcs
  statistics.cxx)

set(scoring_headers
  scoring-config.h
  statistics.h)

vistk_add_library(vistk_scoring
  ${scoring_srcs}
  ${scoring_headers})
set_target_properties(vistk_scoring
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_SCORING_LIB)

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  scoring)

vistk_install_headers(vistk/scoring ${scoring_headers})

vistk_configure_pkgconfig(vistk-scoring)
