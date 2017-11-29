project(sprokit_scoring)

set(scoring_srcs
  statistics.cxx)

set(scoring_headers
  statistics.h)

kwiver_add_library(sprokit_scoring
  ${scoring_srcs}
  ${scoring_headers}
  )


kwiver_install_headers(
  SUBDIR sprokit/scoring
  ${scoring_headers})

# install export header
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/sprokit_scoring_export.h
  SUBDIR   sprokit/scoring
  NOPATH
  )

sprokit_configure_pkgconfig(sprokit-scoring)
