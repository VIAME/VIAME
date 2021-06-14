# Build / Install plugin containing DBoW2 algorithm implementations + support
# structures

set(CMAKE_FOLDER "Arrows/DBoW2")

set( dbow2_headers_public
  match_descriptor_sets.h
  )
set( dbow2_headers_private
  BowVector.h
  DBoW2.h
  FClass.h
  FeatureVector.h
  FORB.h
  QueryResults.h
  Random.h
  ScoringObject.h
  TemplatedDatabase.h
  TemplatedVocabulary.h
  Timestamp.h
  )

kwiver_install_headers(
  SUBDIR     arrows/dbow2
  ${DBoW2_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_dbow2_export.h
  NOPATH   SUBDIR     arrows/dbow2
  )

set( dbow2_sources
  match_descriptor_sets.cxx
  BowVector.cxx
  FeatureVector.cxx
  FORB.cxx
  QueryResults.cxx
  Random.cxx
  ScoringObject.cxx
  Timestamp.cxx
  )

kwiver_add_library( kwiver_algo_dbow2
  ${dbow2_headers_public}
  ${dbow2_headers_private}
  ${dbow2_sources}
  )
target_link_libraries( kwiver_algo_dbow2
  PUBLIC               vital_algo
                       ${OpenCV_LIBS}
  PRIVATE              kwiversys
  )


algorithms_create_plugin( kwiver_algo_dbow2
  register_algorithms.cxx
  )

# if (KWIVER_ENABLE_TESTS)
#   add_subdirectory(tests)
# endif()
