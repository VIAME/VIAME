# Build / Install plugin containing DBoW2 algorithm implementations + support
# structures

set( dbow2_headers_public
  bag_of_words_matching.h
  )
set( dbow2_headers_private
	bag_of_words_matching.h
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
	TimeStamp.h
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
	bag_of_words_matching.cxx
	BowVector.cxx
	FeatureVector.cxx
	FORB.cxx
	QueryResults.cxx
	Random.cxx
	ScoringObject.cxx
	TimeStamp.cxx
  )

if( KWIVER_HAS_OPENCV_VER_3 )
  add_definitions(-DKWIVER_HAS_OPENCV_VER_3)
endif()

kwiver_add_library( kwiver_algo_dbow2
  ${dbow2_headers_public}
  ${dbow2_headers_private}
  ${dbow2_sources}
  )
target_link_libraries( kwiver_algo_dbow2
  PUBLIC               kwiver_algo_core
                       ${OpenCV_LIBS}
  PRIVATE              kwiversys
  )


algorithms_create_plugin( kwiver_algo_dbow2
  register_algorithms.cxx
  )

# if (KWIVER_ENABLE_TESTS)
#   add_subdirectory(tests)
# endif()
