
set( database_headers
  query_track_descriptor_set_db.h
  write_object_track_set_db.h
  write_track_descriptor_set_db.h
  )

set( database_sources
  query_track_descriptor_set_db.cxx
  write_object_track_set_db.cxx
  write_track_descriptor_set_db.cxx
  )

kwiver_install_headers(
  SUBDIR arrows/database
  ${database_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_database_export.h
  NOPATH SUBDIR arrows/database
  )

kwiver_add_library( kwiver_algo_database
  ${database_headers}
  ${database_sources}
  )

target_link_libraries( kwiver_algo_database
  PUBLIC vital
         vital_algo
  PRIVATE ${CppDB_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_database
  register_algorithms.cxx
  )
