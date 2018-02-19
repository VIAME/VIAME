
set( database_headers
  descriptor_db.h
  connection/db_connection.h
  )

set( database_sources
  descriptor_db.cxx
  connection/db_connection.cxx
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
  PRIVATE ${CppDB_LIBRARY}
  )


if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
