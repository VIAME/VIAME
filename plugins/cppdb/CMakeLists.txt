# Build/install plugin for CppDB database support

find_package( CppDB REQUIRED )

set( headers
  query_track_descriptor_set_db.h
  read_object_track_set_db.h
  write_object_track_set_db.h
  write_track_descriptor_set_db.h
  )

set( sources
  query_track_descriptor_set_db.cxx
  read_object_track_set_db.cxx
  write_object_track_set_db.cxx
  write_track_descriptor_set_db.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_cppdb_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_cppdb
  ${headers}
  ${sources}
  )

target_include_directories( viame_cppdb PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_cppdb
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
  PRIVATE              ${CppDB_LIBRARY}
  )

set_target_properties( viame_cppdb PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_cppdb
  register_algorithms.cxx
  )

target_link_libraries( viame_cppdb_plugin
  PUBLIC               kwiver::vital_vpm
  )

# Add process library for database-backed sprokit processes
set( process_headers
  ingest_descriptors_db_process.h
  fetch_descriptors_db_process.h
  object_track_descriptors_db_process.h
)

set( process_sources
  register_processes.cxx
  ingest_descriptors_db_process.cxx
  fetch_descriptors_db_process.cxx
  object_track_descriptors_db_process.cxx
)

kwiver_add_plugin( viame_processes_cppdb
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
  PRIVATE          kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_config
                   kwiver::vital_logger
                   kwiver::vital_util
                   kwiver::vital_vpm
                   kwiver::sprokit_pipeline
                   ${CppDB_LIBRARY}
 )

target_include_directories( viame_processes_cppdb PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
