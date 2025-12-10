# Build / Install Plugin containing SEAGIS measurement implementations

# Process library
set( process_headers
  seagis_measurement_process.h
)

set( process_sources
  register_processes.cxx
  seagis_measurement_process.cxx
)

find_package( Boost ${KWIVER_BOOST_VERSION} REQUIRED
  COMPONENTS
    chrono
    filesystem
    system)

include_directories( SYSTEM ${Boost_INCLUDE_DIRS} )
link_directories( ${Boost_LIBRARY_DIRS} )

# Include SEAGIS headers
include_directories( SYSTEM ${SEAGIS_INCLUDE_DIR} )

set( SEAGIS_LINK_LIBRARIES
  kwiver::vital
  kwiver::vital_algo
  kwiver::vital_config
  kwiver::vital_exceptions
  kwiver::vital_logger
  kwiver::vital_util
  kwiver::sprokit_pipeline
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${SEAGIS_LIBRARY}
)

kwiver_add_plugin( viame_processes_seagis
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          ${SEAGIS_LINK_LIBRARIES}
)
