###
# CMake file for building sprokit schedulers.
project(sprokit_schedulers)
set( source
  registration.cxx
  sync_scheduler.cxx
  thread_per_process_scheduler.cxx
  )

set( private_headers
  sync_scheduler.h
  thread_per_process_scheduler.h
  ${CMAKE_CURRENT_BINARY_DIR}/schedulers_export.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_private_header_group(${private_headers})

kwiver_add_plugin( schedulers
  SOURCES         ${source}
                  ${private_headers}
  PRIVATE         sprokit_pipeline
                  vital_config
                  vital_vpm
                  ${Boost_CHRONO_LIBRARY}
                  ${Boost_DATE_TIME_LIBRARY}
                  ${Boost_THREAD_LIBRARY}
                  ${Boost_SYSTEM_LIBRARY}
                  ${CMAKE_THREAD_LIBS_INIT}
  SUBDIR          ${kwiver_plugin_process_subdir}
  )

add_subdirectory(examples)
