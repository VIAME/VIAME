project(sprokit_schedulers_examples)

set(examples_srcs
  registration.cxx
  thread_pool_scheduler.cxx)

set(examples_private_headers
  thread_pool_scheduler.h
  ${CMAKE_CURRENT_BINARY_DIR}/schedulers_examples_export.h
  )

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_private_header_group(${examples_private_headers})

kwiver_add_plugin( schedulers_examples
  SOURCES        ${examples_srcs}
                 ${examples_private_headers}
  PRIVATE        sprokit_pipeline
                 vital_config
                 vital_vpm
                 ${Boost_CHRONO_LIBRARY}
                 ${Boost_DATE_TIME_LIBRARY}
                 ${Boost_THREAD_LIBRARY}
                 ${Boost_SYSTEM_LIBRARY}
                 ${CMAKE_THREAD_LIBS_INIT}
  SUBDIR         ${kwiver_plugin_scheduler_subdir}
  )
