project(vistk_schedulers_examples)

set(examples_srcs
  registration.cxx
  sync_scheduler.cxx
  thread_per_process_scheduler.cxx
  thread_pool_scheduler.cxx)

set(examples_private_headers
  examples-config.h
  registration.h
  sync_scheduler.h
  thread_per_process_scheduler.h
  thread_pool_scheduler.h)

add_vistk_plugin(schedulers_examples
  MAKE_VISTK_SCHEDULERS_EXAMPLES_LIB
  ${examples_srcs}
  ${examples_private_headers})
target_link_libraries(schedulers_examples
  vistk_pipeline
  ${Boost_CHRONO_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT})
