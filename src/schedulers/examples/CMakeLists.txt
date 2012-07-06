project(vistk_schedulers_examples)

set(examples_srcs
  registration.cxx
  sync_scheduler.cxx
  thread_per_process_scheduler.cxx
  thread_pool_scheduler.cxx)

add_vistk_plugin(schedulers_examples
  MAKE_VISTK_SCHEDULERS_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(schedulers_examples
  vistk_pipeline
  ${Boost_CHRONO_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
