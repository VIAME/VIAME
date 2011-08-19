project(vistk_schedules_examples)

set(examples_srcs
  registration.cxx
  sync_schedule.cxx
  thread_per_process_schedule.cxx
  thread_pool_schedule.cxx)

add_vistk_plugin(schedules_examples
  MAKE_VISTK_SCHEDULES_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(schedules_examples
  vistk_pipeline
  ${Boost_DATE_TIME_LIBRARY}
  ${Boost_THREAD_LIBRARY})
