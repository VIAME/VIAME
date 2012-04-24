project(vistk_processes_scoring)

set(scoring_srcs
  registration.cxx)

add_vistk_plugin(processes_scoring
  MAKE_VISTK_PROCESSES_SCORING_LIB
  ${scoring_srcs})
target_link_libraries(processes_scoring
  vistk_pipeline
  vil)
