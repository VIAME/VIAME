project(vistk_processes_scoring)

set(scoring_srcs
  mask_scoring_process.cxx
  registration.cxx)

add_vistk_plugin(processes_scoring
  MAKE_VISTK_PROCESSES_SCORING_LIB
  ${scoring_srcs})
target_link_libraries(processes_scoring
  vistk_pipeline
  vistk_scoring
  image_helper
  vil)
