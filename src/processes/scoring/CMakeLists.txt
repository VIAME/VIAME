project(vistk_processes_scoring)

set(scoring_srcs
  component_score_json_writer_process.cxx
  mask_scoring_process.cxx
  registration.cxx
  score_aggregation_process.cxx
  score_writer_process.cxx)

add_vistk_plugin(processes_scoring
  MAKE_VISTK_PROCESSES_SCORING_LIB
  ${scoring_srcs})
target_link_libraries(processes_scoring
  vistk_pipeline
  vistk_scoring
  image_helper
  vil)
