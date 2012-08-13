project(vistk_processes_scoring)

set(scoring_srcs
  component_score_json_writer_process.cxx
  mask_scoring_process.cxx
  registration.cxx
  score_aggregation_process.cxx
  score_writer_process.cxx)

set(scoring_private_headers
  component_score_json_writer_process.h
  mask_scoring_process.h
  registration.h
  score_aggregation_process.h
  score_writer_process.h
  scoring-config.h)

vistk_private_header_group(${scoring_private_headers})
add_vistk_plugin(processes_scoring
  MAKE_VISTK_PROCESSES_SCORING_LIB
  ${scoring_srcs}
  ${scoring_private_headers})
target_link_libraries(processes_scoring
  vistk_pipeline
  vistk_scoring
  image_helper
  ${Boost_DATE_TIME_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  vil)
