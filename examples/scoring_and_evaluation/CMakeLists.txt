###
# Scoring and ROC generation examples
##

set( example_files
  detections.csv
  groundtruth.csv )

if( VIAME_ENABLE_PYTHON )
  if( WIN32 )
    set( example_files
         ${example_files}
         track_mot_stats_across_all.bat
         track_mot_stats_per_category.bat )
  else()
    set( example_files
         ${example_files}
         track_mot_stats_across_all.sh
         track_mot_stats_per_category.sh )
  endif()
endif()

if( VIAME_ENABLE_PYTORCH-NETHARN )
  if( WIN32 )
    set( example_files
         ${example_files}
         detection_prcs_and_conf_mat_across_all.bat
         detection_prcs_and_conf_mat_per_category.bat )
  else()
    set( example_files
         ${example_files}
         detection_prcs_and_conf_mat_across_all.sh
         detection_prcs_and_conf_mat_per_category.sh )
  endif()
endif()

install( FILES       ${example_files}
         DESTINATION examples/scoring_and_evaluation )
