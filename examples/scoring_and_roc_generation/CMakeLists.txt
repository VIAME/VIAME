###
# Scoring and ROC generation examples
##

set( example_files
  detections.csv
  groundtruth.csv )

if( VIAME_ENABLE_KWANT )
  if( WIN32 )
    set( example_files
         ${example_files}
         conf_mat_and_stats_jointly_across_all.bat
         conf_mat_and_stats_per_category.bat
         detection_rocs_jointly_across_all.bat
         detection_rocs_per_category.bat
         track_stats_jointly_across_all.bat
         track_stats_per_category.bat )
  else()
    set( example_files
         ${example_files}
         conf_mat_and_stats_jointly_across_all.sh
         conf_mat_and_stats_per_category.sh
         detection_rocs_jointly_across_all.sh
         detection_rocs_per_category.sh
         track_stats_jointly_across_all.sh
         track_stats_per_category.sh )
  endif()
endif()

install( FILES       ${example_files}
         DESTINATION examples/scoring_and_roc_generation )
