###
# Application level pipelines
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     query_box.csv
     ingest_list.txt
     query_list.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       create_index.around_detections.bat
       create_index.around_existing_detections.bat
       create_index.full_frame_only.bat
       create_index.detection_and_tracking.bat
       process_database_using_svm_model.bat
       process_full_frames_using_svm_model.bat
       generate_detections_using_svm_model.bat
       launch_search_interface.bat
       launch_view_interface.bat
       perform_cli_query.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       create_index.around_detections.sh
       create_index.around_existing_detections.sh
       create_index.full_frame_only.sh
       create_index.detection_and_tracking.sh
       process_database_using_svm_model.sh
       process_full_frames_using_svm_model.sh
       generate_detections_using_svm_model.sh
       launch_search_interface.sh
       launch_view_interface.sh
       perform_cli_query.sh )
endif()

###
# Install configuration files to examples directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/search_and_rapid_model_generation )
