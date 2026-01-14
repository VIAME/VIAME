# VIAME Examples Tests
#
# Integration tests for VIAME example scripts.
# These tests run the actual example scripts and verify they produce output.
#
# Test Categories:
#   CRITICAL - Essential tests that must pass for a valid build
#   EXAMPLES - Standard example script tests
#   UNIT     - Unit tests (outside examples tree)
#
# To run only critical tests:
#   ctest -L CRITICAL
#
# To run only example tests:
#   ctest -L EXAMPLES
#
# To check if critical tests pass (for CI/CD):
#   ctest -L CRITICAL --output-on-failure

if( VIAME_ENABLE_PYTHON )
  # Use Python interpreter with -m pytest for reliable execution
  if( NOT Python3_EXECUTABLE )
    find_package( Python3 COMPONENTS Interpreter REQUIRED )
  endif()

  # Common test properties
  set( EXAMPLE_TEST_TIMEOUT 600 )

  # Labels for test categories
  set( CRITICAL_TEST_LABELS "CRITICAL;EXAMPLES" )
  set( EXAMPLE_TEST_LABELS "EXAMPLES" )

  # Set up Python path for tests - use the actual install prefix
  set( VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}" )
  set( VIAME_PYTHON_PATH "${VIAME_INSTALL_DIR}/python" )
  set( VIAME_PYTHON_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages" )

  # Helper function to add a standard example script test
  function( add_example_script_test TEST_NAME TEST_FILE TEST_CLASS WORKING_DIR )
    add_test(
      NAME ${TEST_NAME}
      COMMAND ${Python3_EXECUTABLE} -m pytest
        "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}"
        -v
        --tb=short
        -k "${TEST_CLASS}"
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}/examples/${WORKING_DIR}"
    )
    set_tests_properties( ${TEST_NAME}
      PROPERTIES
        LABELS "${EXAMPLE_TEST_LABELS}"
        TIMEOUT ${EXAMPLE_TEST_TIMEOUT}
        ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH};VIAME_INSTALL=${VIAME_INSTALL_DIR}"
    )
  endfunction()

  # Helper function to add a CRITICAL example script test
  function( add_critical_example_test TEST_NAME TEST_FILE TEST_CLASS WORKING_DIR )
    add_test(
      NAME ${TEST_NAME}
      COMMAND ${Python3_EXECUTABLE} -m pytest
        "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}"
        -v
        --tb=short
        -k "${TEST_CLASS}"
      WORKING_DIRECTORY "${VIAME_INSTALL_DIR}/examples/${WORKING_DIR}"
    )
    set_tests_properties( ${TEST_NAME}
      PROPERTIES
        LABELS "${CRITICAL_TEST_LABELS}"
        TIMEOUT ${EXAMPLE_TEST_TIMEOUT}
        ENVIRONMENT "PYTHONPATH=${VIAME_PYTHON_PATH}:${VIAME_PYTHON_SITE_PACKAGES}:${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH};VIAME_INSTALL=${VIAME_INSTALL_DIR}"
    )
  endfunction()

  # ============================================================================
  # Annotation and Visualization Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:draw_detections_on_frames
    test_annotation_and_visualization.py
    TestDrawDetectionsOnFrames
    annotation_and_visualization
  )
  add_example_script_test(
    viame_examples:extract_chips_from_detections
    test_annotation_and_visualization.py
    TestExtractChipsFromDetections
    annotation_and_visualization
  )
  add_example_script_test(
    viame_examples:extract_frames
    test_annotation_and_visualization.py
    TestExtractFrames
    annotation_and_visualization
  )
  add_example_script_test(
    viame_examples:extract_frames_with_dets_only
    test_annotation_and_visualization.py
    TestExtractFramesWithDetsOnly
    annotation_and_visualization
  )
  add_example_script_test(
    viame_examples:simple_pipeline_display
    test_annotation_and_visualization.py
    TestSimplePipelineDisplay
    annotation_and_visualization
  )

  # ============================================================================
  # Archive Summarization Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:summarize_and_index_videos
    test_archive_summarization.py
    TestSummarizeAndIndexVideos
    archive_summarization
  )
  add_example_script_test(
    viame_examples:summarize_videos
    test_archive_summarization.py
    TestSummarizeVideos
    archive_summarization
  )

  # ============================================================================
  # Building From Source Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:building_example
    test_building_from_source.py
    TestExample
    building_from_source
  )

  # ============================================================================
  # Detection File Conversions Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:bulk_convert_using_pipe
    test_detection_file_conversions.py
    TestBulkConvertUsingPipe
    detection_file_conversions
  )

  # ============================================================================
  # Frame Level Classification Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:continue_training_deep_classifier
    test_frame_level_classification.py
    TestContinueTrainingDeepClassifier
    frame_level_classification
  )
  add_example_script_test(
    viame_examples:run_trained_model_classification
    test_frame_level_classification.py
    TestRunTrainedModel
    frame_level_classification
  )
  add_example_script_test(
    viame_examples:train_deep_frame_classifier
    test_frame_level_classification.py
    TestTrainDeepFrameClassifier
    frame_level_classification
  )
  add_example_script_test(
    viame_examples:train_svm_frame_classifier
    test_frame_level_classification.py
    TestTrainSvmFrameClassifier
    frame_level_classification
  )

  # ============================================================================
  # Hello World Pipeline Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:run_example
    test_example_pipeline.py
    TestRunExample
    example_pipeline
  )
  add_example_script_test(
    viame_examples:run_python_example
    test_example_pipeline.py
    TestRunPythonExample
    example_pipeline
  )

  # ============================================================================
  # Image Enhancement Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:debayer_and_enhance
    test_image_enhancement.py
    TestDebayerAndEnhance
    image_enhancement
  )
  # CRITICAL: viame_examples:enhance
  add_critical_example_test(
    viame_examples:enhance
    test_image_enhancement.py
    TestEnhance
    image_enhancement
  )

  # ============================================================================
  # Object Detection Tests - CRITICAL
  # ============================================================================
  # CRITICAL: viame_examples:run_generic_proposals
  add_critical_example_test(
    viame_examples:run_generic_proposals
    test_object_detection.py
    TestGenericProposalsExample
    object_detection
  )
  # CRITICAL: viame_examples:run_fish_without_motion
  add_critical_example_test(
    viame_examples:run_fish_without_motion
    test_object_detection.py
    TestFishDetectorExample
    object_detection
  )

  # ============================================================================
  # Object Detector Training Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:continue_training_cfrnn
    test_object_detector_training.py
    TestContinueTrainingCfrnn
    object_detector_training
  )
  add_example_script_test(
    viame_examples:run_trained_model_detection
    test_object_detector_training.py
    TestRunTrainedModel
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_netharn_cfrnn_from_habcam_csv
    test_object_detector_training.py
    TestTrainNetharnCfrnnFromHabcamCsv
    object_detector_training
  )
  # CRITICAL: viame_examples:train_netharn_cfrnn_from_viame_csv
  add_critical_example_test(
    viame_examples:train_netharn_cfrnn_from_viame_csv
    test_object_detector_training.py
    TestTrainNetharnCfrnnFromViameCsv
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_mask_rcnn_from_viame_csv
    test_object_detector_training.py
    TestTrainMaskRcnnFromViameCsv
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_motion_cfrnn_from_viame_csv
    test_object_detector_training.py
    TestTrainMotionCfrnnFromViameCsv
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_svm_over_fish_dets_from_viame_csv
    test_object_detector_training.py
    TestTrainSvmOverFishDetsFromViameCsv
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_svm_over_generic_dets_from_viame_csv
    test_object_detector_training.py
    TestTrainSvmOverGenericDetsFromViameCsv
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_yolo_from_habcam_csv
    test_object_detector_training.py
    TestTrainYoloFromHabcamCsv
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_yolo_from_kw18
    test_object_detector_training.py
    TestTrainYoloFromKw18
    object_detector_training
  )
  add_example_script_test(
    viame_examples:train_yolo_from_viame_csv
    test_object_detector_training.py
    TestTrainYoloFromViameCsv
    object_detector_training
  )

  # ============================================================================
  # Object Tracker Training Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:train_st_tracker_viame_csv
    test_object_tracker_training.py
    TestTrainStTrackerViameCsv
    object_tracker_training
  )

  # ============================================================================
  # Object Tracking Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:bulk_run_user_init_tracking
    test_object_tracking.py
    TestBulkRunUserInitTracking
    object_tracking
  )
  add_example_script_test(
    viame_examples:compute_depth_maps_tracking
    test_object_tracking.py
    TestComputeDepthMaps
    object_tracking
  )
  # CRITICAL: viame_examples:run_fish_tracker
  add_critical_example_test(
    viame_examples:run_fish_tracker
    test_object_tracking.py
    TestRunFishTracker
    object_tracking
  )
  # CRITICAL: viame_examples:run_generic_tracker
  add_critical_example_test(
    viame_examples:run_generic_tracker
    test_object_tracking.py
    TestRunGenericTracker
    object_tracking
  )
  add_example_script_test(
    viame_examples:run_simple_tracker
    test_object_tracking.py
    TestRunSimpleTracker
    object_tracking
  )
  add_example_script_test(
    viame_examples:run_track_viewer
    test_object_tracking.py
    TestRunTrackViewer
    object_tracking
  )
  add_example_script_test(
    viame_examples:run_user_init_tracker
    test_object_tracking.py
    TestRunUserInitTracker
    object_tracking
  )

  # ============================================================================
  # Registration and Mosaicing Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:generate_mosaic_for_list
    test_registration_and_mosaicing.py
    TestGenerateMosaicForList
    registration_and_mosaicing
  )
  add_example_script_test(
    viame_examples:generate_mosaics_for_folder
    test_registration_and_mosaicing.py
    TestGenerateMosaicsForFolder
    registration_and_mosaicing
  )
  add_example_script_test(
    viame_examples:generate_mosaics_xcamera_only
    test_registration_and_mosaicing.py
    TestGenerateMosaicsXcameraOnly
    registration_and_mosaicing
  )
  add_example_script_test(
    viame_examples:generate_transform_from_points
    test_registration_and_mosaicing.py
    TestGenerateTransformFromPoints
    registration_and_mosaicing
  )
  add_example_script_test(
    viame_examples:register_eo_ir_per_frame_itk
    test_registration_and_mosaicing.py
    TestRegisterEoIrPerFrameItk
    registration_and_mosaicing
  )
  add_example_script_test(
    viame_examples:register_eo_ir_per_frame_ocv
    test_registration_and_mosaicing.py
    TestRegisterEoIrPerFrameOcv
    registration_and_mosaicing
  )

  # ============================================================================
  # Scoring and Evaluation Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:detection_prcs_and_conf_mat_across_all
    test_scoring_and_evaluation.py
    TestDetectionPrcsAndConfMatAcrossAll
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:detection_prcs_and_conf_mat_per_category
    test_scoring_and_evaluation.py
    TestDetectionPrcsAndConfMatPerCategory
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:detection_rocs_across_all
    test_scoring_and_evaluation.py
    TestDetectionRocsAcrossAll
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:detection_rocs_per_category
    test_scoring_and_evaluation.py
    TestDetectionRocsPerCategory
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:track_kwant_stats_across_all
    test_scoring_and_evaluation.py
    TestTrackKwantStatsAcrossAll
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:track_kwant_stats_per_category
    test_scoring_and_evaluation.py
    TestTrackKwantStatsPerCategory
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:track_mot_stats_across_all
    test_scoring_and_evaluation.py
    TestTrackMotStatsAcrossAll
    scoring_and_evaluation
  )
  add_example_script_test(
    viame_examples:track_mot_stats_per_category
    test_scoring_and_evaluation.py
    TestTrackMotStatsPerCategory
    scoring_and_evaluation
  )

  # ============================================================================
  # Search and Rapid Model Generation Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:create_index_around_detections
    test_search_and_rapid_model_generation.py
    TestCreateIndexAroundDetections
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:create_index_around_existing_detections
    test_search_and_rapid_model_generation.py
    TestCreateIndexAroundExistingDetections
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:create_index_detection_and_tracking
    test_search_and_rapid_model_generation.py
    TestCreateIndexDetectionAndTracking
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:create_index_full_frame_only
    test_search_and_rapid_model_generation.py
    TestCreateIndexFullFrameOnly
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:generate_detections_using_svm_model
    test_search_and_rapid_model_generation.py
    TestGenerateDetectionsUsingSvmModel
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:perform_cli_query
    test_search_and_rapid_model_generation.py
    TestPerformCliQuery
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:process_database_using_svm_model
    test_search_and_rapid_model_generation.py
    TestProcessDatabaseUsingSvmModel
    search_and_rapid_model_generation
  )
  add_example_script_test(
    viame_examples:process_full_frames_using_svm_model
    test_search_and_rapid_model_generation.py
    TestProcessFullFramesUsingSvmModel
    search_and_rapid_model_generation
  )

  # ============================================================================
  # Size Measurement Tests
  # ============================================================================
  add_example_script_test(
    viame_examples:calibrate_cameras
    test_size_measurement.py
    TestCalibrateCameras
    size_measurement
  )
  add_example_script_test(
    viame_examples:compute_depth_maps_measurement
    test_size_measurement.py
    TestComputeDepthMaps
    size_measurement
  )
  add_example_script_test(
    viame_examples:gmm_standalone_tool
    test_size_measurement.py
    TestGmmStandaloneTool
    size_measurement
  )
  add_example_script_test(
    viame_examples:measure_over_manual_annotations
    test_size_measurement.py
    TestMeasureOverManualAnnotations
    size_measurement
  )
  # CRITICAL: viame_examples:measure_via_default_fish
  add_critical_example_test(
    viame_examples:measure_via_default_fish
    test_size_measurement.py
    TestMeasureViaDefaultFish
    size_measurement
  )
  add_example_script_test(
    viame_examples:measure_via_gmm_oriented_boxes
    test_size_measurement.py
    TestMeasureViaGmmOrientedBoxes
    size_measurement
  )

endif()
