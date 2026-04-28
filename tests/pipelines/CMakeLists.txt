# VIAME Examples Tests
#
# Integration tests for VIAME example scripts.
# These tests run the actual example scripts and verify they produce output.
#
# Test Categories:
#   PIPELINES - Standard pipeline script tests
#   UNIT     - Unit tests (outside pipelines tree)
#
# To run only pipelines tests:
#   ctest -L PIPELINES

if (VIAME_ENABLE_PYTHON)
  # Use Python interpreter with -m pytest for reliable execution
  if (NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif ()

  set(VIAME_INSTALL_DIR "${VIAME_BUILD_INSTALL_PREFIX}")
  set(VIAME_PY_PATH "${VIAME_INSTALL_DIR}/python")
  set(VIAME_SITE_PACKAGES "${VIAME_INSTALL_DIR}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

  if (WIN32)
    set(SEP "\;")
  else ()
    set(SEP ":")
  endif ()

  set(INTERNAL_PYTHONPATH "${VIAME_PY_PATH}${SEP}${VIAME_SITE_PACKAGES}${SEP}${CMAKE_CURRENT_SOURCE_DIR}")

  # Common test properties
  set(PIPELINE_TEST_TIMEOUT 600)
  set(PIPELINES_TEST_LABELS "PIPELINES")

  # Helper function to add a standard pipeline script test
  function(add_pipeline_script_test TEST_NAME TEST_FILE TEST_CLASS)
    add_test(
      NAME "viame_pipelines:${TEST_NAME}"
      COMMAND ${Python3_EXECUTABLE} -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}::${TEST_CLASS}" -v --tb=short
    )
    set_tests_properties("viame_pipelines:${TEST_NAME}" PROPERTIES
      LABELS "${PIPELINES_TEST_LABELS}"
      TIMEOUT ${PIPELINE_TEST_TIMEOUT}
      SKIP_RETURN_CODE 5
      ENVIRONMENT "PYTHONPATH=${INTERNAL_PYTHONPATH}${SEP}$ENV{PYTHONPATH};VIAME_INSTALL=${VIAME_INSTALL_DIR}"
    )
    set(options "")
    set(oneValueArgs REQ_ADDON)
    set(multiValueArgs "")
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(TEST_REQ_ADDON)
      if(NOT ${${TEST_REQ_ADDON}})
        set_tests_properties("viame_pipelines:${TEST_NAME}" PROPERTIES DISABLED TRUE)
      endif()
    endif()
  endfunction()

  # ============ Utility Pipelines ============ #

  add_pipeline_script_test(
    test_utility_add_head_tail_keypoints_from_dets
    test_utility.py
    TestUtilityAddHeadTailKeypointsFromDets
  )

  add_pipeline_script_test(
    test_utility_add_segmentation_watershed
    test_utility.py
    TestUtilityAddSegmentationWatershed
  )

  add_pipeline_script_test(
    test_utility_empty_frame_lbls
    test_utility.py
    TestUtilityEmptyFrameLbls
  )

  add_pipeline_script_test(
    test_utility_max_points_per_poly
    test_utility.py
    TestUtilityMaxPointsPerPoly
  )

  add_pipeline_script_test(
    test_utility_register_frames
    test_utility.py
    TestUtilityRegisterFrames
  )

  add_pipeline_script_test(
    test_utility_remove_dets_in_ignore_regions
    test_utility.py
    TestUtilityRemoveDetsInIgnoreRegions
  )

  add_pipeline_script_test(
    test_utility_add_head_tail_keypoints_sam2
    test_utility.py
    TestUtilityAddHeadTailKeypointsSAM2
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SAM2
  )

  add_pipeline_script_test(
    test_utility_add_segmentation_sam2
    test_utility.py
    TestUtilityAddSegmentationSAM2
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SAM2
  )

  add_pipeline_script_test(
    test_utility_add_segmentation_sam3
    test_utility.py
    TestUtilityAddSegmentationSAM3
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SAM3
  )

  # ============ Detector Pipelines ============ #

  add_pipeline_script_test(
    test_detector_calibration_target
    test_detector.py
    TestDetectorCalibrationTarget
  )

  add_pipeline_script_test(
    test_detector_huggingface_zeroshot
    test_detector.py
    TestDetectorHuggingfaceZeroshot
  )

  add_pipeline_script_test(
    test_detector_simple_hough
    test_detector.py
    TestDetectorSimpleHough
  )

  add_pipeline_script_test(
    test_detector_penguin_aerial_multi_eo
    test_detector.py
    TestDetectorPenguinAerialMultiEO
    REQ_ADDON VIAME_DOWNLOAD_MODELS-AERIAL-PENGUIN
  )

  add_pipeline_script_test(
    test_detector_penguin_aerial_multi_ir
    test_detector.py
    TestDetectorPenguinAerialMultiIR
    REQ_ADDON VIAME_DOWNLOAD_MODELS-AERIAL-PENGUIN
  )

  add_pipeline_script_test(
    test_detector_penguin_aerial_single_eo
    test_detector.py
    TestDetectorPenguinAerialSingleEO
    REQ_ADDON VIAME_DOWNLOAD_MODELS-AERIAL-PENGUIN
  )

  add_pipeline_script_test(
    test_detector_arctic_seal_eo_yolo
    test_detector.py
    TestDetectorArcticSealEOYolo
    REQ_ADDON VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL
  )

  add_pipeline_script_test(
    test_detector_arctic_seal_ir_yolo
    test_detector.py
    TestDetectorArcticSealIRYolo
    REQ_ADDON VIAME_DOWNLOAD_MODELS-ARCTIC-SEAL
  )

  add_pipeline_script_test(
    test_detector_community_fish
    test_detector.py
    TestDetectorCommunityFish
    REQ_ADDON VIAME_DOWNLOAD_MODELS-COMMUNITY-FISH
  )

  add_pipeline_script_test(
    test_detector_default_fish
    test_detector.py
    TestDetectorDefaultFish
    REQ_ADDON VIAME_DOWNLOAD_MODELS-DEFAULT-FISH
  )

  add_pipeline_script_test(
    test_detector_em_tuna
    test_detector.py
    TestDetectorEMTuna
    REQ_ADDON VIAME_DOWNLOAD_MODELS-EM-TUNA
  )

  add_pipeline_script_test(
    test_detector_fish_fusion
    test_detector.py
    TestDetectorFishFusion
    REQ_ADDON VIAME_DOWNLOAD_MODELS-EXTRA-FISH
  )

  add_pipeline_script_test(
    test_detector_fish_with_motion
    test_detector.py
    TestDetectorFishWithMotion
    REQ_ADDON VIAME_DOWNLOAD_MODELS-EXTRA-FISH
  )

  add_pipeline_script_test(
    test_detector_fish_without_motion
    test_detector.py
    TestDetectorFishWithoutMotion
    REQ_ADDON VIAME_DOWNLOAD_MODELS-EXTRA-FISH
  )

  add_pipeline_script_test(
    test_detector_generic_proposals
    test_detector.py
    TestDetectorGenericProposals
    REQ_ADDON VIAME_DOWNLOAD_MODELS-GENERIC
  )

  add_pipeline_script_test(
    test_detector_grouper_moon_v1
    test_detector.py
    TestDetectorGrouperMoonV1
    REQ_ADDON VIAME_DOWNLOAD_MODELS-GROUPER-MOON
  )

  add_pipeline_script_test(
    test_detector_motion
    test_detector.py
    TestDetectorMotion
    REQ_ADDON VIAME_DOWNLOAD_MODELS-MOTION
  )

  add_pipeline_script_test(
    test_detector_mouss_deep7
    test_detector.py
    TestDetectorMoussDeep7
    REQ_ADDON VIAME_DOWNLOAD_MODELS-MOUSS-DEEP7
  )

  add_pipeline_script_test(
    test_detector_sealion_yolo_all_classes
    test_detector.py
    TestDetectorSealionYoloAllClasses
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_background_classifier
    test_detector.py
    TestDetectorSealionBackgroundClassifier
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_cfrnn_all_class
    test_detector.py
    TestDetectorSealionCFRNNAllClass
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_cfrnn_two_class
    test_detector.py
    TestDetectorSealionCFRNNTwoClass
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_fusion_all_class
    test_detector.py
    TestDetectorSealionFusionAllClass
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_fusion_two_class
    test_detector.py
    TestDetectorSealionCFRNNTwoClass
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_mask_rcnn_two_class
    test_detector.py
    TestDetectorSealionMaskRCNNTwoClass
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_sealion_mask_reclassifier_five_class
    test_detector.py
    TestDetectorSealionMaskReclassifierFiveClass
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
  )

  add_pipeline_script_test(
    test_detector_pengcam_kw
    test_detector.py
    TestDetectorPengcamKW
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SWFSC-PENGHEAD
  )

  add_pipeline_script_test(
    test_detector_pengcam_swfsc
    test_detector.py
    TestDetectorPengcamSWFSC
    REQ_ADDON VIAME_DOWNLOAD_MODELS-SWFSC-PENGHEAD
  )

  # ============ Filter Pipelines ============ #

  add_pipeline_script_test(
    test_filter_debayer
    test_filter.py
    TestFilterDebayer
  )

  add_pipeline_script_test(
    test_filter_debayer_and_enhance
    test_filter.py
    TestFilterDebayerAndEnhance
  )

  add_pipeline_script_test(
    test_filter_draw_dets
    test_filter.py
    TestFilterDrawDets
  )

  add_pipeline_script_test(
    test_filter_enhance
    test_filter.py
    TestFilterEnhance
  )

  add_pipeline_script_test(
    test_filter_extract_chips
    test_filter.py
    TestFilterExtractChips
  )

  add_pipeline_script_test(
    test_filter_normalize_16bit
    test_filter.py
    TestFilterNormalize16bit
  )

  add_pipeline_script_test(
    test_filter_split_and_debayer
    test_filter.py
    TestFilterSplitAndDebayer
  )

  add_pipeline_script_test(
    test_filter_split_left_side
    test_filter.py
    TestFilterSplitLeftSide
  )

  add_pipeline_script_test(
    test_filter_split_right_side
    test_filter.py
    TestFilterSplitRightSide
  )

  add_pipeline_script_test(
    test_filter_stereo_depth_map
    test_filter.py
    TestFilterStereoDepthMap
  )

  add_pipeline_script_test(
    test_filter_to_kwa
    test_filter.py
    TestFilterToKWA
  )

  add_pipeline_script_test(
    test_filter_to_video
    test_filter.py
    TestFilterToVideo
  )

  add_pipeline_script_test(
    test_filter_tracks_only
    test_filter.py
    TestFilterTracksOnly
  )

  add_pipeline_script_test(
    test_filter_tracks_only_adjust_csv
    test_filter.py
    TestFilterTracksOnlyAdjustCsv
  )

  # ============ Transcode Pipelines ============ #

  add_pipeline_script_test(
    test_transcode_compress
    test_transcode.py
    TestTranscodeCompress
  )

  add_pipeline_script_test(
    test_transcode_default
    test_transcode.py
    TestTranscodeDefault
  )

  add_pipeline_script_test(
    test_transcode_draw_dets
    test_transcode.py
    TestTranscodeDrawDets
  )

  add_pipeline_script_test(
    test_transcode_tracks_only
    test_transcode.py
    TestTranscodeTracksOnly
  )

endif ()