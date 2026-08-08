# ============ Native Pipelines ============ #

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

# ============ ADD-ON AERIAL-PENGUIN Pipelines ============ #

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

# ============ ADD-ON ARCTIC-SEAL Pipelines ============ #

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

# ============ ADD-ON COMMUNITY-FISH Pipelines ============ #

add_pipeline_script_test(
        test_detector_community_fish
        test_detector.py
        TestDetectorCommunityFish
        REQ_ADDON VIAME_DOWNLOAD_MODELS-COMMUNITY-FISH
)

# ============ ADD-ON DEFAULT-FISH Pipelines ============ #

add_pipeline_script_test(
        test_detector_default_fish
        test_detector.py
        TestDetectorDefaultFish
        REQ_ADDON VIAME_DOWNLOAD_MODELS-DEFAULT-FISH
)

# ============ ADD-ON EM-TUNA Pipelines ============ #

add_pipeline_script_test(
        test_detector_em_tuna
        test_detector.py
        TestDetectorEMTuna
        REQ_ADDON VIAME_DOWNLOAD_MODELS-EM-TUNA
)

# ============ ADD-ON EXTRA-FISH Pipelines ============ #

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

# ============ ADD-ON GENERIC Pipelines ============ #

add_pipeline_script_test(
        test_detector_generic_proposals
        test_detector.py
        TestDetectorGenericProposals
        REQ_ADDON VIAME_DOWNLOAD_MODELS-GENERIC
)

# ============ ADD-ON GROUPER-MOON Pipelines ============ #

add_pipeline_script_test(
        test_detector_grouper_moon_v1
        test_detector.py
        TestDetectorGrouperMoonV1
        REQ_ADDON VIAME_DOWNLOAD_MODELS-GROUPER-MOON
)

# ============ ADD-ON MOTION Pipelines ============ #

add_pipeline_script_test(
        test_detector_motion
        test_detector.py
        TestDetectorMotion
        REQ_ADDON VIAME_DOWNLOAD_MODELS-MOTION
)

# ============ ADD-ON MOUSS-DEEP7 Pipelines ============ #

add_pipeline_script_test(
        test_detector_mouss_deep7
        test_detector.py
        TestDetectorMoussDeep7
        REQ_ADDON VIAME_DOWNLOAD_MODELS-MOUSS-DEEP7
)

# ============ ADD-ON SEA-LION Pipelines ============ #

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

# ============ ADD-ON SWFSC-PENGHEAD Pipelines ============ #

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