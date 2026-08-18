# ============ Native Pipelines ============ #

add_pipeline_script_test(
        test_tracker_calibration_target
        test_tracker.py
        TestTrackerCalibrationTarget
)

add_pipeline_script_test(
        test_tracker_generic_proposals
        test_tracker.py
        TestTrackerGenericProposals
)

# ============ ADD-ON AERIAL-PENGUIN Pipelines ============ #

add_pipeline_script_test(
        test_tracker_penguin_aerial_multi_eo
        test_tracker.py
        TestTrackerPenguinAerialMultiEO
        REQ_ADDON VIAME_DOWNLOAD_MODELS-AERIAL-PENGUIN
)

add_pipeline_script_test(
        test_tracker_penguin_aerial_multi_ir
        test_tracker.py
        TestTrackerPenguinAerialMultiIR
        REQ_ADDON VIAME_DOWNLOAD_MODELS-AERIAL-PENGUIN
)

add_pipeline_script_test(
        test_tracker_penguin_aerial_single_eo
        test_tracker.py
        TestTrackerPenguinAerialSingleEO
        REQ_ADDON VIAME_DOWNLOAD_MODELS-AERIAL-PENGUIN
)

# ============ ADD-ON COMMUNITY-FISH Pipelines ============ #

add_pipeline_script_test(
        test_tracker_community_fish
        test_tracker.py
        TestTrackerCommunityFish
        REQ_ADDON VIAME_DOWNLOAD_MODELS-COMMUNITY-FISH
)

# ============ ADD-ON DEFAULT-FISH Pipelines ============ #

add_pipeline_script_test(
        test_tracker_default_fish
        test_tracker.py
        TestTrackerDefaultFish
        REQ_ADDON VIAME_DOWNLOAD_MODELS-DEFAULT-FISH
)

add_pipeline_script_test(
        test_tracker_fish_sfd
        test_tracker.py
        TestTrackerFishSFD
        REQ_ADDON VIAME_DOWNLOAD_MODELS-DEFAULT-FISH
)

add_pipeline_script_test(
        test_tracker_fish_sfd_cpu
        test_tracker.py
        TestTrackerFishSFDCPU
        REQ_ADDON VIAME_DOWNLOAD_MODELS-DEFAULT-FISH
)

# ============ ADD-ON EM-TUNA Pipelines ============ #

add_pipeline_script_test(
        test_tracker_em_tuna
        test_tracker.py
        TestTrackerEMTuna
        REQ_ADDON VIAME_DOWNLOAD_MODELS-EM-TUNA
)

# ============ ADD-ON GROUPER-MOON Pipelines ============ #

add_pipeline_script_test(
        test_tracker_grouper_moon_v1
        test_tracker.py
        TestTrackerGrouperMoonV1
        REQ_ADDON VIAME_DOWNLOAD_MODELS-GROUPER-MOON
)

# ============ ADD-ON MOTION Pipelines ============ #

add_pipeline_script_test(
        test_tracker_motion
        test_tracker.py
        TestTrackerMotion
        REQ_ADDON VIAME_DOWNLOAD_MODELS-MOTION
)

# ============ ADD-ON MOUSS-DEEP7 Pipelines ============ #

add_pipeline_script_test(
        test_tracker_mouss_deep7_vn
        test_tracker.py
        TestTrackerMoussDeep7VN
        REQ_ADDON VIAME_DOWNLOAD_MODELS-MOUSS-DEEP7
)

# ============ ADD-ON SEA-LION Pipelines ============ #

add_pipeline_script_test(
        test_tracker_sealion_suppressor_fusion_all_class
        test_tracker.py
        TestTrackerSealionSuppressorFusionAllClass
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
)

add_pipeline_script_test(
        test_tracker_sealion_suppressor_fusion_two_class
        test_tracker.py
        TestTrackerSealionSuppressorFusionTwoClass
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
)

add_pipeline_script_test(
        test_tracker_sealion_suppressor_reclassifier_all_class
        test_tracker.py
        TestTrackerSealionSuppressorReclassifierAllClass
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
)

add_pipeline_script_test(
        test_tracker_sealion_tracker_fusion_all_class
        test_tracker.py
        TestTrackerSealionTrackerFusionAllClass
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
)

add_pipeline_script_test(
        test_tracker_sealion_tracker_fusion_two_class
        test_tracker.py
        TestTrackerSealionTrackerFusionTwoClass
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
)

add_pipeline_script_test(
        test_tracker_sealion_tracker_reclassifier_all_class
        test_tracker.py
        TestTrackerSealionTrackerReclassifierAllClass
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SEA-LION
)