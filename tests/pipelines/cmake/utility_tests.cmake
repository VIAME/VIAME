# ============ Native Pipelines ============ #

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

# ============ ADD-ON SAM2 Pipelines ============ #

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

# ============ ADD-ON SAM3 Pipelines ============ #

add_pipeline_script_test(
        test_utility_add_segmentation_sam3
        test_utility.py
        TestUtilityAddSegmentationSAM3
        REQ_ADDON VIAME_DOWNLOAD_MODELS-SAM3
)