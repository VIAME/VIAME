# ============ Native Pipelines ============ #

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