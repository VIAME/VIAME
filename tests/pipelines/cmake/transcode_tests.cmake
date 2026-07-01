# ============ Native Pipelines ============ #

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