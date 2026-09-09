"""What the video_input recording covers.

`vidl_ffmpeg` is VXL's video reader. No shipped pipeline names it as a
`:type`, but VIAME's own tooling selects it at run time for every video path
(`tools/run_bulk.py`, `tools/launch_annotator.py`, `tools/train.cxx`,
`plugins/core/utilities_training.cxx`), so it has to keep working.

The variants below are its defaults plus the settings that tooling and the
shipped pipelines actually use: the two `time_source` values that appear in
any config, and the frame selection the trainer applies.
"""

# Each entry: variant id -> config.
VIDEO_INPUTS = {
    "vidl_ffmpeg": [
        ("defaults", {}),
        ("start_at_0", {"time_source": "start_at_0"}),
        ("current", {"time_source": "current"}),
        ("no_metadata", {"use_metadata": "false"}),
        ("stop_after_3", {"stop_after_frame": "3"}),
        ("start_at_3", {"start_at_frame": "3"}),
        ("start_2_stop_5", {"start_at_frame": "2", "stop_after_frame": "5"}),
        ("every_second_frame", {"output_nth_frame": "2"}),
        ("every_third_frame", {"output_nth_frame": "3"}),
    ],
}

# The in-house implementation that has to reproduce it.
VIDEO_REPLACEMENTS = {
    "vidl_ffmpeg": "video_input",
}
