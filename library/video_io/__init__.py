# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to video_io."""


# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
# `viame.plugins.discovery` turns each into a stand-in that imports
# its module the first time something asks for an instance.
#
# This replaced a `__vital_algorithm_register__` that imported every
# implementation module at startup, which was the only way the classes came
# to exist for the subclass walk that registers them. See P8-T10.
__vital_algorithm_declarations__ = [
    ( "video_input", "ffmpeg",
      "Read a video with PyAV",
      "viame.video_io.pyav_video_input:FFmpegVideoInput" ),
    ( "video_input", "ffmpeg_cli",
      "Read a video by driving the ffmpeg binary, for when PyAV is absent",
      "viame.video_io.ffmpeg_cli_video_input:FFmpegCliVideoInput" ),
    ( "video_input", "pyav",
      "Read a video with PyAV",
      "viame.video_io.pyav_video_input:PyAVVideoInput" ),
    ( "video_input", "vidl_ffmpeg",
      "Read a video with PyAV",
      "viame.video_io.pyav_video_input:VidlFFmpegVideoInput" ),
    ( "video_output", "ffmpeg",
      "Write a video with PyAV",
      "viame.video_io.pyav_video_output:FFmpegVideoOutput" ),
    ( "video_output", "pyav",
      "Write a video with PyAV",
      "viame.video_io.pyav_video_output:PyAVVideoOutput" ),
]
