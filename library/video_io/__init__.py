# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to video_io."""


def __vital_algorithm_register__():
    from viame.video_io import (ffmpeg_cli_video_input, pil_image_io,
                                pyav_video_input, pyav_video_output)

    pyav_video_input.__vital_algorithm_register__()
    pyav_video_output.__vital_algorithm_register__()
    ffmpeg_cli_video_input.__vital_algorithm_register__()
    pil_image_io.__vital_algorithm_register__()


def __sprokit_register__():
    from viame.video_io import image_viewer

    image_viewer.__sprokit_register__()
