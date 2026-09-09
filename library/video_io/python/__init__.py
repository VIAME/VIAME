# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to video_io."""


def __vital_algorithm_register__():
    from viame.video_io import pyav_video_input

    pyav_video_input.__vital_algorithm_register__()
