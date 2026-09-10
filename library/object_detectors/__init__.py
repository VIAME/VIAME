# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to object_detectors."""


def __vital_algorithm_register__():
    from viame.object_detectors import hough_circle_detector

    hough_circle_detector.__vital_algorithm_register__()
