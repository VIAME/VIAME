# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to image_processing."""


def __vital_algorithm_register__():
    from viame.image_processing import (ocv_estimators, ocv_flann_matcher,
                                        ocv_segmenters, ocv_sift_surf)

    ocv_sift_surf.__vital_algorithm_register__()
    ocv_flann_matcher.__vital_algorithm_register__()
    ocv_estimators.__vital_algorithm_register__()
    ocv_segmenters.__vital_algorithm_register__()
