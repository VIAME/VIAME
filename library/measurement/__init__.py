# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Python implementations that belong to measurement."""


def __vital_algorithm_register__():
    from viame.measurement import (ocv_calibration_targets,
                                   ocv_optimize_stereo_cameras,
                                   ocv_stereo_disparity)

    ocv_stereo_disparity.__vital_algorithm_register__()
    ocv_calibration_targets.__vital_algorithm_register__()
    ocv_optimize_stereo_cameras.__vital_algorithm_register__()
