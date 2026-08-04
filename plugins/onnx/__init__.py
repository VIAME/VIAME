# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""VIAME ONNX plugins (installed as the ``viame.onnx`` package).

Holds the generic onnxruntime object detector plus the epipolar / foundation
stereo ONNX utilities. Registered with kwiver via ``SPROKIT_PYTHON_MODULES``.
"""


def __vital_algorithm_register__():
    """Register vital algorithm implementations in this package."""
    try:
        from viame.onnx import onnx_detector
        onnx_detector.__vital_algorithm_register__()
    except ImportError as ex:
        import warnings
        warnings.warn(f"viame.onnx: could not register onnx detector: {ex}")

    # The foundation-stereo ONNX algorithm ships in this package too; register
    # it when its (heavier) deps import, but never let that block the detector.
    try:
        from viame.onnx import fast_foundation_stereo
        fast_foundation_stereo.__vital_algorithm_register__()
    except Exception:
        pass
