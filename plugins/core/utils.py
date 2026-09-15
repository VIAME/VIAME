# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Common utility functions for VIAME Python modules.
"""


def str2bool(v):
    """
    Convert a string representation of a boolean to an actual boolean value.

    Parameters
    ----------
    v : str or bool
        The value to convert. Accepts 'yes', 'true', 't', 'y', '1' for True
        and 'no', 'false', 'f', 'n', '0' for False (case-insensitive).

    Returns
    -------
    bool
        The boolean value.

    Raises
    ------
    ValueError
        If the string cannot be interpreted as a boolean.

    Examples
    --------
    >>> str2bool('yes')
    True
    >>> str2bool('False')
    False
    >>> str2bool(True)
    True
    """
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise TypeError(f'Boolean value expected, got {type(v).__name__}')
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got {v}')


def vital_config_update(cfg, cfg_in):
    """
    Update a vital Config from a dict or another Config.

    Works around vital's merge_config not accepting dictionary input.
    Raises KeyError if cfg_in contains a key cfg does not have.
    """
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
            else:
                raise KeyError(f"cfg has no key={key}")
    else:
        cfg.merge_config(cfg_in)
    return cfg


def image_container_to_uint8_hwc(image_container):
    """KWIVER ImageContainer -> uint8 (H, W, 3) ndarray, replicating grayscale."""
    import numpy as np

    img = image_container.image().asarray().astype("uint8")
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    return img


def read_stereo_calibration(cal_fpath):
    """
    Left focal length, principal point and baseline from a stereo calibration
    file, via viame::core::read_stereo_rig (the viame.core._measurement
    bindings), so .json, .yml/.yaml, .npz, .mat and OpenCV calibration
    directories all work. Returns a dict with focal_length, principal_x,
    principal_y and baseline. The baseline is |T_x| for a horizontal rig,
    else |T|.
    """
    import numpy as np
    from viame.core import _measurement

    cal = _measurement.load_stereo_calibration(cal_fpath)
    k_left = cal["k_left"]  # flat row-major 3x3
    T = cal["translation"]
    baseline = abs(float(T[0]))
    if baseline < 1e-6:
        baseline = float(np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2))
    result = {
        "focal_length": float(k_left[0]),
        "principal_x": float(k_left[2]),
        "principal_y": float(k_left[5]),
        "baseline": baseline,
    }
    print(
        f"Loaded calibration: focal_length={result['focal_length']}, "
        f"baseline={result['baseline']}, principal=({result['principal_x']}, {result['principal_y']})"
    )
    return result
