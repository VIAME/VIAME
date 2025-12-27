# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import shutil


def vital_config_update(cfg, cfg_in):
    """
    Treat a vital Config object like a python dictionary

    Args:
        cfg (kwiver.vital.config.config.Config): config to update
        cfg_in (dict | kwiver.vital.config.config.Config): new values
    """
    # vital cfg.merge_config doesnt support dictionary input
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
            else:
                raise KeyError("cfg has no key={}".format(key))
    else:
        cfg.merge_config(cfg_in)
    return cfg


def pad_img_to_fit_bbox(img, x1, y1, x2, y2):
    import cv2

    img = cv2.copyMakeBorder(
        img,
        -min(0, y1),
        max(y2 - img.shape[0], 0),
        -min(0, x1),
        max(x2 - img.shape[1], 0),
        cv2.BORDER_CONSTANT,
    )

    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)

    return img, x1, x2, y1, y2


def safe_crop(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, y1, x2, y2)
    return img[y1:y2, x1:x2, :]


def recurse_copy(src, dst, max_depth=10, ignore=".json"):
    if max_depth < 0:
        return src
    if os.path.isdir(src):
        for entry in os.listdir(src):
            recurse_copy(os.path.join(src, entry), dst, max_depth - 1, ignore)
    elif not src.endswith(ignore):
        shutil.copy2(src, dst)
