# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageFilter

from kwiver.vital.types import Image
from kwiver.vital.types import ImageContainer

import numpy as np


class OpticalFlowFilter(ImageFilter):
    """
    Dense optical-flow image filter.

    Computes Farneback optical flow between the current and previous frame and
    emits it as a byte image. Output is either single-channel flow magnitude or a
    two-channel (dx, dy) vector field (offset-encoded about 128). Optional
    background compensation subtracts the per-frame median flow so only motion
    relative to the scene survives -- useful for small targets over a moving
    background (e.g. water). State is held across calls, so frames must be fed in
    temporal order.

    Configuration:
      - output: "magnitude" (1 channel) or "vector" (2 channels), default magnitude
      - scale: flow magnitude in pixels mapped to the byte extreme, default 8.0
      - compensate_background: subtract the median flow, default True
    """

    def __init__(self):
        ImageFilter.__init__(self)
        self.output = "magnitude"
        self.scale = 8.0
        self.compensate_background = True
        self._prev = None

    def get_configuration(self):
        cfg = super(ImageFilter, self).get_configuration()
        cfg.set_value("output", self.output)
        cfg.set_value("scale", str(self.scale))
        cfg.set_value("compensate_background", str(self.compensate_background))
        return cfg

    def set_configuration(self, cfg_in):
        self.output = cfg_in.get_value("output")
        self.scale = float(cfg_in.get_value("scale"))
        self.compensate_background = str(
            cfg_in.get_value("compensate_background")
        ).lower() in ("true", "1", "yes")

    def check_configuration(self, cfg):
        if cfg.get_value("output") not in ("magnitude", "vector"):
            print("Error: output must be 'magnitude' or 'vector'")
            return False
        if float(cfg.get_value("scale")) <= 0.0:
            print("Error: scale must be positive")
            return False
        return True

    def filter(self, in_img):
        import cv2

        arr = in_img.image().asarray()

        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 3:
            gray = arr[..., 0].astype(np.uint8)
        else:
            gray = arr.astype(np.uint8)

        if self._prev is None or self._prev.shape != gray.shape:
            flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        self._prev = gray

        if self.compensate_background:
            flow[..., 0] -= np.median(flow[..., 0])
            flow[..., 1] -= np.median(flow[..., 1])

        if self.output == "vector":
            encoded = np.clip(128.0 + flow * (127.0 / self.scale), 0, 255).astype(
                np.uint8
            )
        else:
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            encoded = np.clip(mag * (255.0 / self.scale), 0, 255).astype(np.uint8)

        return ImageContainer(Image(encoded))


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OpticalFlowFilter,
        "ocv_optical_flow",
        "Dense Farneback optical-flow image filter",
    )
