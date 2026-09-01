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
      - normalize: "fixed" maps `scale` px to the byte extreme; "adaptive" maps
        the frame's own `adaptive_percentile` of magnitude there instead, floored
        at `adaptive_min_scale` px. Default fixed.
      - adaptive_percentile: percentile taken as full scale, default 99.0
      - adaptive_min_scale: smallest px value allowed as full scale, default 0.5

    A fixed scale suits footage whose motion magnitude is known and stable. It
    serves a mixed corpus poorly: measured across underwater sequences the 95th
    percentile of magnitude spans 0.01-1.24 px, so a scale chosen for the fast
    end leaves the slow end encoded into the bottom percent of the byte range and
    effectively blank. Adaptive spends the range on whatever motion each frame
    actually contains. The floor stops a still frame -- where the percentile is
    ~0 -- from being amplified into pure noise.
    """

    def __init__(self):
        ImageFilter.__init__(self)
        self.output = "magnitude"
        self.scale = 8.0
        self.compensate_background = True
        self.normalize = "fixed"
        self.adaptive_percentile = 99.0
        self.adaptive_min_scale = 0.5
        self._prev = None

    def get_configuration(self):
        cfg = super(ImageFilter, self).get_configuration()
        cfg.set_value("output", self.output)
        cfg.set_value("scale", str(self.scale))
        cfg.set_value("compensate_background", str(self.compensate_background))
        cfg.set_value("normalize", self.normalize)
        cfg.set_value("adaptive_percentile", str(self.adaptive_percentile))
        cfg.set_value("adaptive_min_scale", str(self.adaptive_min_scale))
        return cfg

    def set_configuration(self, cfg_in):
        self.output = cfg_in.get_value("output")
        self.scale = float(cfg_in.get_value("scale"))
        self.compensate_background = str(
            cfg_in.get_value("compensate_background")
        ).lower() in ("true", "1", "yes")
        # Two-argument get_value so a pipe written before these keys existed
        # still configures: absent means keep the default, which is the
        # original fixed-scale behaviour.
        self.normalize = cfg_in.get_value("normalize", self.normalize)
        self.adaptive_percentile = float(
            cfg_in.get_value("adaptive_percentile", str(self.adaptive_percentile))
        )
        self.adaptive_min_scale = float(
            cfg_in.get_value("adaptive_min_scale", str(self.adaptive_min_scale))
        )

    def check_configuration(self, cfg):
        if cfg.get_value("output") not in ("magnitude", "vector"):
            print("Error: output must be 'magnitude' or 'vector'")
            return False
        if float(cfg.get_value("scale")) <= 0.0:
            print("Error: scale must be positive")
            return False
        if cfg.get_value("normalize", "fixed") not in ("fixed", "adaptive"):
            print("Error: normalize must be 'fixed' or 'adaptive'")
            return False
        if not 0.0 < float(cfg.get_value("adaptive_percentile", "99.0")) <= 100.0:
            print("Error: adaptive_percentile must be in (0, 100]")
            return False
        if float(cfg.get_value("adaptive_min_scale", "0.5")) <= 0.0:
            print("Error: adaptive_min_scale must be positive")
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

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        full_scale = self.scale
        if self.normalize == "adaptive":
            full_scale = max(
                float(np.percentile(mag, self.adaptive_percentile)),
                self.adaptive_min_scale,
            )

        if self.output == "vector":
            encoded = np.clip(128.0 + flow * (127.0 / full_scale), 0, 255).astype(
                np.uint8
            )
        else:
            encoded = np.clip(mag * (255.0 / full_scale), 0, 255).astype(np.uint8)

        return ImageContainer(Image(encoded))


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        OpticalFlowFilter,
        "ocv_optical_flow",
        "Dense Farneback optical-flow image filter",
    )
