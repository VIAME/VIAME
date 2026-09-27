"""Fused CUDA implementation of GFIT v3's motion-image preparation."""
import numpy as np
from viame.algo import ImageFilter
from viame.types import Image, ImageContainer


class GFITMotionCUDA(ImageFilter):
    def __init__(self):
        super().__init__()
        self.device = 0
        self._context = None
        self._input = None
        self._output = None

    def get_configuration(self):
        cfg = super().get_configuration()
        cfg.set_value("device", str(self.device))
        return cfg

    def check_configuration(self, cfg):
        try:
            return int(cfg.get_value("device", "0")) >= 0
        except (TypeError, ValueError):
            return False

    def set_configuration(self, cfg):
        from viame.image_kernels import cuda
        if not self.check_configuration(cfg):
            raise ValueError("GFIT CUDA device must be a nonnegative integer")
        self.device = int(cfg.get_value("device", "0"))
        self._context = cuda.Context(self.device)
        self._input = self._output = None

    def filter(self, image_data):
        if image_data is None:
            return None
        if self._context is None:
            self.set_configuration(self.get_configuration())
        source = image_data.asarray()
        if source.dtype != np.uint8:
            raise TypeError("GFIT CUDA motion input must be uint8")
        shape = source.shape if source.ndim != 3 or source.shape[2] != 1 else source.shape[:2]
        if self._input is not None and self._input.shape != shape:
            self._input = self._output = None
        self._input = self._context.upload(source, out=self._input)
        self._output = self._context.gfit_motion(self._input, out=self._output)
        return ImageContainer(Image(self._context.download(self._output)))


def __vital_algorithm_register__():
    from viame.utilities.vital_registration import register_vital_algorithm
    register_vital_algorithm(GFITMotionCUDA, "gfit_motion_cuda",
                             "GFIT v3 motion preparation on optional CUDA kernels")
