"""Selectable CPU/CUDA implementation of GFIT v3's motion-image preparation."""
import numpy as np
from kwiver.vital.algo import ImageFilter
from kwiver.vital.types import Image, ImageContainer


class GFITMotion(ImageFilter):
    def __init__(self):
        super().__init__()
        self.backend = "auto"
        self._filters = None
        self.device = 0
        self._context = None
        self._input = None
        self._output = None

    def get_configuration(self):
        cfg = super().get_configuration()
        cfg.set_value("device", str(self.device))
        cfg.set_value("backend", self.backend)
        return cfg

    def check_configuration(self, cfg):
        try:
            return (int(cfg.get_value("device", "0")) >= 0 and
                    cfg.get_value("backend", "auto") in ("auto", "cpu", "cuda"))
        except (TypeError, ValueError):
            return False

    def set_configuration(self, cfg):
        if not self.check_configuration(cfg):
            raise ValueError("backend must be auto/cpu/cuda; device must be nonnegative")
        self.device = int(cfg.get_value("device", "0"))
        self.backend = cfg.get_value("backend", "auto")
        self._context = None
        self._filters = None
        self._input = self._output = None
        if self.backend != "cpu":
            try:
                from viame.image_kernels import cuda
                self._context = cuda.Context(self.device)
            except (ImportError, RuntimeError):
                if self.backend == "cuda":
                    raise
        if self._context is None:
            self._filters = []
            for name, options in [
                ("vxl_convert_image", dict(format="byte", single_channel="true")),
                ("vxl_average", dict(type="window", window_size="5", round="false", output_variance="true")),
                ("vxl_average", dict(type="window", window_size="30", round="false", output_variance="true")),
                ("vxl_convert_image", dict(format="byte", scale_factor="0.5")),
            ]:
                algorithm = ImageFilter.create(name)
                config = algorithm.get_configuration()
                for key, value in options.items():
                    config.set_value(key, value)
                algorithm.set_configuration(config)
                self._filters.append(algorithm)

    def filter(self, image_data):
        if image_data is None:
            return None
        if self._context is None and self._filters is None:
            self.set_configuration(self.get_configuration())
        if self._filters is not None:
            grey = self._filters[0].filter(image_data)
            shape = grey.image().asarray().shape[:2]
            short = self._filters[3].filter(self._filters[1].filter(grey)).image().asarray().reshape(shape)
            long = self._filters[3].filter(self._filters[2].filter(grey)).image().asarray().reshape(shape)
            result = np.stack([short, grey.image().asarray().reshape(shape), long], axis=2)
            return ImageContainer(Image(result))
        source = image_data.image().asarray()
        if source.dtype != np.uint8:
            raise TypeError("GFIT CUDA motion input must be uint8")
        shape = source.shape if source.ndim != 3 or source.shape[2] != 1 else source.shape[:2]
        if self._input is not None and self._input.shape != shape:
            self._input = self._output = None
        self._input = self._context.upload(source, out=self._input)
        self._output = self._context.gfit_motion(self._input, out=self._output)
        return ImageContainer(Image(self._context.download(self._output)))


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm
    register_vital_algorithm(GFITMotion, "gfit_motion",
                             "GFIT v3 motion preparation with auto/CPU/CUDA backends")
