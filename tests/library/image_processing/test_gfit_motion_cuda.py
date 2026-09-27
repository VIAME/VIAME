"""Exercise the algorithm factory path used by the GFIT CUDA pipelines."""
import numpy as np
import pytest


def test_factory_filter_and_reconfiguration():
    from viame.image_kernels import cuda
    if not cuda.available():
        pytest.skip(cuda.availability_error())
    from viame.modules import modules
    from viame.algo import ImageFilter
    from viame.types import Image, ImageContainer
    modules.load_known_modules()
    algorithm = ImageFilter.create("gfit_motion_cuda")
    cfg = algorithm.get_configuration()
    assert algorithm.check_configuration(cfg)
    algorithm.set_configuration(cfg)
    source = np.full((11, 17, 3), 50, np.uint8)
    first = algorithm.filter(ImageContainer(Image(source))).asarray()
    assert (first[:, :, 1] == 50).all()
    assert not first[:, :, (0, 2)].any()
    algorithm.filter(ImageContainer(Image(source + 50)))
    algorithm.set_configuration(cfg)
    reset = algorithm.filter(ImageContainer(Image(source))).asarray()
    np.testing.assert_array_equal(reset, first)
    cfg.set_value("device", "-1")
    assert not algorithm.check_configuration(cfg)
