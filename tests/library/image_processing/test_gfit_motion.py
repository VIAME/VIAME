"""Exercise the algorithm factory path used by the GFIT CUDA pipelines."""
import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["cpu", "auto", "cuda"])
def test_factory_filter_and_reconfiguration(backend):
    from viame.image_kernels import cuda
    if backend == "cuda" and not cuda.available():
        pytest.skip(cuda.availability_error())
    from viame.modules import modules
    from viame.algo import ImageFilter
    from viame.types import Image, ImageContainer
    modules.load_known_modules()
    algorithm = ImageFilter.create("gfit_motion")
    cfg = algorithm.get_configuration()
    cfg.set_value("backend", backend)
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


def test_auto_motion_falls_back(monkeypatch):
    from viame.image_kernels import cuda
    from viame.modules import modules
    from viame.algo import ImageFilter
    from viame.types import Image, ImageContainer
    modules.load_known_modules()
    def unavailable(*args):
        raise RuntimeError("CUDA extension/device unavailable")
    monkeypatch.setitem(cuda.__dict__, "Context", unavailable)
    algorithms = []
    for backend in ("cpu", "auto"):
        algorithm = ImageFilter.create("gfit_motion")
        cfg = algorithm.get_configuration()
        cfg.set_value("backend", backend)
        algorithm.set_configuration(cfg)
        algorithms.append(algorithm)
    rng = np.random.default_rng(123)
    for _ in range(40):
        image = ImageContainer(Image(rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)))
        np.testing.assert_array_equal(*(a.filter(image).asarray() for a in algorithms))
    cfg.set_value("backend", "cuda")
    with pytest.raises(RuntimeError, match="unavailable"):
        algorithms[1].set_configuration(cfg)
