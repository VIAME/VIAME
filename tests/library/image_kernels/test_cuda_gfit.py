"""GFIT CPU/GPU parity, including its historical temporal update semantics."""
import gc
import numpy as np
import pytest
from viame.image_kernels import cuda


@pytest.fixture
def context():
    if not cuda.available():
        pytest.skip(cuda.availability_error())
    return cuda.Context()


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (13, 1), (7, 11), (111, 319), (449, 313), (448, 448)])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("dsize", [(224, 224), (19, 13)])
def test_letterbox(context, shape, channels, dsize):
    kwimage = pytest.importorskip("kwimage")
    source = np.random.default_rng(8).integers(0, 256, (*shape, channels), dtype=np.uint8)
    expected = kwimage.imresize(source, dsize=dsize, letterbox=True)
    if expected.ndim == 3 and expected.shape[2] == 1:
        expected = expected[:, :, 0]
    actual = context.download(context.resize_letterbox(context.upload(source), *dsize))
    np.testing.assert_array_equal(actual, expected)


def cpu_filters():
    from viame.modules import modules
    from viame.algo import ImageFilter
    modules.load_known_modules()
    filters = []
    for name, values in [
        ("vxl_convert_image", {"format": "byte", "single_channel": "true"}),
        ("vxl_average", {"type": "window", "window_size": "5", "round": "false", "output_variance": "true"}),
        ("vxl_average", {"type": "window", "window_size": "30", "round": "false", "output_variance": "true"}),
        ("vxl_convert_image", {"format": "byte", "scale_factor": "0.5"}),
    ]:
        algo = ImageFilter.create(name)
        cfg = algo.get_configuration()
        for key, value in values.items():
            cfg.set_value(key, value)
        algo.set_configuration(cfg)
        filters.append(algo)
    return filters


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_motion_history_and_reset(context, channels):
    from viame.types import Image, ImageContainer
    rng = np.random.default_rng(10)
    filters = cpu_filters()
    for shape in [(11, 17, channels), (3, 7, channels)]:
        output = None
        for step in range(70):
            source = rng.integers(0, 256, shape, dtype=np.uint8)
            if step % 10 == 0:
                source.fill(255 if step % 20 else 0)
            grey = filters[0].filter(ImageContainer(Image(source)))
            short = filters[3].filter(filters[1].filter(grey)).asarray().reshape(shape[:2])
            long = filters[3].filter(filters[2].filter(grey)).asarray().reshape(shape[:2])
            expected = np.stack([short, grey.asarray().reshape(shape[:2]), long], axis=2)
            output = context.gfit_motion(context.upload(source), out=output)
            np.testing.assert_array_equal(context.download(output), expected)
    context.reset_gfit_motion()
    actual = context.download(context.gfit_motion(context.upload(source)))
    assert not actual[:, :, (0, 2)].any()


def test_torch_owns_borrowed_storage(context):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA unavailable")
    source = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
    image = context.upload(source)
    tensor = torch.as_tensor(image, device="cuda:0")
    del image, context
    gc.collect()
    np.testing.assert_array_equal(tensor.cpu().numpy(), source)


def test_invalid_inputs(context):
    image = context.upload(np.ones((3, 7, 3), np.float32))
    with pytest.raises(ValueError):
        context.gfit_motion(image)
    with pytest.raises(ValueError):
        context.resize_letterbox(image, 10, 10)
    image = context.upload(np.ones((3, 7, 3), np.uint8))
    with pytest.raises(ValueError):
        context.resize_letterbox(image, 0, 10)
    with pytest.raises(ValueError):
        context.gfit_motion(image, out=context.upload(np.ones((3, 7), np.uint8)))
