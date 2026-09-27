"""Optional CUDA parity and ownership tests; GPU cases skip on CPU-only hosts."""
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from viame import image_kernels as cpu
from viame.image_kernels import cuda


def test_optional_import_without_backend(monkeypatch):
    # Execute a fresh wrapper while preventing any extension import.
    spec = importlib.util.spec_from_file_location("optional_cuda", Path(cuda.__file__))
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    def missing(*args):
        raise ModuleNotFoundError("test: CUDA extension absent")
    monkeypatch.setattr(wrapper, "import_module", missing)
    assert not wrapper.available()
    assert "VIAME_ENABLE_CUDA_KERNELS" in wrapper.availability_error()
    with pytest.raises(RuntimeError, match="CUDA extension absent"):
        wrapper.Context()
    with pytest.raises(AttributeError):
        wrapper.unknown_attribute


@pytest.fixture
def context():
    if not cuda.available():
        pytest.skip(cuda.availability_error())
    return cuda.Context()


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1), (7, 13), (31, 67, 3), (19, 23, 4)])
@pytest.mark.parametrize("size,sigma", [(1, 0), (3, 0), (5, 1.7), (9, 0), (21, 3), (51, 0)])
def test_gaussian(context, shape, size, sigma):
    source = np.random.default_rng(2).random(shape, dtype=np.float32)
    source = source[::-1, ::-1]  # Strided host input.
    device = context.upload(source)
    expected = cpu.gaussian_blur(source, size, sigma)
    result = context.gaussian_blur(device, size, sigma)
    np.testing.assert_allclose(context.download(result), expected, rtol=3e-6, atol=3e-7)
    # Reusing a result allocation and in-place execution both remain valid.
    context.gaussian_blur(device, size, sigma, out=result)
    context.gaussian_blur(device, size, sigma, out=device)
    np.testing.assert_array_equal(context.download(device), context.download(result))


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (7, 11), (13, 17, 2), (19, 23, 3)])
@pytest.mark.parametrize("strength,patch,window", [(0, 3, 5), (3, 7, 21), (11, 2, 6), (30, 1, 3)])
def test_nlm(context, shape, strength, patch, window):
    source = np.random.default_rng(3).integers(0, 80, shape, dtype=np.uint8)
    device = context.upload(source)
    expected = cpu.denoise(source, strength, patch, window)
    result = context.denoise_non_local_means(device, strength, patch, window)
    np.testing.assert_array_equal(context.download(result), expected)
    context.denoise_non_local_means(device, strength, patch, window, out=device)
    np.testing.assert_array_equal(context.download(device), expected)


def test_lifetime_reuse_and_concurrent_context(context):
    rng = np.random.default_rng(4)
    for shape in [(50, 60, 3), (2, 4), (77, 31, 2)]:
        source = rng.random(shape, dtype=np.float32)
        image = context.upload(source)
        context.upload(source * 2, out=image)
        expected = cpu.gaussian_blur(source * 2, 5, 1)
        def process(_):
            return context.download(context.gaussian_blur(image, 5, 1))
        with ThreadPoolExecutor(4) as pool:
            for actual in pool.map(process, range(8)):
                np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
    del context
    gc.collect()
    another = cuda.Context()
    np.testing.assert_array_equal(another.download(image), source * 2)


def test_two_contexts_chain_and_distinct_outputs(context):
    source = np.random.default_rng(5).random((41, 53), dtype=np.float32)
    image = context.upload(source)
    first = context.gaussian_blur(image, 5)
    snapshot = context.download(first)
    another = cuda.Context()
    second = another.gaussian_blur(first, 9)
    context.gaussian_blur(image, 21)  # Scratch growth must not change earlier outputs.
    np.testing.assert_array_equal(context.download(first), snapshot)
    expected = cpu.gaussian_blur(cpu.gaussian_blur(source, 5), 9)
    np.testing.assert_allclose(context.download(second), expected, rtol=3e-6, atol=3e-7)


def test_validation(context):
    f = context.upload(np.ones((5, 7), np.float32))
    b = context.upload(np.ones((5, 7), np.uint8))
    for size in (0, 2, 257):
        with pytest.raises(ValueError):
            context.gaussian_blur(f, size)
    for sigma in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            context.gaussian_blur(f, 3, sigma)
    with pytest.raises(ValueError):
        context.gaussian_blur(b, 3)
    with pytest.raises(ValueError):
        context.gaussian_blur(f, 3, out=b)
    for params in [(1, 0, 3), (1, 3, 64), (-1, 3, 3), (float("nan"), 3, 3)]:
        with pytest.raises(ValueError):
            context.denoise_non_local_means(b, *params)
    with pytest.raises(ValueError):
        context.denoise_non_local_means(f, 3)
    with pytest.raises(TypeError):
        context.upload(np.zeros((4, 4), np.float64))
    with pytest.raises(ValueError):
        context.upload(np.zeros((0, 4), np.uint8))
    with pytest.raises(ValueError):
        context.upload(np.zeros((4, 4), np.uint8), out=b)


def test_nlm_cache_changes(context):
    rng = np.random.default_rng(6)
    for shape, strength, patch, window in [((11, 17, 3), 9, 3, 7),
                                         ((7, 13), 0, 2, 4),
                                         ((11, 17, 3), 9, 3, 7)]:
        source = rng.integers(0, 80, shape, dtype=np.uint8)
        image = context.upload(source)
        result = context.denoise_non_local_means(image, strength, patch, window)
        np.testing.assert_array_equal(context.download(result),
                                      cpu.denoise(source, strength, patch, window))
