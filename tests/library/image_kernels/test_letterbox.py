"""Frozen kwimage/OpenCV letterbox pixels, independent of cv2 at test time."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from viame.image_kernels import resize_letterbox

CASES = json.loads(Path(__file__).with_name('letterbox_reference.json').read_text())

@pytest.mark.parametrize('case', CASES)
def test_letterbox_pixels(case):
    image = np.random.default_rng(case['seed']).integers(0,256,case['shape'],dtype=np.uint8)
    result = resize_letterbox(image, *case['size'])
    assert hashlib.sha256(result.tobytes()).hexdigest() == case['sha256']
    assert list(result.shape) == case['result_shape']

@pytest.mark.parametrize('size', [(0,5),(-1,5),(5,0)])
def test_letterbox_rejects_empty_size(size):
    with pytest.raises(ValueError):
        resize_letterbox(np.zeros((7,9,3),np.uint8),*size)


@pytest.mark.parametrize('dtype', ['uint16', 'float32'])
@pytest.mark.parametrize('size', [(5, 7), (17, 19)])
def test_letterbox_preserves_high_precision_images(dtype, size):
    key = dtype + '_' + str(size[0])
    with np.load(Path(__file__).with_name('letterbox_reference_types.npz')) as data:
        expected = data[key + '_output']
        actual = resize_letterbox(data[key + '_input'], *size)
    assert actual.dtype == expected.dtype
    if dtype == 'uint16':
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-4)
