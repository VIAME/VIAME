import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_disparity", Path(__file__).resolve().parents[2] / "tools" / "disparity.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import numpy as np


def test_empty_disparity_is_valid_empty_result():
    with patch.object(m, 'disparity', return_value=np.full((100, 100), -1, dtype=np.float32)):
        result = m.multipass_disparity(np.zeros((100, 100)), np.zeros((100, 100)))
    assert np.all(result == -1)


def test_constant_disparity_second_pass_has_positive_range():
    image = np.zeros((100, 100), dtype=np.uint8)
    result = m.disparity(image, image, (0, 0))
    assert result.shape == image.shape


def test_mismatched_pair_rejected():
    with pytest.raises(ValueError, match='same dimensions'):
        m.scaled_disparity(np.zeros((100, 100)), np.zeros((100, 101)))
