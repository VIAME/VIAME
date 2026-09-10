import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_mosaic", Path(__file__).resolve().parents[2] / "tools" / "mosaic.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import numpy as np


def test_canvas_limit_before_allocation():
    hom = np.eye(3)[None]
    hom[0, 0, 0] = 100000
    with pytest.raises(ValueError, match='max-pixels'):
        m.paste_many(hom, [], np.zeros((100, 100, 3), dtype=np.uint8), max_pixels=1000)


def test_one_frame(tmp_path):
    hom, images = tmp_path / 'h.txt', tmp_path / 'i.txt'
    hom.write_text('1 0 0 0 1 0 0 0 1 0 0\n')
    images.write_text('image.png\n')
    with patch.object(m.skio, 'imread', return_value=np.zeros((4, 4, 3), dtype=np.uint8)), patch.object(m.skio, 'imsave') as save:
        m.main_multi('out.png', [(str(hom), str(images))], frames=1)
    assert save.call_args.args[1].shape == (4, 4, 3)


def test_infinite_homography_rejected():
    h = np.eye(3)[None]
    h[0, 2, 0] = -0.5
    with pytest.raises(ValueError, match='infinity'):
        m.get_extreme_coordinates(h, (4, 4))
