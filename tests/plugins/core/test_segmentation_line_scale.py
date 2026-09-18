"""Keeping a line-prompted mask in scale with its head/tail line."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'segmentation_utils', ROOT / 'plugins/core/segmentation_utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

LINE = [[100, 100], [200, 100]]


def test_oversized_compares_the_longer_box_side_to_the_line():
    assert not utils.mask_oversized_for_line([95, 80, 205, 120], LINE)
    assert not utils.mask_oversized_for_line([0, 0, 240, 240], LINE)
    assert utils.mask_oversized_for_line([0, 0, 100, 260], LINE)
    assert utils.mask_oversized_for_line([0, 0, 400, 300], LINE)
    assert not utils.mask_oversized_for_line([0, 0, 400, 300], [[5, 5]])
    assert utils.mask_undersized_for_line([324, 591, 328, 594], [[300, 600], [340, 600]])
    assert not utils.mask_undersized_for_line([95, 80, 205, 120], LINE)


def test_background_points_ring_the_line_inside_the_image():
    ring = utils.line_background_points(LINE, (1000, 1000))
    assert [65.0, 100.0] in ring and [235.0, 100.0] in ring
    assert [150.0, 160.0] in ring and [150.0, 40.0] in ring
    assert len(ring) == 8
    assert all(y >= 0 for _, y in utils.line_background_points(LINE, (1000, 150)))
    assert len(utils.line_background_points(LINE, (1000, 150))) == 5


def test_clip_keeps_only_the_band_around_the_line():
    mask = np.ones((300, 400), dtype=np.uint8)
    clipped, (x0, y0) = utils.clip_mask_to_line(mask, (0, 0), LINE)
    bounds = [x0, y0, x0 + clipped.shape[1] - 1, y0 + clipped.shape[0] - 1]
    assert not utils.mask_oversized_for_line(bounds, LINE)
    assert y0 >= 65 and y0 + clipped.shape[0] <= 136
    assert clipped[clipped.shape[0] // 2, clipped.shape[1] // 2] == 1


def test_clip_honours_the_crop_offset_and_reports_empty():
    mask = np.ones((20, 20), dtype=np.uint8)
    clipped, offset = utils.clip_mask_to_line(mask, (140, 95), LINE)
    assert offset == (140, 95) and clipped.shape == (20, 20)
    assert utils.clip_mask_to_line(mask, (600, 600), LINE) is None


def test_clip_keeps_the_largest_piece():
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[95:106, 100:201] = 1
    mask[72:75, 120:123] = 1
    clipped, (x0, y0) = utils.clip_mask_to_line(mask, (0, 0), LINE)
    assert (x0, y0) == (100, 95) and clipped.shape == (11, 101)
