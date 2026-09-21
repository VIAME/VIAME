"""A point-prompted mask always covers every positive click and no negative one."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'segmentation_utils', ROOT / 'plugins/core/segmentation_utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def blob(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


SHAPE = (100, 200)
LEFT = blob(SHAPE, 10, 10, 50, 50)
RIGHT = blob(SHAPE, 120, 20, 180, 80)


def test_untouched_when_prompts_already_agree():
    mask, changed = utils.reconcile_mask_with_prompts(LEFT, [[30, 30]], [[150, 50]], lambda p, n: None)
    assert not changed and np.array_equal(mask, LEFT)


def test_missed_positive_adds_the_component_predicted_from_it_alone():
    calls = []

    def predict(pos, neg):
        calls.append((pos, neg))
        return RIGHT | blob(SHAPE, 0, 90, 5, 95)

    mask, changed = utils.reconcile_mask_with_prompts(LEFT, [[30, 30], [150, 50]], [[190, 5]], predict)
    assert changed and calls == [([[150, 50]], [[190, 5]])]
    assert np.array_equal(mask, LEFT | RIGHT)
    assert utils.mask_components(mask)[0] == 2


def test_missed_positive_gets_a_disk_when_the_model_will_not_cover_it():
    mask, changed = utils.reconcile_mask_with_prompts(LEFT, [[150, 50]], [], lambda p, n: None)
    assert changed and utils.point_in_mask(mask, [150, 50]) and utils.point_in_mask(mask, [30, 30])
    assert mask.sum() < LEFT.sum() + 200


def test_negative_drops_a_component_that_holds_no_positive():
    mask, changed = utils.reconcile_mask_with_prompts(LEFT | RIGHT, [[30, 30]], [[150, 50]], lambda p, n: None)
    assert changed and np.array_equal(mask, LEFT)


def test_negative_inside_a_positive_component_replaces_it_with_a_re_prediction():
    calls = []

    def predict(pos, neg):
        calls.append((pos, neg))
        return blob(SHAPE, 120, 20, 150, 80)

    mask, changed = utils.reconcile_mask_with_prompts(
        LEFT | RIGHT, [[30, 30], [130, 50]], [[170, 50]], predict)
    assert changed and calls == [([[130, 50]], [[170, 50]])]
    assert np.array_equal(mask, LEFT | blob(SHAPE, 120, 20, 150, 80))


def test_negative_is_carved_out_when_the_re_prediction_still_covers_it():
    mask, changed = utils.reconcile_mask_with_prompts(
        RIGHT, [[130, 50]], [[170, 50]], lambda p, n: RIGHT)
    assert changed and not utils.point_in_mask(mask, [170, 50]) and utils.point_in_mask(mask, [130, 50])
    assert mask.sum() > RIGHT.sum() - 200


def test_small_component_survives_the_area_filter_when_it_holds_a_prompt():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 1
    mask[2:5, 2:5] = 1
    polygons, _ = utils.mask_to_polygons(mask)
    assert len(polygons) == 1
    polygons, _ = utils.mask_to_polygons(mask, keep_points=[[3, 3]])
    assert len(polygons) == 2
