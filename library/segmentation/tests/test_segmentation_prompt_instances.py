"""Point clicks build a buffer of per-object mask instances."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'segmentation_utils', ROOT / 'library/segmentation/segmentation_utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

SHAPE = (100, 200)


def blob(x0, y0, x1, y1):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


HEAD = blob(10, 10, 40, 50)
FISH = blob(10, 10, 90, 50)
OTHER = blob(120, 20, 180, 80)


class Model:
    """Each positive yields its object; negatives cut a column out of it."""

    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def __call__(self, positives, negatives):
        self.calls.append((list(positives), list(negatives)))
        mask = np.zeros(SHAPE, dtype=bool)
        for point in positives:
            for region in self.objects:
                if utils.point_in_mask(region, point):
                    mask |= region
        for x, _ in negatives:
            mask[:, int(x):int(x) + 3] = False
        return mask


def buffer(model):
    return utils.PromptInstances(SHAPE, model)


def test_first_positive_creates_one_instance():
    b = buffer(Model([FISH, OTHER]))
    b.sync([([30, 30], 1)])
    assert len(b.instances) == 1 and np.array_equal(b.mask(), FISH)


def test_negatives_alone_create_nothing():
    model = Model([FISH])
    b = buffer(model)
    b.sync([([30, 30], 0), ([60, 30], 0)])
    assert b.instances == [] and not b.mask().any() and model.calls == []


def test_positive_on_another_object_becomes_its_own_instance():
    model = Model([FISH, OTHER])
    b = buffer(model)
    b.sync([([30, 30], 1), ([150, 50], 1)])
    assert len(b.instances) == 2
    assert np.array_equal(b.mask(), FISH | OTHER)
    assert ([[150, 50]], []) in model.calls


def test_positive_inside_an_instance_refines_it():
    b = buffer(Model([FISH, OTHER]))
    b.sync([([30, 30], 1), ([70, 30], 1)])
    assert len(b.instances) == 1 and len(b.instances[0]["positives"]) == 2


def test_positive_outside_joins_when_the_joint_mask_is_one_region():
    def model(positives, negatives):
        return FISH if len(positives) > 1 else (HEAD if positives[0][0] < 40 else blob(60, 10, 90, 50))

    b = buffer(model)
    b.sync([([30, 30], 1), ([70, 30], 1)])
    assert len(b.instances) == 1 and np.array_equal(b.mask(), FISH)


def test_join_is_refused_when_the_joint_mask_swallows_the_frame():
    def model(positives, negatives):
        if len(positives) > 1:
            return np.ones(SHAPE, dtype=bool)
        return HEAD if positives[0][0] < 40 else blob(60, 60, 90, 90)

    b = buffer(model)
    b.sync([([30, 30], 1), ([70, 70], 1)])
    assert len(b.instances) == 2


def test_negative_is_applied_to_every_instance():
    b = buffer(Model([FISH, OTHER]))
    b.sync([([30, 30], 1), ([150, 50], 1), ([60, 30], 0), ([165, 50], 0)])
    assert len(b.instances) == 2
    for negative in ([60, 30], [165, 50]):
        assert not utils.point_in_mask(b.mask(), negative)
    assert np.array_equal(b.instances[0]["mask"], blob(10, 10, 60, 50))
    assert np.array_equal(b.instances[1]["mask"], blob(120, 20, 165, 80))


def test_negative_the_model_keeps_covering_is_carved_out():
    b = buffer(lambda positives, negatives: FISH)
    b.sync([([30, 30], 1), ([60, 30], 0)])
    assert utils.point_in_mask(b.mask(), [30, 30]) and not utils.point_in_mask(b.mask(), [60, 30])


def test_positive_the_model_will_not_segment_adds_nothing():
    b = buffer(Model([FISH]))
    b.sync([([30, 30], 1), ([150, 50], 1)])
    assert len(b.instances) == 1 and np.array_equal(b.mask(), FISH)


def test_extending_the_prompts_only_runs_the_new_click():
    model = Model([FISH, OTHER])
    b = buffer(model)
    b.sync([([30, 30], 1)])
    before = len(model.calls)
    b.sync([([30, 30], 1), ([70, 30], 1)])
    assert len(model.calls) == before + 1


def test_changed_history_replays_from_scratch():
    b = buffer(Model([FISH, OTHER]))
    b.sync([([30, 30], 1), ([150, 50], 1)])
    b.sync([([150, 50], 1)])
    assert len(b.instances) == 1 and np.array_equal(b.mask(), OTHER)


def test_point_budget_grows_only_for_complex_rings():
    import math
    square = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]
    assert utils.simplify_polygon_within_error(square, 25, 100) == square
    star = [[(100 if i % 2 else 40) * math.cos(math.pi * i / 40) + 100,
             (100 if i % 2 else 40) * math.sin(math.pi * i / 40) + 100] for i in range(80)]
    star.append(star[0])
    capped = utils.simplify_polygon_within_error(star, 25, 25)
    grown = utils.simplify_polygon_within_error(star, 25, 100)
    assert len(capped) <= 25 < len(grown) <= 100


def test_small_component_survives_the_area_filter_when_it_holds_a_prompt():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 1
    mask[2:5, 2:5] = 1
    polygons, _ = utils.mask_to_polygons(mask)
    assert len(polygons) == 1
    polygons, _ = utils.mask_to_polygons(mask, keep_points=[[3, 3]])
    assert len(polygons) == 2
