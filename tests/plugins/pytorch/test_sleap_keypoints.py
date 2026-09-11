"""Crop/refiner regression checks plus an optional real SLEAP-NN CPU training test.

The KWIVER boundary is stubbed so these tests run without an installed VIAME.
The integration test imports the actual pinned SLEAP-NN package when available.
"""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('cv2')
ROOT = Path(__file__).resolve().parents[3]


class Config(dict):
    def set_value(self, key, value):
        self[key] = value

    def get_value(self, key):
        return self[key]

    def merge_config(self, other):
        self.update(other)


class Algorithm:
    def get_configuration(self):
        return Config()


class DetectionSet(list):
    def add(self, item):
        self.append(item)


class Point:
    def __init__(self, xy=(0, 0)):
        self.value = list(xy)


class Box:
    def __init__(self, coords):
        self.coords = coords

    def min_x(self): return self.coords[0]
    def min_y(self): return self.coords[1]
    def max_x(self): return self.coords[2]
    def max_y(self): return self.coords[3]
    def width(self): return self.max_x() - self.min_x()
    def height(self): return self.max_y() - self.min_y()


class Detection:
    def __init__(self, box, points=None):
        self.bounding_box = Box(box)
        self.keypoints = {name: Point(xy) for name, xy in (points or {}).items()}
        self.type = None
        self.confidence = .87
        self.notes = ['preserve this']
        self.index = 42

    def clone(self): return copy.deepcopy(self)
    def clear_keypoints(self): self.keypoints.clear()
    def add_keypoint(self, name, point): self.keypoints[name] = point


@pytest.fixture
def modules(monkeypatch):
    for name, path in [('viame', ROOT / 'plugins'), ('viame.pytorch', ROOT / 'plugins/pytorch')]:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
    for name in ['kwiver', 'kwiver.vital', 'kwiver.vital.algo', 'kwiver.vital.types', 'viame.pytorch.utilities']:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    algo = sys.modules['kwiver.vital.algo']
    algo.TrainDetector = algo.RefineDetections = Algorithm
    types = sys.modules['kwiver.vital.types']
    types.DetectedObjectSet, types.Point2d = DetectionSet, Point
    sys.modules['viame.pytorch.utilities'].register_vital_algorithm = lambda *args: None
    for name in ['sleap_common', 'sleap_trainer', 'sleap_refiner', 'sleap_launcher']:
        monkeypatch.delitem(sys.modules, 'viame.pytorch.' + name, raising=False)
    loaded = SimpleNamespace(**{name: importlib.import_module('viame.pytorch.' + name)
                                for name in ['sleap_common', 'sleap_trainer', 'sleap_refiner', 'sleap_launcher']})
    yield loaded
    for name in ['sleap_common', 'sleap_trainer', 'sleap_refiner', 'sleap_launcher']:
        sys.modules.pop('viame.pytorch.' + name, None)


@pytest.mark.parametrize('box', [[10, 20, 90, 60], [-10, -5, 30, 15]])
def test_crop_affine_roundtrip_and_letterbox(modules, box):
    common = modules.sleap_common
    image = np.full((80, 100, 3), 100, np.uint8)
    crop, affine = common.crop_detection(image, box, (64, 128), 1.25)
    assert crop.shape == (64, 128, 3)
    points = np.array([[box[0], box[1]], [box[2], box[3]]])
    transformed = common.transform_points(points, affine)
    np.testing.assert_allclose(common.transform_points(transformed, affine, inverse=True), points)
    np.testing.assert_allclose(transformed.mean(axis=0), [64, 32])
    if box[0] < 0:
        assert (crop == 0).any()


@pytest.mark.parametrize('box', [[0, 0, 0, 1], [5, 0, 2, 4], [100, 0, 120, 20],
                                [-20, -20, -1, -1], [0, 0, float('nan'), 1]])
def test_invalid_boxes_skipped(modules, box):
    assert modules.sleap_common.crop_detection(np.zeros((50, 50, 3), np.uint8), box, (32, 32), 1.25) is None


@pytest.mark.parametrize('names', ['', 'head,', 'head,HEAD', []])
def test_invalid_names_rejected(modules, names):
    with pytest.raises(ValueError, match='keypoint_names'):
        modules.sleap_common.parse_keypoint_names(names)


def fake_predictor():
    def predict(crops):
        return (np.repeat([[[8., 16.], [24., 16.]]], len(crops), axis=0),
                np.repeat([[.9, .8]], len(crops), axis=0))
    return SimpleNamespace(names=['head', 'tail'], size=(32, 32), padding=1.0, predict=predict)


def test_refiner_batches_and_fills_missing_points_without_mutation(modules):
    refiner = modules.sleap_refiner.SleapRefiner()
    refiner.options['batch_size'] = 2
    refiner.predictor = fake_predictor()
    calls = []
    predict = refiner.predictor.predict
    refiner.predictor.predict = lambda crops: (calls.append(len(crops)) or predict(crops))
    detections = DetectionSet([Detection([10, 20, 42, 52], {'HEAD': [13, 24], 'fin': [20, 30]})
                               for _ in range(3)])
    output = refiner.refine(SimpleNamespace(asarray=lambda: np.zeros((80, 80, 3), np.uint8)), detections)
    assert calls == [2, 1]
    for before, after in zip(detections, output):
        assert before is not after
        assert set(before.keypoints) == {'HEAD', 'fin'}
        assert after.keypoints['HEAD'].value == [13, 24]
        assert after.keypoints['tail'].value == [34, 36]
        assert after.keypoints['fin'].value == [20, 30]
        assert after.bounding_box.coords == before.bounding_box.coords
        assert (after.index, after.confidence, after.notes) == (before.index, before.confidence, before.notes)


def test_refiner_overwrite_keeps_unrelated_points_and_filters_low_scores(modules):
    refiner = modules.sleap_refiner.SleapRefiner()
    refiner.options['overwrite_existing'] = True
    refiner.predictor = fake_predictor()
    refiner.predictor.predict = lambda crops: (np.array([[[8., 16.], [24., 16.]]]), np.array([[.9, .1]]))
    det = Detection([10, 20, 42, 52], {'HEAD': [13, 24], 'tail': [25, 30], 'fin': [20, 30]})
    output = refiner.refine(SimpleNamespace(asarray=lambda: np.zeros((80, 80, 3), np.uint8)), [det])
    assert set(output[0].keypoints) == {'head', 'fin'}
    assert output[0].keypoints['head'].value == [18, 36]
    assert set(det.keypoints) == {'HEAD', 'tail', 'fin'}


def test_refiner_skips_complete_and_empty_sets(modules):
    refiner = modules.sleap_refiner.SleapRefiner()
    refiner.predictor = fake_predictor()
    refiner.predictor.predict = lambda crops: pytest.fail('should not run inference')
    image = SimpleNamespace(asarray=lambda: np.zeros((40, 40, 3), np.uint8))
    assert not refiner.refine(image, None)
    assert not refiner.refine(image, [])
    output = refiner.refine(image, [Detection([0, 0, 32, 32], {'head': [5, 5], 'TAIL': [20, 20]})])
    assert len(output) == 1


def make_trainer_data(modules, tmp_path):
    import cv2
    trainer = modules.sleap_trainer.SleapTrainer()
    cfg = trainer.get_configuration()
    cfg.update(train_directory=str(tmp_path), crop_height='32', crop_width='32', crop_padding='1',
               filters='4', max_stride='8', output_stride='1', device='cpu', max_epochs='1',
               batch_size='2', num_workers='0', steps_per_epoch='1', augmentation='false')
    trainer.set_configuration(cfg)
    files, sets = [], []
    for i in range(4):
        image = np.zeros((64, 64, 3), np.uint8)
        # BGR red checks that both export and inference agree on RGB channels.
        image[24:29, 16:21] = [0, 0, 255]
        path = tmp_path / ('frame%d.png' % i)
        cv2.imwrite(str(path), image)
        files.append(str(path))
        points = {'HEAD': [18, 26], 'tail': [34, 26]} if i != 0 else {'head': [18, 26]}
        sets.append([Detection([10, 10, 42, 42], points)])
    trainer.add_data_from_disk(None, files[:2], sets[:2], files[2:], sets[2:])
    return trainer


def test_training_export_missing_points_color_and_geometry(modules, tmp_path):
    import cv2
    trainer = make_trainer_data(modules, tmp_path)
    records = trainer.records['train']
    assert len(records) == 2
    assert records[0]['points'] == [[8.0, 16.0], [None, None]]
    assert records[1]['points'] == [[8.0, 16.0], [24.0, 16.0]]
    # Images are written in OpenCV's BGR order, read as RGB by SLEAP.
    assert cv2.imread(records[0]['image'])[16, 8].tolist() == [0, 0, 255]
    assert records[0]['box_diagonal'] == pytest.approx(np.hypot(32, 32))


def test_training_failure_propagates_and_success_maps_artifacts(modules, tmp_path, monkeypatch):
    trainer = make_trainer_data(modules, tmp_path)
    commands = []
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(modules.sleap_trainer.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        trainer.update_model()
    def succeed(command, **kwargs):
        commands.append((command, kwargs))
        request = json.loads(Path(command[-1]).read_text())
        directory = Path(request['output_dir'])
        directory.mkdir()
        (directory / 'trained_keypoints.pt').touch()
        (directory / 'keypoint_metrics.json').write_text('{}')
    monkeypatch.setattr(modules.sleap_trainer.subprocess, 'run', succeed)
    output = trainer.update_model()
    assert output['type'] == 'sleap'
    assert output['sleap:weight'] == 'trained_keypoints.pt'
    assert Path(output['trained_keypoints.pt']).is_file()
    assert commands[0][0][1:3] == ['-m', 'viame.pytorch.sleap_launcher']
    assert commands[0][1] == dict(check=True, timeout=1209600)


def test_training_rejects_empty_supervision_and_split_leakage(modules, tmp_path):
    trainer = modules.sleap_trainer.SleapTrainer()
    with pytest.raises(ValueError, match='visible'):
        trainer.update_model()
    trainer = make_trainer_data(modules, tmp_path)
    trainer.records['val'][0]['source_image'] = trainer.records['train'][0]['source_image']
    with pytest.raises(ValueError, match='same source image'):
        trainer.update_model()


@pytest.mark.parametrize('key,value', [('crop_height', 31), ('crop_padding', .5),
                                      ('batch_size', 0), ('max_stride', 6), ('output_stride', 32)])
def test_training_config_rejects_invalid_options(modules, key, value):
    options = dict(modules.sleap_trainer.SleapTrainer.DEFAULTS)
    options[key] = value
    with pytest.raises(ValueError):
        modules.sleap_trainer.SleapTrainer.validate(options)


def test_native_cpu_train_export_inference_and_finetune(modules, tmp_path):
    pytest.importorskip('sleap_nn')
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    trainer = make_trainer_data(modules, tmp_path)
    request = dict(options=trainer.options, records=trainer.records, output_dir=str(tmp_path / 'native'))
    model = modules.sleap_launcher.run_training(request)
    artifact = modules.sleap_common.load_artifact(model)
    assert artifact['keypoint_names'] == ['head', 'tail']
    assert artifact['crop_size'] == [32, 32]
    assert artifact['format'] == modules.sleap_common.MODEL_FORMAT
    metrics = json.loads(model.with_name('keypoint_metrics.json').read_text())
    assert metrics['keypoints']['head']['labeled'] == 2
    assert metrics['keypoints']['tail']['labeled'] == 2
    # Compare the export against the actual best native checkpoint's tensors.
    checkpoint = torch.load(tmp_path / 'native/training/best.ckpt', map_location='cpu', weights_only=False)
    for key, value in artifact['state_dict'].items():
        torch.testing.assert_close(value, checkpoint['state_dict']['model.' + key])
    refiner = modules.sleap_refiner.SleapRefiner()
    cfg = refiner.get_configuration()
    cfg.update(weight=str(model), device='cpu')
    refiner.set_configuration(cfg)
    result = refiner.refine(SimpleNamespace(asarray=lambda: np.zeros((64, 64, 3), np.uint8)),
                            [Detection([10, 10, 42, 42])])
    assert len(result) == 1
    # Reload a portable model as the initialization for a fresh native run.
    request['output_dir'] = str(tmp_path / 'finetuned')
    request['options'] = dict(trainer.options, seed_model=str(model))
    second = modules.sleap_launcher.run_training(request)
    assert modules.sleap_common.load_artifact(second)['keypoint_names'] == ['head', 'tail']


def test_partial_configuration_uses_defaults(modules, tmp_path):
    trainer = modules.sleap_trainer.SleapTrainer()
    assert trainer.check_configuration(Config(keypoint_names='head,tail,fin'))
    assert not trainer.check_configuration(Config(batch_size='0'))
    refiner = modules.sleap_refiner.SleapRefiner()
    weight = tmp_path / 'model.pt'
    weight.touch()
    assert refiner.check_configuration(Config(weight=str(weight)))
    assert not refiner.check_configuration(Config(weight=str(weight), batch_size='0'))


def test_evaluation_rejects_incompatible_crop_manifest(modules, tmp_path, monkeypatch):
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(options=dict(keypoint_names='head,tail', crop_height=32,
                                                   crop_width=32, crop_padding=1.25))))
    monkeypatch.setattr(modules.sleap_launcher, 'load_artifact', lambda path: dict(
        keypoint_names=['head', 'tail'], crop_size=[64, 64], crop_padding=1.25))
    monkeypatch.setattr(sys, 'argv', ['sleap_launcher', str(request), '--evaluate', 'model.pt'])
    with pytest.raises(ValueError, match='crop settings'):
        modules.sleap_launcher.main()
