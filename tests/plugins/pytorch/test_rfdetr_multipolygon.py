"""Regression coverage for disconnected pieces of one training instance.

Exercises native CSV IO and both windowed trainers, including cache reuse,
then the production RF-DETR COCO exporter without loading a model or GPU.
"""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ubelt as ub
from PIL import Image

pytest.importorskip('kwiver.vital.types')
from kwiver.vital import types as kt
from kwiver.vital.algo import DetectedObjectSetInput, DetectedObjectSetOutput, TrainDetector
from kwiver.vital.plugin_management import plugin_manager_instance

ROOT = Path(__file__).resolve().parents[3]
POLYGONS = [[110., 10., 130., 10., 130., 30., 110., 30.],
            [160., 60., 180., 60., 180., 80., 160., 80.]]


class MultipolygonCaptureTrainer(TrainDetector):
    captured = None

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_default_config(cls, config):
        pass

    def set_configuration(self, config):
        pass

    def check_configuration(self, config):
        return True

    def add_data_from_disk(self, labels, train_files, train_dets, test_files, test_dets):
        type(self).captured = (train_files, train_dets)


@pytest.fixture(scope='module', autouse=True)
def plugins():
    plugin_manager_instance().load_all_plugins()


def read_csv(path):
    reader = DetectedObjectSetInput.create('viame_csv')
    reader.open(str(path))
    dets, _ = reader.read_set()
    reader.close()
    return dets


def input_detection(tmp_path):
    path = tmp_path / 'input.csv'
    fields = [f'(poly) {" ".join(map(str, poly))}' for poly in POLYGONS]
    path.write_text('1,image.png,0,100,0,199,99,1,0,fish,1,' + ','.join(fields) + '\n')
    return read_csv(path)


def load_function(name, namespace):
    path = ROOT / 'plugins/pytorch/rf_detr_trainer.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_csv_roundtrip_and_legacy_api(tmp_path):
    dets = input_detection(tmp_path)
    det = list(dets)[0]
    assert det.get_flattened_polygons() == POLYGONS
    assert det.get_flattened_polygon() == POLYGONS[0]
    clone = det.clone()
    assert clone.get_flattened_polygons() == POLYGONS
    clone.set_flattened_polygon(POLYGONS[1])
    assert clone.get_flattened_polygons() == [POLYGONS[1]]
    assert det.get_flattened_polygons() == POLYGONS
    clone.set_flattened_polygons([])
    assert clone.get_flattened_polygon() == []

    writer = DetectedObjectSetOutput.create('viame_csv')
    output = tmp_path / 'output.csv'
    writer.open(str(output))
    writer.write_set(dets, 'image.png')
    writer.close()
    assert list(read_csv(output))[0].get_flattened_polygons() == POLYGONS


@pytest.mark.parametrize('implementation', ['windowed', 'ocv_windowed'])
@pytest.mark.parametrize('scale', [1, 2])
def test_chips_cache_and_training_export(tmp_path, implementation, scale):
    image = tmp_path / 'image.png'
    Image.new('RGB', (200, 100)).save(image)
    train_dir = tmp_path / 'training'
    labels = kt.CategoryHierarchy()
    labels.add_class('fish')
    trainer = TrainDetector.create(implementation)
    cfg = trainer.get_configuration()
    for key, value in {
        'train_directory': str(train_dir), 'reuse_cache': 'true',
        'trainer:type': 'MultipolygonCaptureTrainer', 'image_reader:type': 'ocv',
        'mode': 'chip', 'chip_width': str(100 * scale), 'chip_height': str(100 * scale),
        'chip_step_width': str(100 * scale), 'chip_step_height': str(100 * scale),
        'scale': str(scale), 'chip_threads': '1', 'chips_w_gt_only': 'true',
        'random_validation': '0', 'small_action': 'none',
    }.items():
        cfg.set_value(key, value)
    trainer.set_configuration(cfg)
    expected = (np.array(POLYGONS).reshape(2, 4, 2) - [100., 0.]) * scale

    def run():
        trainer.add_data_from_disk(labels, [str(image)], [input_detection(tmp_path)], [], [])
        files, sets = MultipolygonCaptureTrainer.captured
        detections = [d for ds in sets for d in ds]
        assert len(detections) == 1
        np.testing.assert_allclose(detections[0].get_flattened_polygons(), expected.reshape(2, 8))
        return files, sets

    files, sets = run()
    manifests = list(train_dir.rglob('*.manifest'))
    assert manifests
    contents = {p: p.read_bytes() for p in manifests}
    mtimes = {p: p.stat().st_mtime_ns for p in manifests}
    run()
    assert {p: p.stat().st_mtime_ns for p in manifests} == mtimes
    # Unversioned caches from the old reader/exporter must be regenerated.
    for p in manifests:
        p.write_bytes(contents[p].split(b'\n', 1)[1])
    files, sets = run()
    assert {p: p.read_bytes() for p in manifests} == contents

    ns = {'ub': ub, 'os': os, 'json': json, 'parse_bool': bool}
    load_function('polygon_area', ns)
    export = load_function('_prepare_roboflow_dataset', ns)
    state = SimpleNamespace(
        _train_directory=str(train_dir), _class_names=['fish'],
        _segmentation=True, _keypoints=False, _verify_images=False,
        _resolve_scan_threads=lambda: 1, _val_subsample=0,
        _train_image_files=files, _train_detections=sets,
        _test_image_files=files, _test_detections=sets,
    )
    dataset, _ = export(state)
    for split in ['train', 'valid', 'test']:
        document = json.loads((dataset / split / '_annotations.coco.json').read_text())
        assert len(document['annotations']) == 1
        ann = document['annotations'][0]
        np.testing.assert_allclose(ann['segmentation'], expected.reshape(2, 8))
        assert ann['area'] == 800. * scale ** 2
        # Exercise COCO's actual rasterization boundary, as RF-DETR does.
        from pycocotools import mask as coco_mask
        rles = coco_mask.frPyObjects(ann['segmentation'], 100 * scale, 100 * scale)
        mask = coco_mask.decode(rles).any(axis=2)
        assert mask[20 * scale, 20 * scale] and mask[70 * scale, 70 * scale]
        assert not mask[45 * scale, 45 * scale]
