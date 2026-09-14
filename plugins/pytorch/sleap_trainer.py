# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
"""KWIVER training adapter for SLEAP-NN keypoints on detection crops."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from kwiver.vital.algo import TrainDetector

from viame.pytorch.sleap_common import (
    as_rgb, crop_detection, detection_points, parse_keypoint_names,
    read_config, transform_points,
)
from viame.pytorch.utilities import register_vital_algorithm


class SleapTrainer(TrainDetector):
    DEFAULTS = dict(
        identifier='viame-sleap-keypoints', train_directory='deep_training',
        keypoint_names='head,tail', crop_width=256, crop_height=256, crop_padding=1.25,
        device='auto', batch_size=16, max_epochs=100, learning_rate=0.001,
        num_workers=4, seed=42, augmentation=True, filters=16,
        max_stride=16, output_stride=2, sigma=2.5, patience=15,
        steps_per_epoch=0, timeout=1209600, seed_model='',
    )

    def __init__(self):
        TrainDetector.__init__(self)
        self.options = dict(self.DEFAULTS)
        self.work_dir = None
        self.records = {'train': [], 'val': []}

    def get_configuration(self):
        cfg = super().get_configuration()
        for key, value in self.options.items():
            cfg.set_value(key, str(value))
        return cfg

    @staticmethod
    def validate(options):
        parse_keypoint_names(options['keypoint_names'])
        for key in ('crop_width', 'crop_height', 'batch_size', 'max_epochs', 'filters', 'max_stride', 'output_stride', 'patience', 'timeout'):
            if options[key] <= 0:
                raise ValueError('%s must be positive' % key)
        stride = options['max_stride']
        output = options['output_stride']
        if stride < 2 or stride & (stride - 1) or output & (output - 1) or output > stride:
            raise ValueError('max_stride and output_stride must be powers of two, with output_stride <= max_stride')
        if options['crop_height'] % stride or options['crop_width'] % stride:
            raise ValueError('Crop dimensions must be divisible by max_stride')
        if options['num_workers'] < 0 or options['steps_per_epoch'] < 0:
            raise ValueError('num_workers and steps_per_epoch must be nonnegative')
        if not np.isfinite(options['crop_padding']) or options['crop_padding'] < 1:
            raise ValueError('crop_padding must be >= 1')
        for key in ('learning_rate', 'sigma'):
            if not np.isfinite(options[key]) or options[key] <= 0:
                raise ValueError('%s must be positive' % key)
        if options['seed_model'] and not Path(options['seed_model']).is_file():
            raise FileNotFoundError(options['seed_model'])

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        options = read_config(self.DEFAULTS, cfg)
        self.validate(options)
        self.options = options
        self.work_dir = None
        self.records = {'train': [], 'val': []}
        return True

    def check_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        try:
            self.validate(read_config(self.DEFAULTS, cfg))
            return True
        except (ValueError, KeyError, OSError):
            return False

    def _workspace(self):
        if self.work_dir is None:
            root = Path(self.options['train_directory']).resolve()
            root.mkdir(parents=True, exist_ok=True)
            self.work_dir = Path(tempfile.mkdtemp(prefix='sleap-', dir=root))
        return self.work_dir

    def add_data_from_disk(self, categories, train_files, train_dets, test_files, test_dets):
        import cv2
        self._workspace()
        names = parse_keypoint_names(self.options['keypoint_names'])
        size = (self.options['crop_height'], self.options['crop_width'])
        for split, files, truth in (('train', train_files, train_dets), ('val', test_files, test_dets)):
            if len(files) != len(truth):
                raise ValueError('%s image and groundtruth counts differ' % split)
            directory = self.work_dir / split
            directory.mkdir(exist_ok=True)
            for filename, detections in zip(files, truth):
                image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError('Unable to read training image: %s' % filename)
                image = as_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                for det in detections:
                    if categories is not None and det.type is not None:
                        if not categories.has_class_name(det.type.get_most_likely_class()):
                            continue
                    points = detection_points(det, names)
                    # Points outside the source image are not training targets.
                    invalid = ((points[:, 0] < 0) | (points[:, 0] >= image.shape[1]) |
                               (points[:, 1] < 0) | (points[:, 1] >= image.shape[0]))
                    points[invalid] = np.nan
                    box = det.bounding_box
                    cropped = crop_detection(image, [box.min_x(), box.min_y(), box.max_x(), box.max_y()],
                                             size, self.options['crop_padding'])
                    if cropped is None:
                        continue
                    crop, affine = cropped
                    points = transform_points(points, affine)
                    visible = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) &
                               (points[:, 0] < size[1]) & (points[:, 1] >= 0) & (points[:, 1] < size[0]))
                    if not visible.any():
                        continue
                    points[~visible] = np.nan
                    path = directory / ('%08d.png' % len(self.records[split]))
                    if not cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                        raise OSError('Unable to write crop: %s' % path)
                    self.records[split].append(dict(
                        image=str(path), source_image=str(Path(filename).resolve()),
                        points=[[float(x), float(y)] if keep else [None, None]
                                for (x, y), keep in zip(points, visible)],
                        box_diagonal=float(np.hypot(box.width(), box.height()) * affine[0, 0]),
                    ))

    def update_model(self):
        for split in ('train', 'val'):
            if not self.records[split]:
                raise ValueError('SLEAP needs %s crops with visible keypoint_names=%s; '
                                 'check the frame split and annotations' % (split, self.options['keypoint_names']))
        # Keep validation frames distinct even when one frame yields many crops.
        train_sources = {r['source_image'] for r in self.records['train']}
        if train_sources.intersection(r['source_image'] for r in self.records['val']):
            raise ValueError('SLEAP training and validation contain the same source image')
        request = dict(options=self.options, records=self.records,
                       output_dir=str(self.work_dir / 'model'))
        path = self.work_dir / 'request.json'
        path.write_text(json.dumps(request, indent=2, allow_nan=False))
        # A fresh subprocess keeps Lightning/data-loader workers outside the
        # embedded KWIVER interpreter; failures and timeout propagate to viame train.
        subprocess.run([sys.executable, '-m', 'viame.pytorch.sleap_launcher', str(path)],
                       check=True, timeout=self.options['timeout'])
        model = self.work_dir / 'model' / 'trained_keypoints.pt'
        metrics = self.work_dir / 'model' / 'keypoint_metrics.json'
        if not model.is_file() or not metrics.is_file():
            raise RuntimeError('SLEAP training did not produce its model and validation report')
        return {'type': 'sleap', 'sleap:weight': 'trained_keypoints.pt',
                'trained_keypoints.pt': str(model), 'keypoint_metrics.json': str(metrics)}


def __vital_algorithm_register__():
    register_vital_algorithm(SleapTrainer, 'sleap', 'Train SLEAP-NN keypoints on detection crops')
