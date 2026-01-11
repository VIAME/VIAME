# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Detectron2 trainer implementation for VIAME.

This module provides a KWIVER TrainDetector implementation for training
object detection models using Facebook's Detectron2 framework.

Supported architectures:
    - Faster R-CNN
    - Mask R-CNN
    - RetinaNet
    - Other Detectron2 models

Dependencies:
    - detectron2: Can be installed via geowatch_tpl or directly from Facebook
    - torch, numpy

Example usage:
    >>> from pytorch.detectron2_trainer import Detectron2Trainer
    >>> trainer = Detectron2Trainer()
    >>> cfg_in = dict(
    ...     identifier='my-detector',
    ...     base='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    ...     max_iter=1000,
    ... )
    >>> trainer.set_configuration(cfg_in)
    >>> trainer.add_data_from_disk(categories, train_files, train_dets, test_files, test_dets)
    >>> trainer.update_model()
"""

from __future__ import print_function

import os
import json

import scriptconfig as scfg
import ubelt as ub

from kwiver.vital.algo import (
    DetectedObjectSetOutput,
    ImageObjectDetector,
    TrainDetector
)

from .kwcoco_train_detector import KWCocoTrainDetector
from .kwcoco_train_detector import KWCocoTrainDetectorConfig

from .utilities import (
    vital_config_update,
    resolve_device_str,
    parse_bool,
    register_vital_algorithm,
    TrainingInterruptHandler,
)


class Detectron2TrainerConfig(KWCocoTrainDetectorConfig):
    """
    Configuration for :class:`Detectron2Trainer`.

    Inherits from KWCocoTrainDetectorConfig for common data handling options.
    """
    # Trainer identification
    identifier = "viame-detectron2-detector"
    train_directory = "deep_training"
    output_directory = "category_models"
    seed_model = ""

    # Temporary file paths
    tmp_training_file = "training_truth.json"
    tmp_validation_file = "validation_truth.json"

    # Model configuration
    base = scfg.Value(
        'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        help='Base model config from Detectron2 model zoo'
    )
    device = scfg.Value(
        'auto',
        help='Device to train on: auto, cpu, cuda, or cuda:N'
    )

    # Training hyperparameters
    max_iter = scfg.Value(
        10000,
        help='Maximum number of training iterations'
    )
    batch_size = scfg.Value(
        2,
        help='Number of images per batch (IMS_PER_BATCH)'
    )
    base_lr = scfg.Value(
        0.00025,
        help='Base learning rate'
    )
    warmup_iters = scfg.Value(
        1000,
        help='Number of warmup iterations'
    )
    warmup_factor = scfg.Value(
        0.001,
        help='Warmup factor (learning rate starts at base_lr * warmup_factor)'
    )
    lr_decay_steps = scfg.Value(
        '',
        help='Comma-separated iteration numbers to decay LR (e.g., "8000,9000")'
    )
    lr_decay_gamma = scfg.Value(
        0.1,
        help='Learning rate decay factor'
    )
    weight_decay = scfg.Value(
        0.0001,
        help='Weight decay for optimizer'
    )
    momentum = scfg.Value(
        0.9,
        help='SGD momentum'
    )

    # Model architecture options
    num_classes = scfg.Value(
        0,
        help='Number of classes (0 = auto-detect from data)'
    )
    anchor_sizes = scfg.Value(
        '',
        help='Comma-separated anchor sizes (e.g., "32,64,128,256,512")'
    )
    roi_batch_size = scfg.Value(
        512,
        help='Number of ROIs per image for training'
    )
    roi_positive_fraction = scfg.Value(
        0.25,
        help='Fraction of positive ROIs per batch'
    )

    # Data augmentation
    min_size_train = scfg.Value(
        640,
        help='Minimum image size during training'
    )
    max_size_train = scfg.Value(
        1333,
        help='Maximum image size during training'
    )
    random_flip = scfg.Value(
        'horizontal',
        help='Random flip augmentation: horizontal, vertical, none'
    )

    # Checkpointing and logging
    checkpoint_period = scfg.Value(
        1000,
        help='Save checkpoint every N iterations'
    )
    eval_period = scfg.Value(
        1000,
        help='Evaluate model every N iterations'
    )
    log_period = scfg.Value(
        20,
        help='Log metrics every N iterations'
    )

    # Advanced options
    fp16 = scfg.Value(
        False,
        help='Use mixed precision (FP16) training'
    )
    num_workers = scfg.Value(
        4,
        help='Number of data loader workers'
    )
    freeze_backbone_at = scfg.Value(
        0,
        help='Freeze backbone at specified stage (0 = no freezing)'
    )
    resume = scfg.Value(
        False,
        help='Resume training from last checkpoint'
    )

    # Pipeline template
    pipeline_template = ""

    # Categories (populated during training)
    categories = []

    def __post_init__(self):
        super().__post_init__()
        import kwutil
        self.max_iter = kwutil.Yaml.coerce(self.max_iter)
        self.batch_size = kwutil.Yaml.coerce(self.batch_size)
        self.base_lr = kwutil.Yaml.coerce(self.base_lr)
        self.warmup_iters = kwutil.Yaml.coerce(self.warmup_iters)
        self.warmup_factor = kwutil.Yaml.coerce(self.warmup_factor)
        self.lr_decay_gamma = kwutil.Yaml.coerce(self.lr_decay_gamma)
        self.weight_decay = kwutil.Yaml.coerce(self.weight_decay)
        self.momentum = kwutil.Yaml.coerce(self.momentum)
        self.num_classes = kwutil.Yaml.coerce(self.num_classes)
        self.roi_batch_size = kwutil.Yaml.coerce(self.roi_batch_size)
        self.roi_positive_fraction = kwutil.Yaml.coerce(self.roi_positive_fraction)
        self.min_size_train = kwutil.Yaml.coerce(self.min_size_train)
        self.max_size_train = kwutil.Yaml.coerce(self.max_size_train)
        self.checkpoint_period = kwutil.Yaml.coerce(self.checkpoint_period)
        self.eval_period = kwutil.Yaml.coerce(self.eval_period)
        self.log_period = kwutil.Yaml.coerce(self.log_period)
        self.num_workers = kwutil.Yaml.coerce(self.num_workers)
        self.freeze_backbone_at = kwutil.Yaml.coerce(self.freeze_backbone_at)


class Detectron2Trainer(KWCocoTrainDetector):
    """
    Implementation of TrainDetector for Detectron2 models.

    This trainer converts KWIVER data to COCO format and uses Detectron2's
    training infrastructure to train detection models.

    The training process:
    1. Receives data via add_data_from_disk() in KWIVER format
    2. Converts to COCO JSON format using the base class writers
    3. Registers the dataset with Detectron2
    4. Trains using Detectron2's DefaultTrainer
    5. Exports the final model for use with Detectron2Detector
    """

    def __init__(self):
        TrainDetector.__init__(self)
        self._config = Detectron2TrainerConfig()
        self._interrupted = False

    def get_configuration(self):
        """Get the algorithm configuration."""
        print('[Detectron2Trainer] get_configuration')
        cfg = super().get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        """Set the algorithm configuration."""
        print('[Detectron2Trainer] set_configuration')
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        # Set underscore-prefixed attributes for convenient access
        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._post_config_set()
        return True

    def _post_config_set(self):
        """Initialize directories and writers after configuration is set."""
        print('[Detectron2Trainer] _post_config_set')
        assert self._config['mode'] == "detector"

        # Set up directories
        if self._train_directory is not None:
            if not os.path.exists(self._train_directory):
                os.makedirs(self._train_directory, exist_ok=True)
            self._training_file = os.path.join(
                self._train_directory, self._tmp_training_file)
            self._validation_file = os.path.join(
                self._train_directory, self._tmp_validation_file)
            self._chip_directory = os.path.join(
                self._train_directory, "image_chips")
        else:
            self._training_file = self._tmp_training_file
            self._validation_file = self._tmp_validation_file

        if self._output_directory is not None:
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory, exist_ok=True)

        # Load KWIVER modules for writers
        from kwiver.vital.modules import load_known_modules
        load_known_modules()

        # Set up COCO format writers
        if not self._no_format:
            self._training_writer = DetectedObjectSetOutput.create("coco")
            self._validation_writer = DetectedObjectSetOutput.create("coco")

            writer_conf = self._training_writer.get_configuration()
            self._training_writer.set_configuration(writer_conf)

            writer_conf = self._validation_writer.get_configuration()
            self._validation_writer.set_configuration(writer_conf)

            self._training_writer.open(self._training_file)
            self._validation_writer.open(self._validation_file)

        # Set up chip directory for detection_refiner mode
        if self._mode == "detection_refiner" and not os.path.exists(self._chip_directory):
            os.makedirs(self._chip_directory, exist_ok=True)

        # Set up background detector if configured
        if self._detector_model:
            self._detector = ImageObjectDetector.create("yolo")
            detector_config = self._detector.get_configuration()
            detector_config.set_value("deployed", self._detector_model)
            if not self._detector.set_configuration(detector_config):
                print("[Detectron2Trainer] Unable to configure detector")
                return False

        # Fix chip extension
        if self._chip_extension and self._chip_extension[0] != '.':
            self._chip_extension = '.' + self._chip_extension

        # Set default chip dimensions
        if int(self._chip_height) <= 0:
            self._chip_height = self._chip_width
        if int(self._chip_width) <= 0:
            self._chip_width = self._chip_height

        # Initialize data storage
        self._training_data = []
        self._validation_data = []
        self._sample_count = 0

    def _ensure_format_writers(self):
        """Finalize the COCO format writers and fix the output files."""
        if not self._no_format:
            self._training_writer.complete()
            self._validation_writer.complete()

            # Use kwcoco to conform the output files
            import kwcoco
            paths_to_fix = [self._training_file, self._validation_file]
            for fpath in paths_to_fix:
                fpath = ub.Path(fpath)
                if fpath.exists():
                    dset = kwcoco.CocoDataset(fpath)
                    dset.conform()
                    dset.dump()

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("[Detectron2Trainer] A model identifier must be specified!")
            return False
        return True

    def _register_coco_dataset(self, name, json_file, image_root):
        """
        Register a COCO format dataset with Detectron2.

        Args:
            name: Dataset name for registration
            json_file: Path to COCO JSON annotations
            image_root: Root directory for images
        """
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import load_coco_json

        # Unregister if already exists
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)

        if name in MetadataCatalog:
            MetadataCatalog.remove(name)

        # Load and get category info from the JSON file
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        categories = coco_data.get('categories', [])
        thing_classes = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]

        # Register dataset
        DatasetCatalog.register(
            name,
            lambda jf=json_file, ir=image_root: load_coco_json(jf, ir)
        )

        # Set metadata
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            thing_classes=thing_classes,
        )

        return thing_classes

    def _setup_detectron2_config(self, num_classes, train_dataset_name, val_dataset_name):
        """
        Set up Detectron2 configuration for training.

        Args:
            num_classes: Number of object classes
            train_dataset_name: Name of registered training dataset
            val_dataset_name: Name of registered validation dataset

        Returns:
            detectron2.config.CfgNode: Configured training config
        """
        from detectron2.config import get_cfg
        from detectron2 import model_zoo

        cfg = get_cfg()

        # Load base configuration from model zoo
        base = self._base
        try:
            cfg.merge_from_file(model_zoo.get_config_file(base))
            # Try to get pretrained weights
            try:
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base)
            except Exception:
                pass
        except Exception:
            # Try as direct path
            if ub.Path(base).exists():
                cfg.merge_from_file(base)
            else:
                print(f"[Detectron2Trainer] Warning: Could not load base config: {base}")
                # Use default config
                default_base = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                cfg.merge_from_file(model_zoo.get_config_file(default_base))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(default_base)

        # Load seed model if provided
        if self._seed_model and ub.Path(self._seed_model).exists():
            cfg.MODEL.WEIGHTS = self._seed_model
            print(f"[Detectron2Trainer] Loading seed model from {self._seed_model}")

        # Set device
        device = resolve_device_str(self._device)
        cfg.MODEL.DEVICE = device

        # Set datasets
        cfg.DATASETS.TRAIN = (train_dataset_name,)
        cfg.DATASETS.TEST = (val_dataset_name,)

        # Set number of classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # Training hyperparameters
        cfg.SOLVER.MAX_ITER = int(self._max_iter)
        cfg.SOLVER.IMS_PER_BATCH = int(self._batch_size)
        cfg.SOLVER.BASE_LR = float(self._base_lr)
        cfg.SOLVER.WARMUP_ITERS = int(self._warmup_iters)
        cfg.SOLVER.WARMUP_FACTOR = float(self._warmup_factor)
        cfg.SOLVER.GAMMA = float(self._lr_decay_gamma)
        cfg.SOLVER.WEIGHT_DECAY = float(self._weight_decay)
        cfg.SOLVER.MOMENTUM = float(self._momentum)

        # Learning rate decay steps
        if self._lr_decay_steps:
            steps = [int(s.strip()) for s in self._lr_decay_steps.split(',') if s.strip()]
            cfg.SOLVER.STEPS = tuple(steps)
        else:
            # Default: decay at 80% and 90% of max_iter
            max_iter = int(self._max_iter)
            cfg.SOLVER.STEPS = (int(max_iter * 0.8), int(max_iter * 0.9))

        # ROI head configuration
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(self._roi_batch_size)
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = float(self._roi_positive_fraction)

        # Anchor sizes if provided
        if self._anchor_sizes:
            sizes = [[int(s.strip())] for s in self._anchor_sizes.split(',') if s.strip()]
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = sizes

        # Data augmentation
        cfg.INPUT.MIN_SIZE_TRAIN = (int(self._min_size_train),)
        cfg.INPUT.MAX_SIZE_TRAIN = int(self._max_size_train)
        cfg.INPUT.MIN_SIZE_TEST = int(self._min_size_train)
        cfg.INPUT.MAX_SIZE_TEST = int(self._max_size_train)

        if self._random_flip == 'none':
            cfg.INPUT.RANDOM_FLIP = "none"
        elif self._random_flip == 'vertical':
            cfg.INPUT.RANDOM_FLIP = "vertical"
        else:
            cfg.INPUT.RANDOM_FLIP = "horizontal"

        # Checkpointing and logging
        cfg.SOLVER.CHECKPOINT_PERIOD = int(self._checkpoint_period)
        cfg.TEST.EVAL_PERIOD = int(self._eval_period)

        # Data loader
        cfg.DATALOADER.NUM_WORKERS = int(self._num_workers)

        # Backbone freezing
        freeze_at = int(self._freeze_backbone_at)
        if freeze_at > 0:
            cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at

        # Mixed precision
        if parse_bool(self._fp16):
            cfg.SOLVER.AMP.ENABLED = True

        # Output directory
        output_dir = ub.Path(self._train_directory) / "detectron2_output"
        output_dir.ensuredir()
        cfg.OUTPUT_DIR = str(output_dir)

        return cfg

    def update_model(self):
        """
        Perform model training using Detectron2.

        This method:
        1. Finalizes data writers
        2. Registers datasets with Detectron2
        3. Sets up the training configuration
        4. Runs training with the DefaultTrainer
        5. Saves the final model
        """
        import torch

        self._ensure_format_writers()

        print("[Detectron2Trainer] Starting Detectron2 training")

        # Check that we have training data
        training_path = ub.Path(self._training_file)
        if not training_path.exists():
            print("[Detectron2Trainer] Error: No training data found")
            return

        # Import detectron2, preferring geowatch_tpl if available
        try:
            import geowatch_tpl
            detectron2 = geowatch_tpl.import_submodule('detectron2')  # NOQA
        except ImportError:
            import detectron2  # NOQA

        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator

        # Register datasets
        train_dataset_name = f"{self._identifier}_train"
        val_dataset_name = f"{self._identifier}_val"

        # Get image root (directory containing the annotation file)
        train_image_root = str(training_path.parent)

        thing_classes = self._register_coco_dataset(
            train_dataset_name,
            str(self._training_file),
            train_image_root
        )

        # Register validation dataset if it exists
        validation_path = ub.Path(self._validation_file)
        if validation_path.exists():
            val_image_root = str(validation_path.parent)
            self._register_coco_dataset(
                val_dataset_name,
                str(self._validation_file),
                val_image_root
            )
        else:
            # Use training set for validation if no validation set
            val_dataset_name = train_dataset_name

        # Determine number of classes
        num_classes = int(self._num_classes)
        if num_classes <= 0:
            num_classes = len(thing_classes)

        print(f"[Detectron2Trainer] Training with {num_classes} classes: {thing_classes}")

        # Set up configuration
        cfg = self._setup_detectron2_config(
            num_classes,
            train_dataset_name,
            val_dataset_name
        )

        # Resume from checkpoint if requested
        if parse_bool(self._resume):
            # Look for last checkpoint in output directory
            last_checkpoint = ub.Path(cfg.OUTPUT_DIR) / "last_checkpoint"
            if last_checkpoint.exists():
                with open(last_checkpoint, 'r') as f:
                    checkpoint_path = f.read().strip()
                if ub.Path(checkpoint_path).exists():
                    cfg.MODEL.WEIGHTS = checkpoint_path
                    print(f"[Detectron2Trainer] Resuming from {checkpoint_path}")

        # Custom trainer with COCO evaluation
        class Trainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name):
                output_folder = ub.Path(cfg.OUTPUT_DIR) / "eval"
                output_folder.ensuredir()
                return COCOEvaluator(dataset_name, output_dir=str(output_folder))

        # Signal handler for graceful interruption
        with TrainingInterruptHandler("Detectron2Trainer") as handler:
            try:
                trainer = Trainer(cfg)
                trainer.resume_or_load(resume=parse_bool(self._resume))
                trainer.train()
            except KeyboardInterrupt:
                print("[Detectron2Trainer] Training interrupted by user")

            self._interrupted = handler.interrupted

        # Save final model
        self.save_final_model(cfg, thing_classes)

        print("\n[Detectron2Trainer] Model training complete!\n")

    def save_final_model(self, cfg=None, class_names=None):
        """
        Save the final trained model and generate pipeline file.

        Args:
            cfg: Detectron2 config used for training
            class_names: List of class names
        """
        import torch
        import shutil

        if len(self._pipeline_template) == 0:
            return

        output_model_name = "trained_detectron2_model.pth"
        output_dpath = ub.Path(self._output_directory)
        output_model = output_dpath / output_model_name

        if cfg is not None:
            output_dir = ub.Path(cfg.OUTPUT_DIR)

            # Find the best model (or last model)
            model_final = output_dir / "model_final.pth"
            if model_final.exists():
                final_model_path = model_final
            else:
                # Look for checkpoint files
                checkpoints = sorted(output_dir.glob("model_*.pth"))
                if checkpoints:
                    final_model_path = checkpoints[-1]
                else:
                    print("[Detectron2Trainer] No model checkpoint found")
                    return

            # Load checkpoint and add class names metadata
            checkpoint = torch.load(final_model_path, map_location='cpu')

            # Add class names to checkpoint for later retrieval
            if class_names:
                if 'args' not in checkpoint:
                    checkpoint['args'] = {}
                checkpoint['args']['class_names'] = class_names
                checkpoint['args']['num_classes'] = len(class_names)

            # Save enriched checkpoint
            torch.save(checkpoint, output_model)
            print(f"[Detectron2Trainer] Saved model to {output_model}")

            # Also copy the config file
            config_file = output_dir / "config.yaml"
            if config_file.exists():
                shutil.copy2(config_file, output_dpath / "model_config.yaml")

        # Generate pipeline file
        if ub.Path(self._pipeline_template).exists():
            with open(self._pipeline_template, 'r') as fin:
                all_lines = fin.readlines()

            with open(output_dpath / "detector.pipe", 'w') as fout:
                for line in all_lines:
                    line = line.replace("[-MODEL-FILE-]", output_model_name)
                    line = line.replace("[-MODEL-BASE-]", str(self._base))
                    if hasattr(self, '_resize_option'):
                        line = line.replace("[-WINDOW-OPTION-]", self._resize_option)
                    fout.write(line)

        print(f"\n[Detectron2Trainer] Wrote finalized model to {output_model}")
        print(f"\n[Detectron2Trainer] The {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")


def __vital_algorithm_register__():
    register_vital_algorithm(
        Detectron2Trainer,
        "trainer_detectron2",
        "Detectron2 detection training routine"
    )
