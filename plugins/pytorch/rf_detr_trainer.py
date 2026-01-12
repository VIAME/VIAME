# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import (
    DetectedObjectSetOutput,
    ImageObjectDetector,
    TrainDetector
)

import os

from .kwcoco_train_detector import KWCocoTrainDetector
from .kwcoco_train_detector import KWCocoTrainDetectorConfig

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    vital_config_update,
    resolve_device_str,
    parse_bool,
    register_vital_algorithm,
    TrainingInterruptHandler,
)


class RFDETRTrainerConfig(KWCocoTrainDetectorConfig):
    """
    The configuration for :class:`RFDETRTrainer`.
    """
    identifier = "viame-rf-detr-detector"
    train_directory = "deep_training"
    output_directory = "category_models"
    seed_model = ""

    tmp_training_file = "training_truth.json"
    tmp_validation_file = "validation_truth.json"

    # RF-DETR model configuration
    model_size = scfg.Value('base', help='Model size: nano, small, medium, base, or large')
    device = scfg.Value('auto', help='Device to train on: auto, cpu, cuda, or cuda:N')

    # Training hyperparameters
    max_epochs = scfg.Value(100, help='Maximum number of epochs to train for')
    batch_size = scfg.Value(4, help='Number of images per batch')
    learning_rate = scfg.Value(1e-4, help='Learning rate')
    learning_rate_encoder = scfg.Value(1.5e-4, help='Learning rate for encoder')
    grad_accum_steps = scfg.Value(4, help='Gradient accumulation steps')
    weight_decay = scfg.Value(1e-4, help='Weight decay for optimizer')
    warmup_epochs = scfg.Value(0.0, help='Number of warmup epochs')
    lr_drop = scfg.Value(100, help='Epoch to drop learning rate')

    # EMA settings
    use_ema = scfg.Value(True, help='Use exponential moving average')
    ema_decay = scfg.Value(0.993, help='EMA decay rate')

    # Early stopping
    early_stopping = scfg.Value(False, help='Enable early stopping')
    early_stopping_patience = scfg.Value(10, help='Early stopping patience')

    # Data augmentation
    multi_scale = scfg.Value(True, help='Use multi-scale training')

    # Checkpointing
    checkpoint_interval = scfg.Value(10, help='Save checkpoint every N epochs')

    # Logging
    use_tensorboard = scfg.Value(True, help='Enable TensorBoard logging')

    pipeline_template = ""

    categories = []

    def __post_init__(self):
        super().__post_init__()


class RFDETRTrainer(KWCocoTrainDetector):
    """
    Implementation of TrainDetector for RF-DETR models
    """
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = RFDETRTrainerConfig()

    def get_configuration(self):
        print('[RFDETRTrainer] get_configuration')
        cfg = super().get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        print('[RFDETRTrainer] set_configuration')
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))
        self._config.__post_init__()

        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._post_config_set()
        return True

    def _post_config_set(self):
        print('[RFDETRTrainer] _post_config_set')
        assert self._config['mode'] == "detector"

        if self._train_directory is not None:
            if not os.path.exists(self._train_directory):
                os.mkdir(self._train_directory)
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
                os.mkdir(self._output_directory)

        from kwiver.vital.modules import load_known_modules
        load_known_modules()

        if not self._no_format:
            self._training_writer = DetectedObjectSetOutput.create("coco")
            self._validation_writer = DetectedObjectSetOutput.create("coco")

            writer_conf = self._training_writer.get_configuration()
            self._training_writer.set_configuration(writer_conf)

            writer_conf = self._validation_writer.get_configuration()
            self._validation_writer.set_configuration(writer_conf)

            self._training_writer.open(self._training_file)
            self._validation_writer.open(self._validation_file)

        if self._mode == "detection_refiner" and not os.path.exists(self._chip_directory):
            os.mkdir(self._chip_directory)

        if self._detector_model:
            self._detector = ImageObjectDetector.create("yolo")
            detector_config = self._detector.get_configuration()
            detector_config.set_value("deployed", self._detector_model)
            if not self._detector.set_configuration(detector_config):
                print("Unable to configure detector")
                return False

        if self._chip_extension and self._chip_extension[0] != '.':
            self._chip_extension = '.' + self._chip_extension

        if int(self._chip_height) <= 0:
            self._chip_height = self._chip_width
        if int(self._chip_width) <= 0:
            self._chip_width = self._chip_height

        self._training_data = []
        self._validation_data = []
        self._sample_count = 0

    def _ensure_format_writers(self):
        if not self._no_format:
            self._training_writer.complete()
            self._validation_writer.complete()

            import kwcoco
            paths_to_fix = [self._training_file, self._validation_file]
            for fpath in paths_to_fix:
                fpath = ub.Path(fpath)
                if fpath.exists():
                    dset = kwcoco.CocoDataset(fpath)
                    dset.conform()
                    dset.dump()

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def _convert_coco_to_roboflow_format(self, train_coco_path, val_coco_path, output_dir):
        """
        Convert COCO format annotations to Roboflow format expected by RF-DETR.

        Roboflow format expects:
          output_dir/
            train/
              _annotations.coco.json
            valid/
              _annotations.coco.json
            test/
              _annotations.coco.json

        Uses absolute paths in the COCO JSON to avoid copying images.
        """
        import json

        output_dir = ub.Path(output_dir)
        train_dir = output_dir / "train"
        valid_dir = output_dir / "valid"
        test_dir = output_dir / "test"

        # Create directory structure
        (train_dir).ensuredir()
        (valid_dir).ensuredir()
        (test_dir).ensuredir()

        def process_split(coco_path, split_dir):
            coco_path = ub.Path(coco_path)
            if not coco_path.exists():
                print(f"[RFDETRTrainer] Warning: {coco_path} does not exist")
                return

            with open(coco_path, 'r') as f:
                coco_data = json.load(f)

            # Convert to absolute paths without copying
            new_images = []
            for img in coco_data.get('images', []):
                old_path = ub.Path(img['file_name'])
                if not old_path.is_absolute():
                    old_path = coco_path.parent / old_path

                if old_path.exists():
                    # Use absolute path directly
                    img['file_name'] = str(old_path.resolve())
                    new_images.append(img)

            coco_data['images'] = new_images

            # Ensure categories have supercategory field (required by RF-DETR)
            # Also remap category IDs to be 0-indexed (RF-DETR expects 0-indexed)
            old_id_to_new = {}
            for new_id, cat in enumerate(coco_data.get('categories', [])):
                old_id_to_new[cat['id']] = new_id
                cat['id'] = new_id
                if 'supercategory' not in cat:
                    cat['supercategory'] = cat.get('name', 'object')

            # Update annotation category IDs
            for ann in coco_data.get('annotations', []):
                if ann['category_id'] in old_id_to_new:
                    ann['category_id'] = old_id_to_new[ann['category_id']]

            # Write the annotations file
            annotations_path = split_dir / "_annotations.coco.json"
            with open(annotations_path, 'w') as f:
                json.dump(coco_data, f)

            print(f"[RFDETRTrainer] Prepared {len(new_images)} images for {split_dir}")

        process_split(train_coco_path, train_dir)
        process_split(val_coco_path, valid_dir)
        # RF-DETR requires a test split; use validation data for test
        process_split(val_coco_path, test_dir)

        return output_dir

    def update_model(self):
        import torch

        self._ensure_format_writers()

        print("[RFDETRTrainer] Starting RF-DETR training")

        # Prepare dataset directory in Roboflow format
        dataset_dir = ub.Path(self._train_directory) / "rf_detr_dataset"
        dataset_dir.ensuredir()

        self._convert_coco_to_roboflow_format(
            self._training_file,
            self._validation_file,
            dataset_dir
        )

        # Determine device
        device = resolve_device_str(self._device)

        # Select model class based on size
        model_size = self._model_size.lower()
        if model_size == 'nano':
            from rfdetr import RFDETRNano as RFDETRModel
        elif model_size == 'small':
            from rfdetr import RFDETRSmall as RFDETRModel
        elif model_size == 'medium':
            from rfdetr import RFDETRMedium as RFDETRModel
        elif model_size == 'base':
            from rfdetr import RFDETRBase as RFDETRModel
        elif model_size == 'large':
            from rfdetr import RFDETRLarge as RFDETRModel
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        print(f"[RFDETRTrainer] Using RF-DETR {model_size} model on {device}")

        # Create model
        if len(self._seed_model) > 0 and ub.Path(self._seed_model).exists():
            # Load from checkpoint
            checkpoint = torch.load(self._seed_model, map_location=device)
            if 'args' in checkpoint and 'num_classes' in checkpoint['args']:
                num_classes = checkpoint['args']['num_classes']
            else:
                num_classes = len(self._categories) if self._categories else 90

            model = RFDETRModel(
                pretrain_weights=None,
                num_classes=num_classes,
                device=device
            )
            if 'model' in checkpoint:
                model.model.model.load_state_dict(checkpoint['model'])
        else:
            # Use pretrained weights
            model = RFDETRModel(device=device)

        # Parse training parameters
        epochs = int(self._max_epochs)
        batch_size = int(self._batch_size)
        lr = float(self._learning_rate)
        lr_encoder = float(self._learning_rate_encoder)
        grad_accum_steps = int(self._grad_accum_steps)
        weight_decay = float(self._weight_decay)
        warmup_epochs = float(self._warmup_epochs)
        lr_drop = int(self._lr_drop)
        use_ema = parse_bool(self._use_ema)
        ema_decay = float(self._ema_decay)
        early_stopping = parse_bool(self._early_stopping)
        early_stopping_patience = int(self._early_stopping_patience)
        multi_scale = parse_bool(self._multi_scale)
        checkpoint_interval = int(self._checkpoint_interval)
        use_tensorboard = parse_bool(self._use_tensorboard)

        output_dir = ub.Path(self._train_directory) / "rf_detr_output"
        output_dir.ensuredir()

        # Signal handler for graceful interruption
        with TrainingInterruptHandler("RFDETRTrainer") as handler:
            try:
                # Train the model
                model.train(
                    dataset_dir=str(dataset_dir),
                    output_dir=str(output_dir),
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    lr_encoder=lr_encoder,
                    grad_accum_steps=grad_accum_steps,
                    weight_decay=weight_decay,
                    warmup_epochs=warmup_epochs,
                    lr_drop=lr_drop,
                    use_ema=use_ema,
                    ema_decay=ema_decay,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience,
                    multi_scale=multi_scale,
                    checkpoint_interval=checkpoint_interval,
                    tensorboard=use_tensorboard,
                    wandb=False,
                )
            except KeyboardInterrupt:
                print("[RFDETRTrainer] Training interrupted by user")

            self._interrupted = handler.interrupted

        self.save_final_model(model, output_dir)

        print("\n[RFDETRTrainer] Model training complete!\n")

    def save_final_model(self, model=None, output_dir=None):
        import shutil

        output_model_name = "trained_rf_detr_checkpoint.pth"
        output_dpath = ub.Path(self._output_directory)
        output_model = output_dpath / output_model_name

        # Find the best checkpoint
        if output_dir is not None:
            output_dir = ub.Path(output_dir)
            checkpoint_candidates = sorted(output_dir.glob("*.pth"))

            # Prefer best checkpoints in order of preference
            best_candidates = [
                output_dir / "checkpoint_best_total.pth",
                output_dir / "checkpoint_best_ema.pth",
                output_dir / "checkpoint_best_regular.pth",
                output_dir / "checkpoint_best.pth",
            ]

            final_ckpt = None
            for candidate in best_candidates:
                if candidate.exists():
                    final_ckpt = candidate
                    break

            if final_ckpt is None and checkpoint_candidates:
                final_ckpt = checkpoint_candidates[-1]

            if final_ckpt is None:
                print("[RFDETRTrainer] No checkpoint found")
                return

            # Copy checkpoint to output directory
            shutil.copy2(final_ckpt, output_model)
            print(f"[RFDETRTrainer] Copied {final_ckpt} to {output_model}")
        else:
            print("[RFDETRTrainer] No output directory specified")
            return

        # Generate pipeline file if template exists
        if len(self._pipeline_template) > 0 and ub.Path(self._pipeline_template).exists():
            with open(self._pipeline_template, 'r') as fin:
                all_lines = fin.readlines()

            with open(output_dpath / "detector.pipe", 'w') as fout:
                for line in all_lines:
                    line = line.replace("[-MODEL-FILE-]", output_model_name)
                    line = line.replace("[-WINDOW-OPTION-]", self._resize_option)
                    fout.write(line)

        print(f"[RFDETRTrainer] Wrote finalized model to {output_model}")
        print(f"[RFDETRTrainer] The {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")


def __vital_algorithm_register__():
    register_vital_algorithm(
        RFDETRTrainer, "rf_detr", "PyTorch RF-DETR detection training routine"
    )
