# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json
import os
import sys

from kwiver.vital.algo import TrainDetector

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    vital_config_update,
    resolve_device_str,
    parse_bool,
    register_vital_algorithm,
    TrainingInterruptHandler,
    ensure_rfdetr_compatibility,
)


class RFDETRTrainerConfig(scfg.DataConfig):
    """
    The configuration for :class:`RFDETRTrainer`.
    """
    identifier = "viame-rf-detr-detector"
    train_directory = "deep_training"
    seed_model = ""

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

    # Timeout
    timeout = scfg.Value('1209600', help='Max training time in seconds (default=1209600, two weeks)')

    categories = []

    def __post_init__(self):
        super().__post_init__()


class RFDETRTrainer(TrainDetector):
    """
    Implementation of TrainDetector for RF-DETR models.
    """
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = RFDETRTrainerConfig()
        self._class_names = []
        self._train_image_files = []
        self._train_detections = []
        self._test_image_files = []
        self._test_detections = []

    def get_configuration(self):
        print('[RFDETRTrainer] get_configuration')
        cfg = super(TrainDetector, self).get_configuration()
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

        if self._train_directory is not None:
            if not os.path.exists(self._train_directory):
                os.mkdir(self._train_directory)

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(self, categories, train_files, train_dets,
                           test_files, test_dets):
        print("[RFDETRTrainer] Adding training data from disk")
        print("  Training files: ", len(train_files))
        print("  Training detections: ", len(train_dets))
        print("  Test files: ", len(test_files))
        print("  Test detections: ", len(test_dets))

        if categories is not None:
            self._class_names = categories.all_class_names()
        else:
            self._class_names = []

        self._train_image_files = list(train_files)
        self._train_detections = list(train_dets)
        self._test_image_files = list(test_files)
        self._test_detections = list(test_dets)

    def _prepare_roboflow_dataset(self):
        """
        Convert stored kwiver data directly to Roboflow directory format
        expected by RF-DETR.

        Creates:
          rf_detr_dataset/
            train/_annotations.coco.json
            valid/_annotations.coco.json
            test/_annotations.coco.json

        Returns (dataset_dir, class_names).
        """
        from PIL import Image

        dataset_dir = ub.Path(self._train_directory) / "rf_detr_dataset"
        train_dir = dataset_dir / "train"
        valid_dir = dataset_dir / "valid"
        test_dir = dataset_dir / "test"

        train_dir.ensuredir()
        valid_dir.ensuredir()
        test_dir.ensuredir()

        # Build category mapping (0-indexed for RF-DETR)
        class_names = list(self._class_names)
        cat_name_to_id = {name: idx for idx, name in enumerate(class_names)}

        categories_json = [
            {"id": idx, "name": name, "supercategory": name}
            for idx, name in enumerate(class_names)
        ]

        def build_coco_json(image_files, detection_sets):
            images_json = []
            annotations_json = []
            ann_id = 1

            for img_idx, (img_path, det_set) in enumerate(
                zip(image_files, detection_sets)
            ):
                img_path = str(img_path)
                if not os.path.exists(img_path):
                    continue

                # Read dimensions from header only
                with Image.open(img_path) as im:
                    width, height = im.size

                img_id = img_idx
                abs_path = os.path.abspath(img_path)

                images_json.append({
                    "id": img_id,
                    "file_name": abs_path,
                    "width": width,
                    "height": height,
                })

                if det_set is None:
                    continue

                for det in det_set:
                    bbox = det.bounding_box
                    x1, y1 = bbox.min_x(), bbox.min_y()
                    x2, y2 = bbox.max_x(), bbox.max_y()
                    w = x2 - x1
                    h = y2 - y1

                    if w <= 0 or h <= 0:
                        continue

                    det_type = det.type
                    if det_type is None:
                        continue
                    class_name = det_type.get_most_likely_class()
                    if class_name not in cat_name_to_id:
                        continue

                    annotations_json.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_name_to_id[class_name],
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    })
                    ann_id += 1

            return {
                "images": images_json,
                "annotations": annotations_json,
                "categories": categories_json,
            }

        # Build train split
        train_coco = build_coco_json(
            self._train_image_files, self._train_detections
        )
        with open(train_dir / "_annotations.coco.json", "w") as f:
            json.dump(train_coco, f)
        print(f"[RFDETRTrainer] Train: {len(train_coco['images'])} images, "
              f"{len(train_coco['annotations'])} annotations")

        # Build valid split (from test data)
        valid_coco = build_coco_json(
            self._test_image_files, self._test_detections
        )
        with open(valid_dir / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)
        print(f"[RFDETRTrainer] Valid: {len(valid_coco['images'])} images, "
              f"{len(valid_coco['annotations'])} annotations")

        # RF-DETR requires a test split; reuse validation data
        with open(test_dir / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)

        return dataset_dir, class_names

    def update_model(self):
        import torch

        # On Windows, reconfigure stdout/stderr to use UTF-8 encoding.
        # Third-party libraries (rich, etc.) use emoji characters that
        # cannot be encoded in the default cp1252 Windows codepage.
        if sys.platform == 'win32':
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, 'reconfigure'):
                    stream.reconfigure(encoding='utf-8', errors='replace')

        print("[RFDETRTrainer] Starting RF-DETR training")

        # Prepare dataset directory in Roboflow format
        dataset_dir, self._class_names = self._prepare_roboflow_dataset()

        # Determine device
        device = resolve_device_str(self._device)

        ensure_rfdetr_compatibility()

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
            checkpoint = torch.load(self._seed_model, map_location=device, weights_only=False)
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

        # Parse timeout (in seconds, or "default" for no limit)
        import time
        from collections import defaultdict
        timeout_str = str(self._timeout).lower()
        if timeout_str == "default" or timeout_str == "none" or timeout_str == "":
            timeout_seconds = None
        else:
            timeout_seconds = float(self._timeout)

        output_dir = ub.Path(self._train_directory) / "rf_detr_output"
        output_dir.ensuredir()

        # Add timeout callback to model's callbacks if timeout is specified
        if timeout_seconds is not None:
            train_start_time = time.time()
            def timeout_callback(log_stats):
                elapsed = time.time() - train_start_time
                if elapsed >= timeout_seconds:
                    print(f"[RFDETRTrainer] Timeout reached ({elapsed:.0f}s >= {timeout_seconds:.0f}s)")
                    model.model.request_early_stop()
            model.callbacks["on_fit_epoch_end"].append(timeout_callback)

        # On Windows, DataLoader worker subprocesses fail because
        # multiprocessing spawn tries to re-invoke viame.exe as Python.
        train_kwargs = dict(
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
        if sys.platform == "win32":
            train_kwargs["num_workers"] = 0

        # Signal handler for graceful interruption
        with TrainingInterruptHandler("RFDETRTrainer", on_interrupt=model.model.request_early_stop) as handler:
            try:
                # Train the model
                model.train(**train_kwargs)
            except KeyboardInterrupt:
                print("[RFDETRTrainer] Training interrupted by user")

            self._interrupted = handler.interrupted

        output = self._get_output_map(output_dir)

        print("\n[RFDETRTrainer] Model training complete!")

        return output

    def _get_output_map(self, output_dir):
        """
        Build and return output map containing template replacements and
        file copies. The C++ process_trainer_output handles copying files
        to the output directory and generating the pipeline from a template.
        """
        output = {}
        output_model_name = "trained_detector.pth"

        # Find the best checkpoint
        output_dir = ub.Path(output_dir)
        checkpoint_candidates = sorted(output_dir.glob("*.pth"))

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
            print("\n[RFDETRTrainer] No checkpoint found, training may have failed")
            return output

        # Embed class metadata into the checkpoint so the detector knows
        # the trained categories and model architecture.
        if hasattr(self, '_class_names') and self._class_names:
            import torch
            checkpoint = torch.load(final_ckpt, map_location='cpu', weights_only=False)
            if not isinstance(checkpoint, dict):
                checkpoint = {'model': checkpoint}
            args = checkpoint.get('args', {})
            if not isinstance(args, dict):
                args = vars(args)
            if 'num_classes' not in args:
                args['num_classes'] = len(self._class_names)
            args['class_names'] = self._class_names
            args['model_size'] = self._model_size
            checkpoint['args'] = args
            torch.save(checkpoint, final_ckpt)
            print(f"[RFDETRTrainer] Embedded {len(self._class_names)} class names into checkpoint")

        algo = "rf_detr"

        output["type"] = algo

        # Config key matching rf_detr detector inference config
        output[algo + ":weight"] = output_model_name

        # File copy entry (key=destination filename, value=source path)
        output[output_model_name] = str(final_ckpt)

        print(f"\nModel found at: {final_ckpt}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        RFDETRTrainer, "rf_detr", "PyTorch RF-DETR detection training routine"
    )
