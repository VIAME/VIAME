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


class LitDetTrainerConfig(KWCocoTrainDetectorConfig):
    """
    The configuration for :class:`LitDetTrainer`.
    """
    identifier = "viame-litdet-detector"
    train_directory = "deep_training"
    output_directory = "category_models"
    seed_model = ""

    tmp_training_file = "training_truth.json"
    tmp_validation_file = "validation_truth.json"

    # LitDet model configuration
    model_type = scfg.Value('faster_rcnn', help='Model type: faster_rcnn or fcnn')
    backbone = scfg.Value('resnet50_fpn', help='Backbone architecture')
    device = scfg.Value('auto', help='Device to train on: auto, cpu, cuda, or cuda:N')

    # Training hyperparameters
    max_epochs = scfg.Value(100, help='Maximum number of epochs to train for')
    batch_size = scfg.Value(2, help='Number of images per batch')
    learning_rate = scfg.Value(1e-3, help='Learning rate')
    weight_decay = scfg.Value(1e-6, help='Weight decay for optimizer')
    num_workers = scfg.Value(0, help='Number of data loader workers')

    # Model architecture settings
    trainable_backbone_layers = scfg.Value(3, help='Number of trainable backbone layers')
    compile_model = scfg.Value(False, help='Whether to compile the model with torch.compile')

    # Checkpointing
    save_top_k = scfg.Value(1, help='Save top k checkpoints')

    # Logging
    use_tensorboard = scfg.Value(True, help='Enable TensorBoard logging')

    # Testing
    run_test = scfg.Value(True, help='Run test evaluation after training')

    # Seed
    seed = scfg.Value(None, help='Random seed for reproducibility')

    pipeline_template = ""

    categories = []

    def __post_init__(self):
        super().__post_init__()


class LitDetTrainer(KWCocoTrainDetector):
    """
    Implementation of TrainDetector for LitDet (Lightning Hydra Detection) models.

    Uses the Hydra Python API to configure and train detection models.
    """
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = LitDetTrainerConfig()

    def get_configuration(self):
        print('[LitDetTrainer] get_configuration')
        cfg = super().get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        print('[LitDetTrainer] set_configuration')
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
        print('[LitDetTrainer] _post_config_set')
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

    def _prepare_litdet_data_structure(self, train_coco_path, val_coco_path, output_dir):
        """
        Prepare the data directory structure expected by LitDet's COCODataModule.

        LitDet expects:
          output_dir/
            images/
              train/
              valid/
              test/
            labels/
              train/
                instances_train.json
              valid/
                instances_valid.json

        Uses absolute paths in the COCO JSON to avoid copying images.
        """
        import json
        import shutil

        output_dir = ub.Path(output_dir)

        # Create directory structure
        images_train_dir = output_dir / "images" / "train"
        images_valid_dir = output_dir / "images" / "valid"
        images_test_dir = output_dir / "images" / "test"
        labels_train_dir = output_dir / "labels" / "train"
        labels_valid_dir = output_dir / "labels" / "valid"

        images_train_dir.ensuredir()
        images_valid_dir.ensuredir()
        images_test_dir.ensuredir()
        labels_train_dir.ensuredir()
        labels_valid_dir.ensuredir()

        def process_split(coco_path, images_dir, labels_dir, split_name):
            coco_path = ub.Path(coco_path)
            if not coco_path.exists():
                print(f"[LitDetTrainer] Warning: {coco_path} does not exist")
                return 0, []

            with open(coco_path, 'r') as f:
                coco_data = json.load(f)

            # Symlink images and update paths
            new_images = []
            for img in coco_data.get('images', []):
                old_path = ub.Path(img['file_name'])
                if not old_path.is_absolute():
                    old_path = coco_path.parent / old_path

                if old_path.exists():
                    # Create symlink in images directory
                    new_path = images_dir / old_path.name
                    if not new_path.exists():
                        try:
                            new_path.symlink_to(old_path.resolve())
                        except OSError:
                            # Fall back to copy if symlink fails
                            shutil.copy2(old_path, new_path)

                    # Update file_name to just the filename (relative to images dir)
                    img['file_name'] = old_path.name
                    new_images.append(img)

            coco_data['images'] = new_images

            # Ensure categories have required fields
            categories = []
            for cat in coco_data.get('categories', []):
                if 'supercategory' not in cat:
                    cat['supercategory'] = cat.get('name', 'object')
                categories.append(cat)
            coco_data['categories'] = categories

            # Write the annotations file
            annotations_path = labels_dir / f"instances_{split_name}.json"
            with open(annotations_path, 'w') as f:
                json.dump(coco_data, f)

            print(f"[LitDetTrainer] Prepared {len(new_images)} images for {split_name}")
            return len(new_images), categories

        num_train, categories = process_split(
            train_coco_path, images_train_dir, labels_train_dir, "train")
        num_valid, _ = process_split(
            val_coco_path, images_valid_dir, labels_valid_dir, "valid")

        # Copy validation to test (LitDet needs test set)
        if num_valid > 0:
            val_ann_path = labels_valid_dir / "instances_valid.json"
            if val_ann_path.exists():
                with open(val_ann_path, 'r') as f:
                    val_data = json.load(f)

                # Symlink validation images to test
                for img in val_data.get('images', []):
                    src = images_valid_dir / img['file_name']
                    dst = images_test_dir / img['file_name']
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src.resolve())
                        except OSError:
                            shutil.copy2(src, dst)

        return output_dir, len(categories)

    def _build_hydra_config(self, data_dir, num_classes, output_dir):
        """
        Build a Hydra DictConfig programmatically for LitDet training.
        """
        from omegaconf import DictConfig, OmegaConf

        # Parse training parameters
        max_epochs = int(self._max_epochs)
        batch_size = int(self._batch_size)
        lr = float(self._learning_rate)
        weight_decay = float(self._weight_decay)
        num_workers = int(self._num_workers)
        trainable_backbone_layers = int(self._trainable_backbone_layers)
        compile_model = parse_bool(self._compile_model)
        save_top_k = int(self._save_top_k)
        use_tensorboard = parse_bool(self._use_tensorboard)
        run_test = parse_bool(self._run_test)
        seed = self._seed if self._seed and self._seed != 'None' else None
        if seed is not None:
            seed = int(seed)

        # Resolve device
        device_str = resolve_device_str(self._device)
        if device_str.startswith('cuda'):
            accelerator = 'gpu'
            if ':' in device_str:
                devices = [int(device_str.split(':')[1])]
            else:
                devices = 1
        else:
            accelerator = 'cpu'
            devices = 1

        # Build the config dictionary
        cfg_dict = {
            # Stage and tags
            'stage': 'train',
            'tags': ['viame-training'],

            # Training flags
            'train': True,
            'test': run_test,
            'ckpt_path': None,
            'seed': seed,

            # Data configuration
            'data': {
                '_target_': 'lightning_hydra_detection.data.coco_datamodule.COCODataModule',
                '_convert_': 'object',
                'data_dir': str(data_dir),
                'data_name': '',  # Empty since data_dir already points to the dataset
                'batch_size': batch_size,
                'auto_partition_from_train': False,
                'train_val_test_split': [0.8, 0.1, 0.1],
                'num_workers': num_workers,
                'pin_memory': True,
            },

            # Task configuration
            'task': {
                '_target_': 'lightning_hydra_detection.tasks.detect_module.DetectLitModule',
                'model': {
                    '_target_': 'lightning_hydra_detection.tasks.components.faster_rcnn.build_faster_rcnn_resnet50_fpn',
                    'weights': {
                        '_target_': 'hydra.utils.get_object',
                        'path': 'torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT',
                    },
                    'num_classes': num_classes + 1,  # +1 for background
                    'trainable_backbone_layers': trainable_backbone_layers,
                },
                'optimizer': {
                    '_target_': 'torch.optim.Adam',
                    '_partial_': True,
                    'lr': lr,
                    'weight_decay': weight_decay,
                },
                'lr_decay_config': None,
                'lr_scheduler_config': None,
                'compile': compile_model,
            },

            # Trainer configuration
            'trainer': {
                '_target_': 'lightning.Trainer',
                'default_root_dir': str(output_dir),
                'min_epochs': 1,
                'max_epochs': max_epochs,
                'accelerator': accelerator,
                'devices': devices,
                'check_val_every_n_epoch': 1,
                'deterministic': False,
            },

            # Callbacks
            'callbacks': {
                'model_checkpoint': {
                    '_target_': 'lightning.pytorch.callbacks.ModelCheckpoint',
                    'dirpath': str(output_dir / 'checkpoints'),
                    'filename': 'epoch_{epoch:03d}',
                    'monitor': 'val/acc',
                    'verbose': False,
                    'save_last': True,
                    'save_top_k': save_top_k,
                    'mode': 'max',
                    'auto_insert_metric_name': False,
                    'save_weights_only': False,
                    'every_n_train_steps': None,
                    'train_time_interval': None,
                    'every_n_epochs': None,
                    'save_on_train_epoch_end': None,
                },
                'early_stopping': {
                    '_target_': 'lightning.pytorch.callbacks.EarlyStopping',
                    'monitor': 'val/acc',
                    'min_delta': 0.0,
                    'patience': 10,
                    'verbose': False,
                    'mode': 'max',
                    'strict': True,
                    'check_finite': True,
                    'stopping_threshold': None,
                    'divergence_threshold': None,
                    'check_on_train_epoch_end': None,
                },
                'rich_progress_bar': {
                    '_target_': 'lightning.pytorch.callbacks.RichProgressBar',
                },
            },

            # Logger configuration
            'logger': None,

            # Paths configuration
            'paths': {
                'root_dir': str(output_dir),
                'data_dir': str(data_dir),
                'log_dir': str(output_dir / 'logs'),
                'output_dir': str(output_dir),
                'work_dir': str(output_dir),
            },

            # Extras
            'extras': {
                'ignore_warnings': False,
                'enforce_tags': False,
                'print_config': True,
            },
        }

        # Add logger if tensorboard is enabled
        if use_tensorboard:
            cfg_dict['logger'] = {
                'tensorboard': {
                    '_target_': 'lightning.pytorch.loggers.tensorboard.TensorBoardLogger',
                    'save_dir': str(output_dir / 'logs'),
                    'name': 'litdet',
                    'log_graph': False,
                    'default_hp_metric': True,
                    'prefix': '',
                }
            }

        cfg = OmegaConf.create(cfg_dict)
        return cfg

    def update_model(self):
        import torch
        from hydra.core.global_hydra import GlobalHydra
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        self._ensure_format_writers()

        print("[LitDetTrainer] Starting LitDet training")

        # Prepare dataset directory in LitDet format
        dataset_dir = ub.Path(self._train_directory) / "litdet_dataset"
        dataset_dir.ensuredir()

        output_dir = ub.Path(self._train_directory) / "litdet_output"
        output_dir.ensuredir()

        data_dir, num_classes = self._prepare_litdet_data_structure(
            self._training_file,
            self._validation_file,
            dataset_dir
        )

        print(f"[LitDetTrainer] Prepared data with {num_classes} classes")

        # Build Hydra config
        cfg = self._build_hydra_config(data_dir, num_classes, output_dir)

        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Set up minimal HydraConfig for the train function
        # Note: The train function uses HydraConfig internally
        hydra_cfg = OmegaConf.create({
            'run': {'dir': str(output_dir)},
            'sweep': {'dir': str(output_dir), 'subdir': ''},
            'runtime': {'output_dir': str(output_dir), 'choices': {}},
            'verbose': False,
        })

        # Import and call the train function
        from lightning_hydra_detection.train import train

        # Signal handler for graceful interruption
        with TrainingInterruptHandler("LitDetTrainer") as handler:
            try:
                # Initialize HydraConfig with our config
                HydraConfig().set_config(cfg)

                # Run training
                metric_dict, object_dict = train(cfg)

                print(f"[LitDetTrainer] Training metrics: {metric_dict}")

            except KeyboardInterrupt:
                print("[LitDetTrainer] Training interrupted by user")

            self._interrupted = handler.interrupted

        self.save_final_model(output_dir)

        print("\n[LitDetTrainer] Model training complete!\n")

    def save_final_model(self, output_dir=None):
        import shutil

        output_model_name = "trained_litdet_checkpoint.ckpt"
        output_dpath = ub.Path(self._output_directory)
        output_model = output_dpath / output_model_name

        # Find the best checkpoint
        if output_dir is not None:
            output_dir = ub.Path(output_dir)
            checkpoint_dir = output_dir / "checkpoints"

            if checkpoint_dir.exists():
                checkpoint_candidates = sorted(checkpoint_dir.glob("*.ckpt"))

                # Prefer 'last.ckpt' or best checkpoint
                best_candidates = [
                    checkpoint_dir / "last.ckpt",
                ]

                # Also look for epoch checkpoints
                epoch_ckpts = sorted(checkpoint_dir.glob("epoch_*.ckpt"))
                if epoch_ckpts:
                    best_candidates.append(epoch_ckpts[-1])

                final_ckpt = None
                for candidate in best_candidates:
                    if candidate.exists():
                        final_ckpt = candidate
                        break

                if final_ckpt is None and checkpoint_candidates:
                    final_ckpt = checkpoint_candidates[-1]

                if final_ckpt is None:
                    print("[LitDetTrainer] No checkpoint found")
                    return

                # Copy checkpoint to output directory
                shutil.copy2(final_ckpt, output_model)
                print(f"[LitDetTrainer] Copied {final_ckpt} to {output_model}")
            else:
                print("[LitDetTrainer] No checkpoint directory found")
                return
        else:
            print("[LitDetTrainer] No output directory specified")
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

        print(f"[LitDetTrainer] Wrote finalized model to {output_model}")
        print(f"[LitDetTrainer] The {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")


def __vital_algorithm_register__():
    register_vital_algorithm(
        LitDetTrainer, "litdet", "PyTorch LitDet detection training routine"
    )
