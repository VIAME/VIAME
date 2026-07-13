# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json
import os
import sys
import shutil
import subprocess

from kwiver.vital.algo import TrainDetector

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    report_cuda_errors,
    vital_config_update,
    resolve_device_str,
    parse_bool,
    register_vital_algorithm,
    TrainingInterruptHandler,
    ensure_rfdetr_compatibility,
)


# Detection (box-only) and segmentation (mask) RF-DETR variants, by size.
_DET_MODELS = {
    'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium',
    'base': 'RFDETRBase', 'large': 'RFDETRLarge',
}
_SEG_MODELS = {
    'nano': 'RFDETRSegNano', 'small': 'RFDETRSegSmall',
    'medium': 'RFDETRSegMedium', 'large': 'RFDETRSegLarge',
}


def select_model_class(model_size, segmentation):
    """Return the RF-DETR model class for a size + detection/segmentation mode."""
    table = _SEG_MODELS if segmentation else _DET_MODELS
    key = str(model_size).lower()
    if key not in table:
        kind = 'segmentation' if segmentation else 'detection'
        raise ValueError(f"Unknown {kind} model size: {model_size}")
    import rfdetr
    return getattr(rfdetr, table[key])


def polygon_area(flat_poly):
    """Shoelace area of a flattened [x1,y1,x2,y2,...] polygon (>=3 points)."""
    n = len(flat_poly) // 2
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = flat_poly[2 * i], flat_poly[2 * i + 1]
        j = (i + 1) % n
        x2, y2 = flat_poly[2 * j], flat_poly[2 * j + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


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
    devices = scfg.Value('auto', help=(
        'GPU count for training. "auto" uses all visible GPUs, training with '
        'DDP when more than one is present. Pinning device=cuda:N forces 1.'))
    strategy = scfg.Value('auto', help=(
        'Multi-GPU strategy. "auto" uses ddp_find_unused_parameters_true when '
        'training on more than one GPU.'))
    num_channels = scfg.Value(3, help=(
        'Number of input channels. 3 = RGB; 4 = RGB + a motion/flow channel '
        '(RGBA). RF-DETR adapts the pretrained input conv to match.'))
    resolution = scfg.Value(0, help=(
        'Square input resolution fed to the network. 0 = use the model-size '
        'default (large=704). Larger values let the network resolve smaller '
        'objects on high-res imagery without gridding; the pretrained '
        'positional embeddings are bicubic-interpolated to the new grid. '
        'Must be divisible by patch_size*num_windows (32 for large, 56 for '
        'base, 32 for nano/small/medium).'))
    gradient_checkpointing = scfg.Value(False, help=(
        'Trade compute for memory by recomputing backbone activations in the '
        'backward pass (~30%% slower, roughly halves activation memory). '
        'Enable to fit higher resolutions or larger batches on limited VRAM.'))
    segmentation = scfg.Value(False, help=(
        'Train a segmentation (mask) model (RFDETRSeg*) instead of the '
        'detection-only model. Requires polygon annotations in the training '
        'data; the chip polygons are written to the COCO segmentation field. '
        'Note: seg model sizes (e.g. large) constrain the input resolution '
        '(divisible by patch_size*num_windows = 24 for seg-large).'))

    # Training hyperparameters
    max_epochs = scfg.Value(100, help='Maximum number of epochs to train for')
    batch_size = scfg.Value(4, help=(
        'Images per micro-batch, or "auto" to let RF-DETR probe for the '
        'largest micro-batch that fits in VRAM and derive grad_accum_steps to '
        'reach auto_batch_target_effective.'))
    auto_batch_target_effective = scfg.Value(16, help=(
        'Per-device effective batch size (micro-batch * grad_accum) targeted '
        'when batch_size="auto". Ignored for a fixed batch_size.'))
    learning_rate = scfg.Value(1e-4, help='Learning rate')
    learning_rate_encoder = scfg.Value(1.5e-4, help='Learning rate for encoder')
    grad_accum_steps = scfg.Value(4, help='Gradient accumulation steps')
    weight_decay = scfg.Value(1e-4, help='Weight decay for optimizer')
    warmup_epochs = scfg.Value(0.0, help='Number of warmup epochs')
    lr_drop = scfg.Value(100, help='Epoch to drop learning rate (step schedule)')
    lr_scheduler = scfg.Value('step', help=(
        'Learning-rate schedule: "step" (hold LR, then a single 10x drop at '
        'lr_drop) or "cosine" (smoothly anneal LR toward 0 across all epochs). '
        'Cosine + early stopping avoids the constant-LR overfitting seen with '
        'a step drop scheduled beyond the run length.'))

    # EMA settings
    use_ema = scfg.Value(True, help='Use exponential moving average')
    ema_decay = scfg.Value(0.993, help='EMA decay rate')

    # Early stopping
    early_stopping = scfg.Value(False, help='Enable early stopping')
    early_stopping_patience = scfg.Value(10, help='Early stopping patience')

    # Data augmentation
    multi_scale = scfg.Value(True, help='Use multi-scale training')
    augmentation = scfg.Value('default', help=(
        'Augmentation preset: "default" (RF-DETR built-in), "geometric" '
        '(flips only, no photometric ops — safe for motion-infused channels), '
        '"none", or a named preset (conservative, aggressive, aerial, industrial)'))

    # Checkpointing
    checkpoint_interval = scfg.Value(10, help='Save checkpoint every N epochs')

    # DataLoader performance (accuracy-neutral). Seg models augment on the CPU,
    # so the default of 2 workers/GPU can starve the GPUs; set num_workers near
    # the CPUs-per-GPU available. These change throughput only, not the model.
    num_workers = scfg.Value(2, help='DataLoader worker processes per GPU (0 = main process)')
    pin_memory = scfg.Value(True, help='Pin host memory for faster host->GPU copies')
    persistent_workers = scfg.Value(True, help='Keep DataLoader workers alive across epochs')
    prefetch_factor = scfg.Value(4, help='Batches each worker prefetches (only used when num_workers>0)')

    # Validation cost (accuracy-neutral for model selection; do a full eval on
    # the final best checkpoint if the headline number must use every detection).
    eval_interval = scfg.Value(1, help='Run validation every N epochs')
    eval_max_dets = scfg.Value(500, help=(
        'Max detections per image scored during validation. The model emits '
        '~num_queries predictions; values below that drop the lowest-scoring '
        'ones from scoring and shrink the (slow) IoU matrices.'))
    val_subsample = scfg.Value(0, help=(
        'If > 0, validate on at most this many chips (deterministically '
        'sampled). Seg validation forward-passes every vali chip at batch 1 '
        '(batch>1 is unusable), so this dominates epoch time on large vali '
        'sets. A subsample gives a fast, stable selection signal; run a full '
        'eval on the final best checkpoint for the headline number.'))
    max_mask_instances = scfg.Value(0, help=(
        'Cap matched instances per chip used by the segmentation mask loss '
        '(0 = off). The RF-DETR mask loss allocates per-matched-instance '
        'tensors and OOMs on densely-annotated chips (100s of objects); this '
        'bounds it (box/class losses still use all objects). Exported as '
        'RFDETR_MAX_MASK_INSTANCES so the rfdetr criterion (incl. the DDP '
        'subprocess) reads it.'))

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

    @report_cuda_errors("RFDETRTrainer initialization")
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

    def _resolve_aug_config(self, augmentation):
        """
        Map the 'augmentation' config value to an RF-DETR aug_config dict.

        Returns None to defer to RF-DETR's built-in default preset. "geometric"
        is restricted to flips so no photometric/color augmentation runs, which
        would otherwise corrupt motion-infused channels.
        """
        key = str(augmentation).strip().lower()
        if key in ('', 'default'):
            return None
        if key == 'none':
            return {}
        if key == 'geometric':
            return {"HorizontalFlip": {"p": 0.5}, "VerticalFlip": {"p": 0.5}}

        presets = {
            'conservative': 'AUG_CONSERVATIVE',
            'aggressive': 'AUG_AGGRESSIVE',
            'aerial': 'AUG_AERIAL',
            'industrial': 'AUG_INDUSTRIAL',
        }
        if key in presets:
            from rfdetr.datasets import aug_config as rfdetr_aug
            return dict(getattr(rfdetr_aug, presets[key]))

        raise ValueError(f"Unknown augmentation preset: {augmentation}")

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

        seg_enabled = parse_bool(self._segmentation)

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

                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_name_to_id[class_name],
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }

                    # For a segmentation run, attach the chip-space polygon
                    # (carried through by the windowed trainer) as the COCO
                    # segmentation; RF-DETR's loader rasterizes it to a mask.
                    if seg_enabled:
                        poly = list(det.get_flattened_polygon())
                        if len(poly) >= 6:
                            ann["segmentation"] = [poly]
                            area = polygon_area(poly)
                            if area > 0:
                                ann["area"] = area

                    annotations_json.append(ann)
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

        # Build valid split (from test data). Optionally subsample the vali
        # chips: seg validation forward-passes every chip at batch 1, which
        # dominates epoch time on large vali sets. The subsample is
        # deterministic so the selection signal is stable across epochs/runs.
        val_files = self._test_image_files
        val_dets = self._test_detections
        n_sub = int(self._val_subsample)
        if n_sub > 0 and len(val_files) > n_sub:
            import random
            idx = list(range(len(val_files)))
            random.Random(1234).shuffle(idx)
            idx = sorted(idx[:n_sub])
            val_files = [val_files[i] for i in idx]
            val_dets = [val_dets[i] for i in idx]
            print(f"[RFDETRTrainer] Validation subsampled to {len(val_files)} "
                  f"of {len(self._test_image_files)} chips (deterministic)")

        valid_coco = build_coco_json(val_files, val_dets)
        with open(valid_dir / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)
        print(f"[RFDETRTrainer] Valid: {len(valid_coco['images'])} images, "
              f"{len(valid_coco['annotations'])} annotations")

        # RF-DETR requires a test split; reuse validation data
        with open(test_dir / "_annotations.coco.json", "w") as f:
            json.dump(valid_coco, f)

        return dataset_dir, class_names

    @report_cuda_errors("RFDETRTrainer training")
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

        # Make the mask-loss instance cap reproducible from the config: export it
        # to the env the rfdetr criterion reads. The DDP subprocess copies
        # os.environ, so this is inherited there too.
        if int(self._max_mask_instances) > 0:
            os.environ['RFDETR_MAX_MASK_INSTANCES'] = str(int(self._max_mask_instances))

        # Use all available GPUs by default. DDP cannot launch from this embedded
        # interpreter, so multi-GPU training runs in a subprocess (see
        # rf_detr_launcher.py); single-GPU stays in-process below.
        n_gpus = self._resolve_gpu_count(device)
        if n_gpus > 1:
            print(f"[RFDETRTrainer] {n_gpus} GPUs visible; training with DDP "
                  "across all of them")
            return self._train_multi_gpu(dataset_dir, n_gpus)

        # Select model class based on size and detection/segmentation mode
        model_size = self._model_size.lower()
        segmentation = parse_bool(self._segmentation)
        RFDETRModel = select_model_class(model_size, segmentation)
        if segmentation:
            print(f"[RFDETRTrainer] Segmentation (mask) model enabled: "
                  f"RFDETRSeg{model_size}")

        num_channels = int(self._num_channels)
        self._resolution = int(self._resolution)
        gradient_checkpointing = parse_bool(self._gradient_checkpointing)

        # Shared construction kwargs. resolution is a ModelConfig field: passing
        # it builds the network at that grid and triggers bicubic interpolation
        # of the pretrained positional embeddings to match (see
        # rfdetr.models.weights.interpolate_position_embeddings).
        model_kwargs = dict(num_channels=num_channels, device=device)
        if self._resolution > 0:
            model_kwargs['resolution'] = self._resolution
        if gradient_checkpointing:
            model_kwargs['gradient_checkpointing'] = True

        print(f"[RFDETRTrainer] Using RF-DETR {model_size} model on {device} "
              f"with {num_channels} input channels"
              + (f" at {self._resolution}px" if self._resolution > 0 else "")
              + (" (gradient checkpointing)" if gradient_checkpointing else ""))

        # Create model. Seed via pretrain_weights so the weights survive into
        # RFDETRModelModule, which rebuilds the network from model_config inside
        # train() and loads only model_config.pretrain_weights (a load_state_dict on
        # the wrapper here would be discarded). load_pretrain_weights aligns
        # num_classes from the checkpoint/dataset.
        if len(self._seed_model) > 0 and ub.Path(self._seed_model).exists():
            model = RFDETRModel(pretrain_weights=self._seed_model, **model_kwargs)
        else:
            # Use pretrained weights
            model = RFDETRModel(**model_kwargs)

        # Parse training parameters
        epochs = int(self._max_epochs)
        # "auto" lets RF-DETR probe for the largest micro-batch that fits VRAM.
        if str(self._batch_size).strip().lower() == 'auto':
            batch_size = 'auto'
        else:
            batch_size = int(self._batch_size)
        lr = float(self._learning_rate)
        lr_encoder = float(self._learning_rate_encoder)
        grad_accum_steps = int(self._grad_accum_steps)
        weight_decay = float(self._weight_decay)
        warmup_epochs = float(self._warmup_epochs)
        lr_drop = int(self._lr_drop)
        lr_scheduler = str(self._lr_scheduler).strip().lower()
        use_ema = parse_bool(self._use_ema)
        ema_decay = float(self._ema_decay)
        early_stopping = parse_bool(self._early_stopping)
        early_stopping_patience = int(self._early_stopping_patience)
        multi_scale = parse_bool(self._multi_scale)
        checkpoint_interval = int(self._checkpoint_interval)
        use_tensorboard = parse_bool(self._use_tensorboard)
        aug_config = self._resolve_aug_config(self._augmentation)

        # Parse timeout (in seconds, or "default" for no limit)
        import time
        timeout_str = str(self._timeout).lower()
        if timeout_str == "default" or timeout_str == "none" or timeout_str == "":
            timeout_seconds = None
        else:
            timeout_seconds = float(self._timeout)

        output_dir = ub.Path(self._train_directory) / "rf_detr_output"
        output_dir.ensuredir()

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
            lr_scheduler=lr_scheduler,
            use_ema=use_ema,
            ema_decay=ema_decay,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            multi_scale=multi_scale,
            checkpoint_interval=checkpoint_interval,
            eval_interval=int(self._eval_interval),
            eval_max_dets=int(self._eval_max_dets),
            tensorboard=use_tensorboard,
            wandb=False,
        )
        # Omit aug_config when None so RF-DETR applies its own default preset.
        if aug_config is not None:
            train_kwargs["aug_config"] = aug_config
        # When probing for the batch size, target this per-device effective batch
        # (micro-batch * grad_accum); RF-DETR derives grad_accum_steps itself.
        if batch_size == 'auto':
            train_kwargs["auto_batch_target_effective"] = int(self._auto_batch_target_effective)
        train_kwargs.update(self._dataloader_kwargs())

        # RF-DETR (>=1.7.0) trains via PyTorch Lightning; graceful stop now goes
        # through Trainer.should_stop. Inject a callback that flips it on timeout
        # or interrupt, wrapping build_trainer since train() exposes no other seam.
        import pytorch_lightning as pl
        import rfdetr.training as rfdetr_training

        class _StopControlCallback(pl.Callback):
            def __init__(self, timeout_seconds=None):
                self._timeout_seconds = timeout_seconds
                self._start_time = None
                self._stop_requested = False

            def request_stop(self):
                self._stop_requested = True

            def on_train_start(self, trainer, *args, **kwargs):
                self._start_time = time.monotonic()

            def _maybe_stop(self, trainer):
                if self._stop_requested:
                    trainer.should_stop = True
                    return
                if (self._timeout_seconds is not None and self._start_time is not None
                        and time.monotonic() - self._start_time >= self._timeout_seconds):
                    print(f"[RFDETRTrainer] Timeout reached ({self._timeout_seconds:.0f}s); stopping")
                    trainer.should_stop = True

            def on_train_batch_end(self, trainer, *args, **kwargs):
                self._maybe_stop(trainer)

            def on_train_epoch_end(self, trainer, *args, **kwargs):
                self._maybe_stop(trainer)

        stop_control = _StopControlCallback(timeout_seconds)
        original_build_trainer = rfdetr_training.build_trainer

        def _build_trainer_with_stop_control(*args, **kwargs):
            trainer = original_build_trainer(*args, **kwargs)
            trainer.callbacks.append(stop_control)
            return trainer

        rfdetr_training.build_trainer = _build_trainer_with_stop_control

        with TrainingInterruptHandler("RFDETRTrainer", on_interrupt=stop_control.request_stop) as handler:
            try:
                model.train(**train_kwargs)
            except KeyboardInterrupt:
                print("[RFDETRTrainer] Training interrupted by user")
            finally:
                rfdetr_training.build_trainer = original_build_trainer

            self._interrupted = handler.interrupted

        output = self._get_output_map(output_dir)

        print("\n[RFDETRTrainer] Model training complete!")

        return output

    def _resolve_gpu_count(self, device):
        """Number of GPUs to train on. 'auto' uses all visible GPUs; a pinned
        cuda:N device or a CPU device forces a single-process run."""
        requested = str(self._devices).strip().lower()
        if requested not in ('auto', ''):
            try:
                return max(0, int(requested))
            except ValueError:
                pass
        if str(device).startswith('cpu'):
            return 0
        if ':' in str(device):
            return 1
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 1

    def _resolve_strategy(self, n_gpus):
        strat = str(self._strategy).strip().lower()
        if strat not in ('auto', ''):
            return self._strategy
        return 'ddp_find_unused_parameters_true'

    def _dataloader_kwargs(self):
        """Accuracy-neutral DataLoader tuning passed to model.train(). On
        Windows, worker subprocesses fail (spawn re-invokes viame.exe), so force
        0 workers there. persistent_workers/prefetch_factor only apply when
        num_workers > 0."""
        num_workers = 0 if sys.platform == "win32" else int(self._num_workers)
        kw = dict(num_workers=num_workers, pin_memory=parse_bool(self._pin_memory))
        if num_workers > 0:
            kw["persistent_workers"] = parse_bool(self._persistent_workers)
            kw["prefetch_factor"] = int(self._prefetch_factor)
        return kw

    def _train_multi_gpu(self, dataset_dir, n_gpus):
        """Launch DDP training in a subprocess across all GPUs. PTL re-execs the
        entrypoint once per rank; this process waits, then packages the result."""
        output_dir = ub.Path(self._train_directory) / "rf_detr_output"
        output_dir.ensuredir()

        if str(self._batch_size).strip().lower() == 'auto':
            batch_size = 'auto'
        else:
            batch_size = int(self._batch_size)

        train_kwargs = dict(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            epochs=int(self._max_epochs),
            batch_size=batch_size,
            lr=float(self._learning_rate),
            lr_encoder=float(self._learning_rate_encoder),
            grad_accum_steps=int(self._grad_accum_steps),
            weight_decay=float(self._weight_decay),
            warmup_epochs=float(self._warmup_epochs),
            lr_drop=int(self._lr_drop),
            lr_scheduler=str(self._lr_scheduler).strip().lower(),
            use_ema=parse_bool(self._use_ema),
            ema_decay=float(self._ema_decay),
            early_stopping=parse_bool(self._early_stopping),
            early_stopping_patience=int(self._early_stopping_patience),
            multi_scale=parse_bool(self._multi_scale),
            checkpoint_interval=int(self._checkpoint_interval),
            eval_interval=int(self._eval_interval),
            eval_max_dets=int(self._eval_max_dets),
            tensorboard=parse_bool(self._use_tensorboard),
            wandb=False,
            class_names=list(self._class_names),
            devices=n_gpus,
            strategy=self._resolve_strategy(n_gpus),
        )
        if batch_size == 'auto':
            train_kwargs["auto_batch_target_effective"] = \
                int(self._auto_batch_target_effective)
        aug_config = self._resolve_aug_config(self._augmentation)
        if aug_config is not None:
            train_kwargs["aug_config"] = aug_config
        train_kwargs.update(self._dataloader_kwargs())

        params = dict(
            model_size=self._model_size.lower(),
            segmentation=parse_bool(self._segmentation),
            num_channels=int(self._num_channels),
            resolution=int(self._resolution),
            gradient_checkpointing=parse_bool(self._gradient_checkpointing),
            seed_model=self._seed_model,
            class_names=list(self._class_names),
            train_kwargs=train_kwargs,
        )

        params_path = os.path.join(self._train_directory,
                                   "rf_detr_mgpu_params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f)

        # Pick an interpreter matching this one's version so VIAME's
        # version-specific extension modules (torch/torchvision/rfdetr) import
        # in the subprocess. A bare "python" on PATH can resolve to an unrelated
        # interpreter (e.g. a base conda env) that lacks the VIAME packages.
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        python = (shutil.which(py_ver)
                  or (sys.executable
                      if os.path.basename(sys.executable or "").startswith("python")
                      else None)
                  or shutil.which("python3")
                  or shutil.which("python"))
        impl = os.path.join(os.path.dirname(__file__), "rf_detr_launcher.py")

        # The embedded VIAME interpreter resolves its packages via sys.path
        # entries that are not all present in the inherited PYTHONPATH env var,
        # so a plain subprocess can find a partial/!=torchvision. Propagate this
        # process's sys.path AND the site dirs of the packages it actually
        # imported (torch/torchvision/rfdetr) so the DDP subprocess (and PTL's
        # per-rank re-execs) import exactly what this process does.
        env = dict(os.environ)
        extra_paths = list(sys.path)
        for mod_name in ("torch", "torchvision", "rfdetr", "viame"):
            try:
                mod = __import__(mod_name)
                extra_paths.append(
                    os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__))))
            except Exception:
                pass
        existing_pp = env.get("PYTHONPATH", "")
        if existing_pp:
            extra_paths.append(existing_pp)
        # De-dup while preserving order.
        seen = set()
        ordered = []
        for p in extra_paths:
            if p and p not in seen:
                seen.add(p)
                ordered.append(p)
        env["PYTHONPATH"] = os.pathsep.join(ordered)

        print(f"[RFDETRTrainer] Launching {n_gpus}-GPU DDP training: "
              f"{python} {impl}", flush=True)
        try:
            subprocess.run([python, impl, params_path], check=True, env=env)
        except subprocess.CalledProcessError as exc:
            # A rank killed by SIGKILL is the kernel/cgroup OOM killer reclaiming
            # host RAM -- it raises no Python exception and no CUDA OOM, so the
            # bare CalledProcessError gives no hint of the actual cause. Every
            # rank forks num_workers DataLoader workers, so host memory scales as
            # n_gpus * num_workers * prefetch_factor * batch_size.
            if exc.returncode == -9:
                n_workers = int(self._num_workers)
                raise RuntimeError(
                    f"A DDP rank was killed with SIGKILL, which means the host "
                    f"(not the GPU) ran out of memory. This run had {n_gpus} "
                    f"ranks x {n_workers} DataLoader workers = "
                    f"{n_gpus * n_workers} worker processes, each prefetching "
                    f"{int(self._prefetch_factor)} x {self._batch_size} images. "
                    f"Ask the scheduler for more memory (Slurm: --mem and "
                    f"--cpus-per-task), or lower num_workers / prefetch_factor / "
                    f"val_subsample."
                ) from exc
            raise

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
            # rfdetr may pre-seed args['num_classes']=None, so check the value,
            # not just key presence, or the deployed model gets num_classes=None.
            if not args.get('num_classes'):
                args['num_classes'] = len(self._class_names)
            args['class_names'] = self._class_names
            args['model_size'] = self._model_size
            args['segmentation'] = parse_bool(self._segmentation)
            args['num_channels'] = int(self._num_channels)
            if int(self._resolution) > 0:
                args['resolution'] = int(self._resolution)
            checkpoint['args'] = args
            torch.save(checkpoint, final_ckpt)
            print(f"[RFDETRTrainer] Embedded {len(self._class_names)} class names into checkpoint")

        algo = "rf_detr"

        output["type"] = algo

        # Config key matching rf_detr detector inference config
        output[algo + ":weight"] = output_model_name

        # Record the architecture in the generated pipeline so inference builds
        # the right model directly, without having to recover it from the
        # checkpoint weights. model_size and segmentation select the model class;
        # num_channels and resolution size the inputs and positional embeddings.
        output[algo + ":model_size"] = str(self._model_size)
        output[algo + ":num_channels"] = str(int(self._num_channels))
        output[algo + ":segmentation"] = str(parse_bool(self._segmentation))
        if int(self._resolution) > 0:
            output[algo + ":resolution"] = str(int(self._resolution))

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
