# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import TrainDetector

import os

import scriptconfig as scfg
import ubelt as ub
import yaml

from viame.pytorch.utilities import (
    vital_config_update,
    resolve_device,
    parse_bool,
    register_vital_algorithm,
    TrainingInterruptHandler,
)


class UltralyticsTrainerConfig(scfg.DataConfig):
    """
    The configuration for :class:`UltralyticsTrainer`.
    """
    identifier = "viame-ultralytics-detector"
    train_directory = "deep_training"
    seed_model = ""

    # Model configuration
    model_type = scfg.Value('yolov8n.pt', help=ub.paragraph('''
        Model type or path to pretrained weights. Options include:
        yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (detection)
        yolov8n-seg.pt, yolov8s-seg.pt, etc. (segmentation)
        yolov9c.pt, yolov9e.pt (YOLOv9)
        yolov10n.pt, yolov10s.pt, yolov10m.pt, yolov10l.pt, yolov10x.pt (YOLOv10)
        yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt (YOLO11)
        Or path to a custom .pt file.
        '''))
    device = scfg.Value('auto', help='Device to train on: auto, cpu, cuda, or cuda:N')

    # Training hyperparameters
    max_epochs = scfg.Value(100, help='Maximum number of epochs to train for')
    batch_size = scfg.Value(16, help='Number of images per batch (-1 for auto)')
    image_size = scfg.Value(640, help='Input image size')
    learning_rate = scfg.Value(0.01, help='Initial learning rate (lr0)')
    learning_rate_final = scfg.Value(0.01, help='Final learning rate factor (lrf)')
    momentum = scfg.Value(0.937, help='SGD momentum / Adam beta1')
    weight_decay = scfg.Value(0.0005, help='Weight decay')
    warmup_epochs = scfg.Value(3.0, help='Number of warmup epochs')
    warmup_momentum = scfg.Value(0.8, help='Warmup initial momentum')

    # Optimizer
    optimizer = scfg.Value('auto', help='Optimizer: auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp')

    # Data augmentation
    hsv_h = scfg.Value(0.015, help='HSV-Hue augmentation')
    hsv_s = scfg.Value(0.7, help='HSV-Saturation augmentation')
    hsv_v = scfg.Value(0.4, help='HSV-Value augmentation')
    degrees = scfg.Value(0.0, help='Rotation augmentation (+/- degrees)')
    translate = scfg.Value(0.1, help='Translation augmentation (+/- fraction)')
    scale = scfg.Value(0.5, help='Scale augmentation (+/- gain)')
    shear = scfg.Value(0.0, help='Shear augmentation (+/- degrees)')
    perspective = scfg.Value(0.0, help='Perspective augmentation (+/- fraction)')
    flipud = scfg.Value(0.0, help='Vertical flip probability')
    fliplr = scfg.Value(0.5, help='Horizontal flip probability')
    mosaic = scfg.Value(1.0, help='Mosaic augmentation probability')
    mixup = scfg.Value(0.0, help='Mixup augmentation probability')
    copy_paste = scfg.Value(0.0, help='Copy-paste augmentation probability')

    # Other training options
    patience = scfg.Value(100, help='Early stopping patience (0 to disable)')
    workers = scfg.Value(8, help='Number of data loader workers')
    cos_lr = scfg.Value(False, help='Use cosine learning rate scheduler')
    close_mosaic = scfg.Value(10, help='Disable mosaic for final N epochs')
    amp = scfg.Value(True, help='Use automatic mixed precision')
    fraction = scfg.Value(1.0, help='Fraction of dataset to use')
    freeze = scfg.Value(None, help='Number of layers to freeze (None for no freeze)')
    multi_scale = scfg.Value(False, help='Use multi-scale training')
    overlap_mask = scfg.Value(True, help='Overlap masks during training (segmentation)')
    mask_ratio = scfg.Value(4, help='Mask downsample ratio (segmentation)')
    dropout = scfg.Value(0.0, help='Dropout rate for classification')

    # Validation
    val_period = scfg.Value(1, help='Validation every N epochs')

    # Checkpointing
    save_period = scfg.Value(-1, help='Save checkpoint every N epochs (-1 to disable)')

    # Resume training
    resume = scfg.Value(False, help='Resume training from last checkpoint')

    pipeline_template = ""

    def __post_init__(self):
        super().__post_init__()


class UltralyticsLabelMapper:
    """
    Manages explicit image-to-label path mapping, bypassing Ultralytics'
    default /images/ -> /labels/ path convention.

    This allows training on images at arbitrary paths without copying or
    symlinking them into a specific directory structure.
    """

    def __init__(self):
        self._mapping = {}
        self._original_func = None
        self._is_patched = False

    def add_mapping(self, image_path, label_path):
        """Add a mapping from image path to label path."""
        # Normalize paths for consistent lookup
        self._mapping[str(ub.Path(image_path).resolve())] = str(label_path)

    def clear(self):
        """Clear all mappings."""
        self._mapping.clear()

    def patch_ultralytics(self):
        """
        Monkey-patch Ultralytics' img2label_paths function to use our mapping.
        """
        if self._is_patched:
            return

        import ultralytics.data.utils as ul_utils

        self._original_func = ul_utils.img2label_paths
        mapping = self._mapping

        def custom_img2label_paths(img_paths):
            """Custom label path resolver using explicit mapping."""
            result = []
            for p in img_paths:
                p_resolved = str(ub.Path(p).resolve())
                if p_resolved in mapping:
                    result.append(mapping[p_resolved])
                else:
                    # Fall back to original behavior for unmapped paths
                    sa = f'{os.sep}images{os.sep}'
                    sb = f'{os.sep}labels{os.sep}'
                    if sa in str(p):
                        result.append(sb.join(str(p).rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt')
                    else:
                        # Last resort: just change extension
                        result.append(str(p).rsplit('.', 1)[0] + '.txt')
            return result

        ul_utils.img2label_paths = custom_img2label_paths
        self._is_patched = True

    def restore_ultralytics(self):
        """Restore original Ultralytics behavior."""
        if not self._is_patched or self._original_func is None:
            return

        import ultralytics.data.utils as ul_utils
        ul_utils.img2label_paths = self._original_func
        self._is_patched = False


class UltralyticsTrainer(TrainDetector):
    """
    Implementation of TrainDetector for Ultralytics YOLO models.

    Directly converts KWIVER DetectedObjectSet to YOLO format without
    copying or symlinking images. Uses a custom label mapping to bypass
    Ultralytics' default directory structure requirements.

    Supports YOLOv8, YOLOv9, YOLOv10, and YOLO11 models for detection,
    segmentation, and classification tasks.
    """
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = UltralyticsTrainerConfig()
        self._categories = []
        self._category_to_idx = {}
        self._data_yaml_path = None
        self._label_mapper = UltralyticsLabelMapper()

    def get_configuration(self):
        cfg = super(TrainDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        print('[UltralyticsTrainer] set_configuration')
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))
        self._config.__post_init__()

        # Set attributes with underscore prefix for easy access
        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._setup_directories()
        return True

    def _setup_directories(self):
        """Create necessary directories for training."""
        if self._train_directory:
            os.makedirs(self._train_directory, exist_ok=True)

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(self, categories, train_files, train_dets, test_files, test_dets):
        """
        Receive training data from KWIVER and write directly to YOLO format.

        Images are referenced by their original paths (no copying/symlinking).
        Labels are written to a local directory with explicit path mapping.

        Args:
            categories: CategoryTree with class names and IDs
            train_files: List of training image file paths
            train_dets: List of DetectedObjectSet for each training image
            test_files: List of validation image file paths
            test_dets: List of DetectedObjectSet for each validation image
        """
        print('[UltralyticsTrainer] add_data_from_disk')

        if len(train_files) != len(train_dets):
            print("Error: train file and groundtruth count mismatch")
            return

        # Build category mapping
        if categories is not None:
            self._categories = categories.all_class_names()
        self._category_to_idx = {name: idx for idx, name in enumerate(self._categories)}

        # Create dataset directory for labels and metadata
        dataset_dir = ub.Path(self._train_directory) / "yolo_dataset"
        labels_dir = dataset_dir / "labels"
        labels_dir.ensuredir()

        # Clear any previous mappings
        self._label_mapper.clear()

        # Process training data
        train_image_list = self._write_yolo_labels(
            train_files, train_dets, categories,
            labels_dir, "train"
        )
        print(f"[UltralyticsTrainer] Processed {len(train_image_list)} training samples")

        # Process validation data
        val_image_list = self._write_yolo_labels(
            test_files, test_dets, categories,
            labels_dir, "val"
        )
        print(f"[UltralyticsTrainer] Processed {len(val_image_list)} validation samples")

        # Write image list files (absolute paths to original images)
        train_list_file = dataset_dir / "train.txt"
        val_list_file = dataset_dir / "val.txt"

        with open(train_list_file, 'w') as f:
            f.write('\n'.join(train_image_list))

        with open(val_list_file, 'w') as f:
            f.write('\n'.join(val_image_list))

        # Create data.yaml
        self._data_yaml_path = self._write_data_yaml(
            dataset_dir, train_list_file, val_list_file
        )

    def _write_yolo_labels(self, image_files, detection_sets, categories, labels_dir, split_name):
        """
        Write YOLO format labels and register image-to-label mappings.

        Args:
            image_files: List of image file paths
            detection_sets: List of DetectedObjectSet
            categories: CategoryTree for class name lookup
            labels_dir: Directory to write label files
            split_name: Name of split (train/val) for organizing labels

        Returns:
            list: List of absolute image paths that were successfully processed
        """
        from PIL import Image

        split_labels_dir = labels_dir / split_name
        split_labels_dir.ensuredir()

        processed_images = []

        for idx, (img_path, detections) in enumerate(zip(image_files, detection_sets)):
            img_path = ub.Path(img_path)

            if not img_path.exists():
                print(f"[UltralyticsTrainer] Warning: Image not found: {img_path}")
                continue

            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"[UltralyticsTrainer] Warning: Could not read image {img_path}: {e}")
                continue

            # Create unique label filename using index and original name
            label_filename = f"{idx:08d}_{img_path.stem}.txt"
            label_path = split_labels_dir / label_filename

            # Build label content
            label_lines = []

            for det in detections:
                # Get bounding box
                bbox = det.bounding_box
                x_min = bbox.min_x()
                y_min = bbox.min_y()
                x_max = bbox.max_x()
                y_max = bbox.max_y()

                # Skip invalid boxes
                width = x_max - x_min
                height = y_max - y_min
                if width <= 0 or height <= 0:
                    continue

                # Get class index
                if det.type is None:
                    continue

                class_name = det.type.get_most_likely_class()

                if class_name not in self._category_to_idx:
                    # Add new category if not seen before
                    self._category_to_idx[class_name] = len(self._categories)
                    self._categories.append(class_name)

                class_idx = self._category_to_idx[class_name]

                # Convert to YOLO format: x_center, y_center, width, height (normalized)
                x_center = (x_min + x_max) / 2.0 / img_width
                y_center = (y_min + y_max) / 2.0 / img_height
                w_norm = width / img_width
                h_norm = height / img_height

                # Clamp to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                label_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # Write label file
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))

            # Register the mapping from image path to label path
            abs_img_path = str(img_path.resolve())
            self._label_mapper.add_mapping(abs_img_path, str(label_path))

            processed_images.append(abs_img_path)

        return processed_images

    def _write_data_yaml(self, dataset_dir, train_list_file, val_list_file):
        """
        Write the data.yaml file required by Ultralytics.

        Args:
            dataset_dir: Path to the dataset directory
            train_list_file: Path to text file listing training images
            val_list_file: Path to text file listing validation images

        Returns:
            Path: Path to the created data.yaml file
        """
        # Build names dict with sequential indices
        names = {idx: name for idx, name in enumerate(self._categories)}

        data_yaml = {
            'path': str(dataset_dir.resolve()),
            'train': str(train_list_file.resolve()),
            'val': str(val_list_file.resolve()),
            'names': names,
        }

        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"[UltralyticsTrainer] Created data.yaml with {len(names)} classes: {list(names.values())}")

        return yaml_path

    def update_model(self):
        """Train the Ultralytics YOLO model.

        Returns:
            dict: Map of template replacements and file copies
        """
        import torch
        from ultralytics import YOLO

        if self._data_yaml_path is None:
            print("[UltralyticsTrainer] Error: No training data. Call add_data_from_disk first.")
            return {}

        print("[UltralyticsTrainer] Starting Ultralytics YOLO training")

        # Patch Ultralytics to use our label mapping
        self._label_mapper.patch_ultralytics()

        try:
            return self._run_training()
        finally:
            # Restore original Ultralytics behavior
            self._label_mapper.restore_ultralytics()

    def _run_training(self):
        """Execute the training loop."""
        import torch
        from ultralytics import YOLO

        # Determine device
        device = resolve_device(self._device)
        # Ultralytics accepts int for GPU index
        if hasattr(device, 'index') and device.index is not None:
            device = device.index
        elif str(device) == 'cpu':
            device = 'cpu'

        # Load model
        model_type = self._model_type
        if self._seed_model and ub.Path(self._seed_model).exists():
            print(f"[UltralyticsTrainer] Loading seed model from {self._seed_model}")
            model = YOLO(self._seed_model)
        else:
            print(f"[UltralyticsTrainer] Using pretrained model: {model_type}")
            model = YOLO(model_type)

        # Parse training parameters
        epochs = int(self._max_epochs)
        batch_size = int(self._batch_size)
        image_size = int(self._image_size)
        lr0 = float(self._learning_rate)
        lrf = float(self._learning_rate_final)
        momentum = float(self._momentum)
        weight_decay = float(self._weight_decay)
        warmup_epochs = float(self._warmup_epochs)
        warmup_momentum = float(self._warmup_momentum)
        patience = int(self._patience)
        workers = int(self._workers)
        save_period = int(self._save_period)

        # Parse augmentation parameters
        hsv_h = float(self._hsv_h)
        hsv_s = float(self._hsv_s)
        hsv_v = float(self._hsv_v)
        degrees = float(self._degrees)
        translate = float(self._translate)
        scale = float(self._scale)
        shear = float(self._shear)
        perspective = float(self._perspective)
        flipud = float(self._flipud)
        fliplr = float(self._fliplr)
        mosaic = float(self._mosaic)
        mixup = float(self._mixup)
        copy_paste = float(self._copy_paste)
        close_mosaic = int(self._close_mosaic)

        # Parse other options
        optimizer = self._optimizer
        cos_lr = parse_bool(self._cos_lr)
        amp = parse_bool(self._amp)
        fraction = float(self._fraction)
        multi_scale = parse_bool(self._multi_scale)
        overlap_mask = parse_bool(self._overlap_mask)
        mask_ratio = int(self._mask_ratio)
        dropout = float(self._dropout)
        resume = parse_bool(self._resume)

        freeze = self._freeze
        if freeze and freeze != 'None' and freeze != '':
            freeze = int(freeze)
        else:
            freeze = None

        # Output directory
        project_dir = ub.Path(self._train_directory) / "yolo_runs"
        run_name = self._identifier

        # Signal handler for graceful interruption
        with TrainingInterruptHandler("UltralyticsTrainer") as handler:
            try:
                model.train(
                    data=str(self._data_yaml_path),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=image_size,
                    device=device,
                    project=str(project_dir),
                    name=run_name,
                    exist_ok=True,
                    pretrained=True,
                    optimizer=optimizer,
                    lr0=lr0,
                    lrf=lrf,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    warmup_epochs=warmup_epochs,
                    warmup_momentum=warmup_momentum,
                    patience=patience,
                    workers=workers,
                    save_period=save_period,
                    # Augmentation
                    hsv_h=hsv_h,
                    hsv_s=hsv_s,
                    hsv_v=hsv_v,
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    perspective=perspective,
                    flipud=flipud,
                    fliplr=fliplr,
                    mosaic=mosaic,
                    mixup=mixup,
                    copy_paste=copy_paste,
                    close_mosaic=close_mosaic,
                    # Other options
                    cos_lr=cos_lr,
                    amp=amp,
                    fraction=fraction,
                    multi_scale=multi_scale,
                    overlap_mask=overlap_mask,
                    mask_ratio=mask_ratio,
                    dropout=dropout,
                    freeze=freeze,
                    resume=resume,
                    verbose=True,
                )
            except KeyboardInterrupt:
                print("[UltralyticsTrainer] Training interrupted by user")

            self._interrupted = handler.interrupted

        output = self._get_output_map(project_dir / run_name)
        print("\n[UltralyticsTrainer] Model training complete!\n")
        return output

    def _get_output_map(self, train_output_dir):
        """Build output map with template replacements and file copies.

        Returns:
            dict: Map where keys with '[-' and '-]' are template replacements,
                  other keys are file copies (key=output filename, value=source path)
        """
        output = {}
        output_model_name = "trained_ultralytics_checkpoint.pt"

        if train_output_dir:
            train_output_dir = ub.Path(train_output_dir)
            weights_dir = train_output_dir / "weights"

            # Prefer best.pt, fall back to last.pt
            best_ckpt = weights_dir / "best.pt"
            last_ckpt = weights_dir / "last.pt"

            if best_ckpt.exists():
                final_ckpt = best_ckpt
            elif last_ckpt.exists():
                final_ckpt = last_ckpt
            else:
                pt_files = sorted(weights_dir.glob("*.pt")) if weights_dir.exists() else []
                if pt_files:
                    final_ckpt = pt_files[-1]
                else:
                    print("[UltralyticsTrainer] No checkpoint found")
                    return output

            algo = "ultralytics"

            output["type"] = algo

            # Config keys matching ultralytics_detector inference config
            output[algo + ":weight"] = output_model_name

            # File copies (key=output filename, value=source path)
            output[output_model_name] = str(final_ckpt)

            print(f"[UltralyticsTrainer] Model found at {final_ckpt}")

        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        UltralyticsTrainer, "ultralytics", "PyTorch Ultralytics YOLO training routine"
    )
