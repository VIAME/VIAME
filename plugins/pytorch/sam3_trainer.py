# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3 (Segment Anything Model 3 / 3.1) fine-tuning trainers.

This module provides two trainers:

1. SAM3Trainer (train_detector type, "sam3" implementation) - fine-tunes
   SAM3 for per-frame segmentation using detection-level annotations.
   Annotations are converted to COCO format with polygon masks (or box
   fills) used for segmentation supervision.

2. SAM3TrackerTrainer (train_tracker type, "sam3" implementation) -
   fine-tunes SAM3 mask propagation for video object tracking using
   track-level annotations. Ground-truth tracks are sampled into short
   clips; the first frame of a clip is prompted with the ground-truth
   box while subsequent frames are prompted only with the previous
   frame's predicted low-resolution mask, mimicking inference-time mask
   propagation. All frames are supervised with BCE + Dice losses. The
   resulting checkpoint is emitted with type "sam3_tracker" so it can be
   consumed by the sam3_tracker TrackObjects implementation.

Both support freezing the image/prompt encoders (default) so only the
mask decoder is fine-tuned, plus mixed precision training.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from kwiver.vital.algo import TrainDetector, TrainTracker

from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)

from viame.pytorch.utilities import report_cuda_errors


def _resolve_attr(model, names):
    """Return the first present attribute from names, or None."""
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


class SAM3Trainer(TrainDetector):
    """
    Implementation of TrainDetector class for SAM3 fine-tuning.

    Fine-tunes SAM3 segmentation models using detection annotations with
    optional polygon masks. Supports both detection and tracking datasets.
    """

    def __init__(self):
        TrainDetector.__init__(self)

        # Model configuration
        self._identifier = "viame-sam3-segmentation"
        self._sam_model_id = "facebook/sam2.1-hiera-large"
        self._train_directory = "deep_training"

        # Training configuration
        self._gpu_count = -1
        self._max_epochs = "50"
        self._batch_size = "4"
        self._learning_rate = "1e-5"
        self._timeout = "604800"
        self._chip_width = "1024"
        self._chip_height = "1024"

        # Fine-tuning configuration
        self._freeze_image_encoder = "true"
        self._freeze_prompt_encoder = "true"
        self._freeze_memory_encoder = "false"
        self._amp_enabled = "true"

        # Data configuration
        self._threshold = "0.00"
        self._augmentation = "standard"

        # Internal state
        self._categories = []
        self._train_image_files = []
        self._train_detections = []
        self._test_image_files = []
        self._test_detections = []

    def get_configuration(self):
        cfg = super(TrainDetector, self).get_configuration()

        cfg.set_value("identifier", self._identifier)
        cfg.set_value("sam_model_id", self._sam_model_id)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("gpu_count", str(self._gpu_count))
        cfg.set_value("max_epochs", self._max_epochs)
        cfg.set_value("batch_size", self._batch_size)
        cfg.set_value("learning_rate", self._learning_rate)
        cfg.set_value("timeout", self._timeout)
        cfg.set_value("chip_width", self._chip_width)
        cfg.set_value("chip_height", self._chip_height)
        cfg.set_value("freeze_image_encoder", self._freeze_image_encoder)
        cfg.set_value("freeze_prompt_encoder", self._freeze_prompt_encoder)
        cfg.set_value("freeze_memory_encoder", self._freeze_memory_encoder)
        cfg.set_value("amp_enabled", self._amp_enabled)
        cfg.set_value("threshold", self._threshold)
        cfg.set_value("augmentation", self._augmentation)

        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._sam_model_id = str(cfg.get_value("sam_model_id"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._gpu_count = int(cfg.get_value("gpu_count"))
        self._max_epochs = str(cfg.get_value("max_epochs"))
        self._batch_size = str(cfg.get_value("batch_size"))
        self._learning_rate = str(cfg.get_value("learning_rate"))
        self._timeout = str(cfg.get_value("timeout"))
        self._chip_width = str(cfg.get_value("chip_width"))
        self._chip_height = str(cfg.get_value("chip_height"))
        self._freeze_image_encoder = str(cfg.get_value("freeze_image_encoder"))
        self._freeze_prompt_encoder = str(cfg.get_value("freeze_prompt_encoder"))
        self._freeze_memory_encoder = str(cfg.get_value("freeze_memory_encoder"))
        self._amp_enabled = str(cfg.get_value("amp_enabled"))
        self._threshold = str(cfg.get_value("threshold"))
        self._augmentation = str(cfg.get_value("augmentation"))

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                if self._gpu_count < 0:
                    self._gpu_count = torch.cuda.device_count()
        except ImportError:
            print("PyTorch not available, defaulting to 1 GPU")
            if self._gpu_count < 0:
                self._gpu_count = 1

        if self._train_directory:
            if not os.path.exists(self._train_directory):
                os.makedirs(self._train_directory)

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or \
           len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(self, categories, train_files, train_dets,
                           test_files, test_dets):
        print("Adding training data from disk...")
        print("  Training files: ", len(train_files))
        print("  Training detections: ", len(train_dets))
        print("  Test files: ", len(test_files))
        print("  Test detections: ", len(test_dets))

        if categories is not None:
            self._categories = categories.all_class_names()
        else:
            self._categories = []

        self._train_image_files = list(train_files)
        self._train_detections = list(train_dets)
        self._test_image_files = list(test_files)
        self._test_detections = list(test_dets)

    def _prepare_coco_dataset(self):
        """
        Convert VIAME annotations to COCO format for SAM3 training.

        Creates:
        - sam3_data/train/images/ - Training images (symlinks)
        - sam3_data/train/annotations.json - COCO format annotations
        - sam3_data/val/images/ - Validation images (symlinks)
        - sam3_data/val/annotations.json - COCO format annotations
        """
        sam3_dir = Path(self._train_directory) / "sam3_data"
        if sam3_dir.exists():
            shutil.rmtree(sam3_dir)

        train_dir = sam3_dir / "train"
        val_dir = sam3_dir / "val"
        (train_dir / "images").mkdir(parents=True)
        (val_dir / "images").mkdir(parents=True)

        print("Preparing SAM3 COCO-format dataset...")

        # Build category mapping
        category_map = {}
        categories_json = []
        for i, cat in enumerate(self._categories):
            cat_id = i + 1
            category_map[cat] = cat_id
            categories_json.append({
                "id": cat_id,
                "name": cat,
                "supercategory": "object"
            })

        # If no categories, use generic
        if len(categories_json) == 0:
            categories_json.append({
                "id": 1,
                "name": "object",
                "supercategory": "object"
            })

        # Process training data
        train_images, train_annotations = self._process_split(
            self._train_image_files, self._train_detections,
            train_dir / "images", category_map, "train"
        )

        # Process validation data
        val_images, val_annotations = self._process_split(
            self._test_image_files, self._test_detections,
            val_dir / "images", category_map, "val"
        )

        # Write COCO JSON files
        train_coco = {
            "images": train_images,
            "annotations": train_annotations,
            "categories": categories_json
        }
        with open(train_dir / "annotations.json", 'w') as f:
            json.dump(train_coco, f)

        val_coco = {
            "images": val_images,
            "annotations": val_annotations,
            "categories": categories_json
        }
        with open(val_dir / "annotations.json", 'w') as f:
            json.dump(val_coco, f)

        print(f"  Train: {len(train_images)} images, {len(train_annotations)} annotations")
        print(f"  Val: {len(val_images)} images, {len(val_annotations)} annotations")

        return sam3_dir

    def _process_split(self, image_files, detection_sets, output_dir,
                       category_map, split_name):
        """Process one split (train/val) of the data."""
        import cv2

        images_json = []
        annotations_json = []
        annotation_id = 1

        for img_idx, (img_path, det_set) in enumerate(zip(image_files, detection_sets)):
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image: {img_path}")
                continue

            height, width = img.shape[:2]
            img_id = img_idx + 1

            # Create symlink to image
            img_name = f"{img_id:08d}{Path(img_path).suffix}"
            link_path = output_dir / img_name
            if not link_path.exists():
                try:
                    os.symlink(os.path.abspath(img_path), link_path)
                except OSError:
                    # Fall back to copy if symlink fails
                    shutil.copy(img_path, link_path)

            images_json.append({
                "id": img_id,
                "file_name": img_name,
                "width": width,
                "height": height
            })

            # Process detections
            if det_set is None:
                continue

            for det in det_set:
                bbox = det.bounding_box
                x1, y1 = bbox.min_x(), bbox.min_y()
                x2, y2 = bbox.max_x(), bbox.max_y()
                w, h = x2 - x1, y2 - y1

                if w <= 0 or h <= 0:
                    continue

                # Get category
                det_type = det.type
                if det_type is not None:
                    class_name = det_type.get_most_likely_class()
                    if class_name in category_map:
                        cat_id = category_map[class_name]
                    else:
                        cat_id = 1
                else:
                    cat_id = 1

                # Build annotation
                ann = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                }

                # Check for polygon annotation
                polygon = det.polygon()
                if polygon is not None and polygon.num_vertices() >= 3:
                    # Convert polygon to COCO segmentation format
                    segmentation = []
                    for i in range(polygon.num_vertices()):
                        pt = polygon.at(i)
                        segmentation.extend([pt.x(), pt.y()])
                    ann["segmentation"] = [segmentation]
                else:
                    # Create box-based segmentation
                    ann["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]

                annotations_json.append(ann)
                annotation_id += 1

        return images_json, annotations_json

    def _generate_training_config(self, sam3_dir):
        """Generate Hydra-style YAML config for SAM3 training."""
        config_dir = Path(self._train_directory) / "config"
        config_dir.mkdir(exist_ok=True)

        chip_width = int(self._chip_width)
        chip_height = int(self._chip_height)
        batch_size = int(self._batch_size)
        max_epochs = int(self._max_epochs)
        lr = float(self._learning_rate)

        freeze_image = self._freeze_image_encoder.lower() == "true"
        freeze_prompt = self._freeze_prompt_encoder.lower() == "true"
        freeze_memory = self._freeze_memory_encoder.lower() == "true"
        amp_enabled = self._amp_enabled.lower() == "true"

        config = {
            "paths": {
                "dataset_root": str(sam3_dir),
                "train_annotation": str(sam3_dir / "train" / "annotations.json"),
                "train_images": str(sam3_dir / "train" / "images"),
                "val_annotation": str(sam3_dir / "val" / "annotations.json"),
                "val_images": str(sam3_dir / "val" / "images"),
                "checkpoint_dir": str(Path(self._train_directory) / "checkpoints"),
                "log_dir": str(Path(self._train_directory) / "logs"),
            },
            "model": {
                "sam_model_id": self._sam_model_id,
                "freeze_image_encoder": freeze_image,
                "freeze_prompt_encoder": freeze_prompt,
                "freeze_memory_encoder": freeze_memory,
            },
            "training": {
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": 0.01,
                "warmup_epochs": 2,
                "image_size": [chip_height, chip_width],
                "amp_enabled": amp_enabled,
                "gradient_clip": 1.0,
            },
            "data": {
                "num_workers": 4,
                "max_annotations_per_image": 64,
            },
            "distributed": {
                "gpu_count": self._gpu_count,
            }
        }

        config_path = config_dir / "sam3_finetune.yaml"

        # Write as YAML
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def _train_sam3_model(self, sam3_dir, config_path):
        """
        Train SAM3 model using the prepared dataset.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
            import torchvision.transforms as transforms
        except ImportError as e:
            print(f"PyTorch not available: {e}")
            return

        print("Starting SAM3 fine-tuning...")

        # Load config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Initialize SAM3 model
        try:
            self._train_with_sam3_native(config, device)
        except ImportError:
            print("SAM3 native training not available, using simplified training")
            self._train_simplified(config, device)

    def _train_with_sam3_native(self, config, device):
        """Train using SAM3's native training infrastructure."""
        from sam3.train.trainer import Trainer
        from sam3.model_builder import build_sam3

        print("Using SAM3 native training infrastructure")

        # This would require setting up the full Hydra config
        # For now, we'll use a simplified approach
        raise ImportError("Use simplified training")

    def _train_simplified(self, config, device):
        """Simplified training loop for SAM3 fine-tuning."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        import json

        print("Using simplified SAM3 fine-tuning")

        # Load SAM3 model
        try:
            from sam3.model_builder import build_sam3
            model = build_sam3(config["model"]["sam_model_id"])
        except ImportError:
            try:
                from sam2.build_sam import build_sam2
                model = build_sam2(
                    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                    ckpt_path=config["model"]["sam_model_id"],
                    device=str(device),
                    mode='train',
                )
            except ImportError:
                print("SAM3/SAM2 not available. Using HuggingFace fallback.")
                from transformers import Sam2Model
                model = Sam2Model.from_pretrained(config["model"]["sam_model_id"])

        model = model.to(device)

        # Freeze specified components
        if config["model"]["freeze_image_encoder"]:
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            print("  Image encoder frozen")

        if config["model"]["freeze_prompt_encoder"]:
            for param in model.prompt_encoder.parameters():
                param.requires_grad = False
            print("  Prompt encoder frozen")

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,}")

        # Create simple dataset
        class SimpleSAM3Dataset(Dataset):
            def __init__(self, ann_path, img_dir, img_size):
                with open(ann_path, 'r') as f:
                    coco = json.load(f)

                self.img_dir = Path(img_dir)
                self.img_size = img_size
                self.images = {img['id']: img for img in coco['images']}
                self.annotations = coco['annotations']

                # Group annotations by image
                self.img_to_anns = {}
                for ann in self.annotations:
                    img_id = ann['image_id']
                    if img_id not in self.img_to_anns:
                        self.img_to_anns[img_id] = []
                    self.img_to_anns[img_id].append(ann)

                self.img_ids = list(self.img_to_anns.keys())

            def __len__(self):
                return len(self.img_ids)

            def __getitem__(self, idx):
                img_id = self.img_ids[idx]
                img_info = self.images[img_id]
                anns = self.img_to_anns[img_id]

                # Load image
                img_path = self.img_dir / img_info['file_name']
                img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img.size

                # Resize
                img = img.resize((self.img_size[1], self.img_size[0]))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

                # Process annotations
                boxes = []
                masks = []
                scale_x = self.img_size[1] / orig_w
                scale_y = self.img_size[0] / orig_h

                for ann in anns[:16]:  # Limit annotations per image
                    # Scale bbox
                    x, y, w, h = ann['bbox']
                    x1 = x * scale_x
                    y1 = y * scale_y
                    x2 = (x + w) * scale_x
                    y2 = (y + h) * scale_y
                    boxes.append([x1, y1, x2, y2])

                    # Create mask from segmentation
                    if 'segmentation' in ann and len(ann['segmentation']) > 0:
                        import cv2
                        mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
                        for seg in ann['segmentation']:
                            pts = np.array(seg).reshape(-1, 2)
                            pts[:, 0] *= scale_x
                            pts[:, 1] *= scale_y
                            pts = pts.astype(np.int32)
                            cv2.fillPoly(mask, [pts], 1)
                        masks.append(mask)
                    else:
                        # Box mask
                        mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
                        mask[int(y1):int(y2), int(x1):int(x2)] = 1
                        masks.append(mask)

                if len(boxes) == 0:
                    boxes = [[0, 0, 1, 1]]
                    masks = [np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)]

                boxes = torch.tensor(boxes, dtype=torch.float32)
                masks = torch.tensor(np.stack(masks), dtype=torch.float32)

                return img_tensor, boxes, masks

        # Create datasets
        train_dataset = SimpleSAM3Dataset(
            config["paths"]["train_annotation"],
            config["paths"]["train_images"],
            config["training"]["image_size"]
        )

        val_dataset = SimpleSAM3Dataset(
            config["paths"]["val_annotation"],
            config["paths"]["val_images"],
            config["training"]["image_size"]
        )

        if len(train_dataset) == 0:
            print("No training data found!")
            return

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            collate_fn=self._collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            collate_fn=self._collate_fn
        )

        # Optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["max_epochs"]
        )

        # Loss functions
        bce_loss = nn.BCEWithLogitsLoss()
        dice_loss_fn = self._dice_loss

        # AMP scaler
        scaler = torch.amp.GradScaler(
            'cuda',
            enabled=config["training"]["amp_enabled"]
        )

        # Training loop
        best_val_loss = float('inf')
        checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        checkpoint_dir.mkdir(exist_ok=True)

        for epoch in range(config["training"]["max_epochs"]):
            model.train()
            train_loss = 0
            num_batches = 0

            for batch_idx, (images, boxes_list, masks_list) in enumerate(train_loader):
                images = images.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=config["training"]["amp_enabled"]):
                    # Process each image in batch
                    total_loss = 0
                    for i in range(len(images)):
                        boxes = boxes_list[i].to(device)
                        gt_masks = masks_list[i].to(device)

                        # Get image embeddings
                        with torch.no_grad() if config["model"]["freeze_image_encoder"] else torch.enable_grad():
                            image_embeddings = model.image_encoder(images[i:i+1])

                        # Predict masks for each box
                        for j in range(len(boxes)):
                            box = boxes[j:j+1]
                            gt_mask = gt_masks[j:j+1]

                            # Encode prompts
                            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                                points=None,
                                boxes=box,
                                masks=None,
                            )

                            # Decode masks
                            low_res_masks, iou_predictions = model.mask_decoder(
                                image_embeddings=image_embeddings,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )

                            # Upsample masks
                            pred_masks = nn.functional.interpolate(
                                low_res_masks,
                                size=gt_mask.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )

                            # Compute loss
                            loss = bce_loss(pred_masks.squeeze(1), gt_mask) + \
                                   dice_loss_fn(pred_masks.squeeze(1), gt_mask)
                            total_loss += loss

                    if total_loss > 0:
                        scaler.scale(total_loss).backward()

                        if config["training"]["gradient_clip"] > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config["training"]["gradient_clip"]
                            )

                        scaler.step(optimizer)
                        scaler.update()

                        train_loss += total_loss.item()
                        num_batches += 1

            scheduler.step()

            avg_train_loss = train_loss / max(num_batches, 1)

            # Validation
            model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for images, boxes_list, masks_list in val_loader:
                    images = images.to(device)

                    for i in range(len(images)):
                        boxes = boxes_list[i].to(device)
                        gt_masks = masks_list[i].to(device)

                        image_embeddings = model.image_encoder(images[i:i+1])

                        for j in range(len(boxes)):
                            box = boxes[j:j+1]
                            gt_mask = gt_masks[j:j+1]

                            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                                points=None,
                                boxes=box,
                                masks=None,
                            )

                            low_res_masks, _ = model.mask_decoder(
                                image_embeddings=image_embeddings,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )

                            pred_masks = nn.functional.interpolate(
                                low_res_masks,
                                size=gt_mask.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )

                            loss = bce_loss(pred_masks.squeeze(1), gt_mask) + \
                                   dice_loss_fn(pred_masks.squeeze(1), gt_mask)
                            val_loss += loss.item()
                            num_val_batches += 1

            avg_val_loss = val_loss / max(num_val_batches, 1)

            print(f"Epoch {epoch+1}/{config['training']['max_epochs']}: "
                  f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_e{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = checkpoint_dir / "best_model.pth"
                torch.save(model.state_dict(), best_path)

        # Store checkpoint dir for later use by _get_output_map
        self._final_checkpoint_dir = checkpoint_dir

    def _collate_fn(self, batch):
        """Custom collate function for variable-size annotations."""
        images = torch.stack([item[0] for item in batch])
        boxes_list = [item[1] for item in batch]
        masks_list = [item[2] for item in batch]
        return images, boxes_list, masks_list

    def _dice_loss(self, pred, target, smooth=1.0):
        """Compute Dice loss."""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def _get_output_map(self, checkpoint_dir):
        """Build output map for process_trainer_output."""
        from pathlib import Path

        output = {}
        output_model_name = "sam3_finetuned.pth"

        checkpoint_dir = Path(checkpoint_dir)
        best_model = checkpoint_dir / "best_model.pth"

        if not best_model.exists():
            print("\n[SAM3Trainer] No best model found, training may have failed")
            return output

        algo = "sam3"

        output["type"] = algo
        output[algo + ":checkpoint_path"] = output_model_name
        output[output_model_name] = str(best_model)

        print(f"\nModel found at: {best_model}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output

    def update_model(self):
        """Main training entry point."""
        print("Starting SAM3 training...")

        # Prepare COCO-format dataset
        sam3_dir = self._prepare_coco_dataset()

        # Generate training config
        config_path = self._generate_training_config(sam3_dir)

        # Train the model
        self._train_sam3_model(sam3_dir, config_path)

        checkpoint_dir = getattr(self, '_final_checkpoint_dir',
                                 Path(self._train_directory) / "checkpoints")
        output = self._get_output_map(checkpoint_dir)

        print("\nSAM3 training complete!")

        return output


class SAM3TrackerTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for SAM3 tracker fine-tuning.

    Fine-tunes SAM3 mask propagation using ground-truth tracks so that
    tracked masks stay locked onto the annotated object class across
    frames.
    """

    def __init__(self):
        TrainTracker.__init__(self)

        # Model configuration
        self._identifier = "viame-sam3-tracker"
        self._sam_model_id = "facebook/sam2.1-hiera-large"
        self._train_directory = "deep_training"

        # Training configuration
        self._gpu_count = -1
        self._max_epochs = "20"
        self._batch_size = "4"
        self._learning_rate = "1e-5"
        self._timeout = "604800"
        self._chip_width = "1024"
        self._chip_height = "1024"

        # Clip sampling configuration
        self._clip_length = "8"
        self._clip_stride = "4"
        self._min_clip_length = "2"
        self._max_clips_per_track = "20"

        # Fine-tuning configuration
        self._freeze_image_encoder = "true"
        self._freeze_prompt_encoder = "true"
        self._amp_enabled = "true"

        # Internal state
        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration(self):
        cfg = super(TrainTracker, self).get_configuration()

        cfg.set_value("identifier", self._identifier)
        cfg.set_value("sam_model_id", self._sam_model_id)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("gpu_count", str(self._gpu_count))
        cfg.set_value("max_epochs", self._max_epochs)
        cfg.set_value("batch_size", self._batch_size)
        cfg.set_value("learning_rate", self._learning_rate)
        cfg.set_value("timeout", self._timeout)
        cfg.set_value("chip_width", self._chip_width)
        cfg.set_value("chip_height", self._chip_height)
        cfg.set_value("clip_length", self._clip_length)
        cfg.set_value("clip_stride", self._clip_stride)
        cfg.set_value("min_clip_length", self._min_clip_length)
        cfg.set_value("max_clips_per_track", self._max_clips_per_track)
        cfg.set_value("freeze_image_encoder", self._freeze_image_encoder)
        cfg.set_value("freeze_prompt_encoder", self._freeze_prompt_encoder)
        cfg.set_value("amp_enabled", self._amp_enabled)

        return cfg

    @report_cuda_errors("SAM3TrackerTrainer initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._sam_model_id = str(cfg.get_value("sam_model_id"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._gpu_count = int(cfg.get_value("gpu_count"))
        self._max_epochs = str(cfg.get_value("max_epochs"))
        self._batch_size = str(cfg.get_value("batch_size"))
        self._learning_rate = str(cfg.get_value("learning_rate"))
        self._timeout = str(cfg.get_value("timeout"))
        self._chip_width = str(cfg.get_value("chip_width"))
        self._chip_height = str(cfg.get_value("chip_height"))
        self._clip_length = str(cfg.get_value("clip_length"))
        self._clip_stride = str(cfg.get_value("clip_stride"))
        self._min_clip_length = str(cfg.get_value("min_clip_length"))
        self._max_clips_per_track = str(cfg.get_value("max_clips_per_track"))
        self._freeze_image_encoder = str(cfg.get_value("freeze_image_encoder"))
        self._freeze_prompt_encoder = str(cfg.get_value("freeze_prompt_encoder"))
        self._amp_enabled = str(cfg.get_value("amp_enabled"))

        try:
            import torch
            if torch.cuda.is_available():
                if self._gpu_count < 0:
                    self._gpu_count = torch.cuda.device_count()
        except ImportError:
            if self._gpu_count < 0:
                self._gpu_count = 1

        if self._train_directory:
            if not os.path.exists(self._train_directory):
                os.makedirs(self._train_directory)

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or \
           len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(self, categories, train_files, train_tracks,
                           test_files, test_tracks):
        print("Adding training data from disk...")
        print("  Training files: ", len(train_files))
        print("  Training tracks: ", len(train_tracks))
        print("  Test files: ", len(test_files))
        print("  Test tracks: ", len(test_tracks))

        if categories is not None:
            self._categories = categories.all_class_names()
        else:
            self._categories = []

        self._train_image_files = list(train_files)
        self._train_tracks = list(train_tracks)
        self._test_image_files = list(test_files)
        self._test_tracks = list(test_tracks)

    def _extract_clip_samples(self, image_files, track_sets):
        """
        Sample fixed-length clips from ground-truth tracks.

        Each clip follows a single object and is a list of dicts with keys
        'image', 'bbox' ([x1, y1, x2, y2]) and 'polygon' (flat [x, y, ...]
        list or None). Frame IDs index into the split's image file list.
        """
        clip_length = int(self._clip_length)
        clip_stride = int(self._clip_stride)
        min_clip_length = int(self._min_clip_length)
        max_clips_per_track = int(self._max_clips_per_track)

        image_map = {}
        for i, img_file in enumerate(image_files):
            image_map[i] = img_file

        clips = []

        for track_set in track_sets:
            if track_set is None:
                continue

            for track in track_set.tracks():
                frames = []
                for state in track:
                    det = state.detection()
                    if det is None:
                        continue

                    frame_id = state.frame_id
                    if frame_id not in image_map:
                        continue

                    img_path = image_map[frame_id]
                    if not os.path.exists(img_path):
                        continue

                    bbox = det.bounding_box
                    x1, y1 = bbox.min_x(), bbox.min_y()
                    x2, y2 = bbox.max_x(), bbox.max_y()
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue

                    polygon = None
                    try:
                        poly = det.polygon()
                        if poly is not None and poly.num_vertices() >= 3:
                            polygon = []
                            for i in range(poly.num_vertices()):
                                pt = poly.at(i)
                                polygon.extend([pt.x(), pt.y()])
                    except (AttributeError, RuntimeError):
                        polygon = None

                    frames.append({
                        'image': img_path,
                        'bbox': [x1, y1, x2, y2],
                        'polygon': polygon,
                    })

                if len(frames) < min_clip_length:
                    continue

                track_clips = 0
                for start in range(0, len(frames), clip_stride):
                    clip = frames[start:start + clip_length]
                    if len(clip) < min_clip_length:
                        break
                    clips.append(clip)
                    track_clips += 1
                    if track_clips >= max_clips_per_track:
                        break

        return clips

    def _load_model(self, device):
        """
        Load the SAM model to fine-tune.

        Tries native sam3, then sam2, then HuggingFace transformers. The
        returned model must expose image encoder, prompt encoder and mask
        decoder components (SAM / SAM2 style).
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            model = build_sam3_image_model(
                checkpoint_path=self._sam_model_id
                    if os.path.exists(self._sam_model_id) else None,
                device=str(device),
                eval_mode=False,
                load_from_HF=not os.path.exists(self._sam_model_id),
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )
            print("Loaded SAM3 model via native sam3 module")
            return model
        except ImportError:
            pass

        try:
            from sam2.build_sam import build_sam2
            model = build_sam2(
                config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                ckpt_path=self._sam_model_id,
                device=str(device),
                mode='train',
            )
            print("Loaded SAM2 model via native sam2 module")
            return model
        except ImportError:
            pass

        from transformers import Sam2Model
        model = Sam2Model.from_pretrained(self._sam_model_id)
        print("Loaded SAM2 model via HuggingFace transformers")
        return model

    @report_cuda_errors("SAM3TrackerTrainer training")
    def update_model(self):
        """Main training entry point."""
        print("Starting SAM3 tracker fine-tuning...")

        train_clips = self._extract_clip_samples(
            self._train_image_files, self._train_tracks)
        val_clips = self._extract_clip_samples(
            self._test_image_files, self._test_tracks)

        print(f"  Training clips: {len(train_clips)}")
        print(f"  Validation clips: {len(val_clips)}")

        if len(train_clips) == 0:
            print("No training clips could be sampled from track data!")
            return {}

        self._train_model(train_clips, val_clips)

        output = self._get_output_map()

        print("\nSAM3 tracker training complete!")

        return output

    def _train_model(self, train_clips, val_clips):
        """Fine-tune SAM mask propagation on the sampled clips."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        chip_w = int(self._chip_width)
        chip_h = int(self._chip_height)
        batch_size = int(self._batch_size)
        max_epochs = int(self._max_epochs)
        lr = float(self._learning_rate)
        amp_enabled = self._amp_enabled.lower() == "true"
        freeze_image = self._freeze_image_encoder.lower() == "true"
        freeze_prompt = self._freeze_prompt_encoder.lower() == "true"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        amp_enabled = amp_enabled and device.type == 'cuda'

        model = self._load_model(device)
        model = model.to(device)
        model.train()

        image_encoder = _resolve_attr(
            model, ['image_encoder', 'vision_encoder'])
        prompt_encoder = _resolve_attr(
            model, ['prompt_encoder', 'sam_prompt_encoder'])
        mask_decoder = _resolve_attr(
            model, ['mask_decoder', 'sam_mask_decoder'])

        if image_encoder is None or prompt_encoder is None or \
           mask_decoder is None:
            raise RuntimeError(
                "Loaded SAM model does not expose image/prompt/mask "
                "components required for fine-tuning")

        if freeze_image:
            for param in image_encoder.parameters():
                param.requires_grad = False
            print("  Image encoder frozen")

        if freeze_prompt:
            for param in prompt_encoder.parameters():
                param.requires_grad = False
            print("  Prompt encoder frozen")

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs)
        bce_loss = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        checkpoint_dir = Path(self._train_directory) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        best_val_loss = float('inf')

        def run_clip(clip):
            """
            Propagate through one clip; returns (loss, frame count).

            Frame 0 is prompted with the GT box, later frames only with
            the previous frame's predicted low-res mask so gradients teach
            the decoder to hold onto the object over time.
            """
            clip_loss = 0
            frames_used = 0
            prev_low_res = None

            for t, frame in enumerate(clip):
                sample = self._load_frame(frame, chip_w, chip_h)
                if sample is None:
                    continue
                img_tensor, box, gt_mask = sample
                img_tensor = img_tensor.unsqueeze(0).to(device)
                gt_mask = gt_mask.unsqueeze(0).to(device)

                encoder_ctx = torch.no_grad() if freeze_image \
                    else torch.enable_grad()
                with encoder_ctx:
                    image_embeddings = image_encoder(img_tensor)
                if isinstance(image_embeddings, dict):
                    image_embeddings = image_embeddings.get(
                        'vision_features', image_embeddings)

                if t == 0 or prev_low_res is None:
                    boxes = torch.tensor(
                        box, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    masks_prompt = None
                else:
                    boxes = None
                    masks_prompt = prev_low_res.detach()

                sparse_embeddings, dense_embeddings = prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=masks_prompt,
                )

                low_res_masks, iou_predictions = mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                pred_masks = nn.functional.interpolate(
                    low_res_masks,
                    size=gt_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

                loss = bce_loss(pred_masks.squeeze(1), gt_mask) + \
                    self._dice_loss(pred_masks.squeeze(1), gt_mask)
                clip_loss = clip_loss + loss
                frames_used += 1

                prev_low_res = low_res_masks

            return clip_loss, frames_used

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            num_clips = 0
            optimizer.zero_grad()

            for clip_idx, clip in enumerate(train_clips):
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    clip_loss, frames_used = run_clip(clip)

                if frames_used == 0:
                    continue

                clip_loss = clip_loss / frames_used
                scaler.scale(clip_loss / batch_size).backward()

                # Step once every batch_size clips (gradient accumulation)
                if (clip_idx + 1) % batch_size == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += clip_loss.item()
                num_clips += 1

            scheduler.step()
            avg_train_loss = train_loss / max(num_clips, 1)

            model.eval()
            val_loss = 0
            num_val_clips = 0

            with torch.no_grad():
                for clip in val_clips:
                    with torch.amp.autocast('cuda', enabled=amp_enabled):
                        clip_loss, frames_used = run_clip(clip)
                    if frames_used == 0:
                        continue
                    val_loss += (clip_loss / frames_used).item()
                    num_val_clips += 1

            avg_val_loss = val_loss / max(num_val_clips, 1)

            print(f"Epoch {epoch+1}/{max_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}")

            checkpoint_path = checkpoint_dir / f"checkpoint_e{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

            # With no validation clips fall back to tracking train loss
            selection_loss = avg_val_loss if num_val_clips > 0 \
                else avg_train_loss
            if selection_loss < best_val_loss:
                best_val_loss = selection_loss
                torch.save(model.state_dict(),
                           checkpoint_dir / "best_model.pth")

    def _load_frame(self, frame, chip_w, chip_h):
        """
        Load one clip frame as (image tensor, scaled box, GT mask).

        Returns None if the image cannot be read.
        """
        import cv2
        import torch

        img = cv2.imread(frame['image'])
        if img is None:
            return None

        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (chip_w, chip_h))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        scale_x = chip_w / orig_w
        scale_y = chip_h / orig_h

        x1, y1, x2, y2 = frame['bbox']
        box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

        mask = np.zeros((chip_h, chip_w), dtype=np.uint8)
        if frame['polygon'] is not None:
            pts = np.array(frame['polygon'], dtype=np.float64).reshape(-1, 2)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        else:
            mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

        gt_mask = torch.from_numpy(mask).float()

        return img_tensor, box, gt_mask

    def _dice_loss(self, pred, target, smooth=1.0):
        """Compute Dice loss."""
        import torch
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / \
            (pred_flat.sum() + target_flat.sum() + smooth)

    def _get_output_map(self):
        """Build output map for process_trainer_output."""
        output = {}
        output_model_name = "sam3_tracker_finetuned.pth"

        checkpoint_dir = Path(self._train_directory) / "checkpoints"
        best_model = checkpoint_dir / "best_model.pth"

        if not best_model.exists():
            print("\n[SAM3TrackerTrainer] No best model found, "
                  "training may have failed")
            return output

        algo = "sam3_tracker"

        output["type"] = algo
        output[algo + ":sam_model_id"] = output_model_name
        output[output_model_name] = str(best_model)

        print(f"\nModel found at: {best_model}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register detector trainer (train_detector type)
    implementation_name = "sam3"
    if not algorithm_factory.has_algorithm_impl_name(
            SAM3Trainer.static_type_name(), implementation_name):
        algorithm_factory.add_algorithm(
            implementation_name,
            "SAM3 (Segment Anything Model 3) fine-tuning for segmentation",
            SAM3Trainer
        )
        algorithm_factory.mark_algorithm_as_loaded(implementation_name)

    # Register tracker trainer (train_tracker type)
    if not algorithm_factory.has_algorithm_impl_name(
            SAM3TrackerTrainer.static_type_name(), implementation_name):
        algorithm_factory.add_algorithm(
            implementation_name,
            "SAM3 tracker fine-tuning with temporal mask propagation",
            SAM3TrackerTrainer
        )
        algorithm_factory.mark_algorithm_as_loaded(implementation_name)
