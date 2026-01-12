# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
DeepSORT tracker training implementation.

DeepSORT uses a Re-ID (re-identification) network to extract appearance
features for matching detections across frames. This trainer:
1. Extracts detection crops from track groundtruth
2. Trains a Re-ID network using triplet loss
3. Outputs the trained model for use with deepsort_tracker

The Re-ID network learns to produce similar embeddings for the same
object across different frames and dissimilar embeddings for different objects.
"""

from kwiver.vital.algo import TrainTracker

from kwiver.vital.types import (
    CategoryHierarchy,
    ObjectTrackSet, ObjectTrackState,
    BoundingBoxD, DetectedObjectType
)

from distutils.util import strtobool
from shutil import copyfile
from pathlib import Path

import os
import sys
import shutil
import subprocess
import signal
import time
import threading
import json
import random


class DeepSORTTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for DeepSORT Re-ID model training.

    Trains a CNN to extract appearance features for re-identification.
    """
    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-deepsort-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "deepsort_tracker"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._max_epochs = "50"
        self._batch_size = "32"
        self._learning_rate = "0.0003"
        self._threshold = "0.00"
        self._timeout = "604800"
        self._crop_size = "128x64"  # HxW for Re-ID crops
        self._embedding_dim = "512"
        self._backbone = "resnet18"

        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration(self):
        cfg = super(TrainTracker, self).get_configuration()

        cfg.set_value("identifier", self._identifier)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("output_directory", self._output_directory)
        cfg.set_value("output_prefix", self._output_prefix)
        cfg.set_value("pipeline_template", self._pipeline_template)
        cfg.set_value("gpu_count", str(self._gpu_count))
        cfg.set_value("max_epochs", self._max_epochs)
        cfg.set_value("batch_size", self._batch_size)
        cfg.set_value("learning_rate", self._learning_rate)
        cfg.set_value("threshold", self._threshold)
        cfg.set_value("timeout", self._timeout)
        cfg.set_value("crop_size", self._crop_size)
        cfg.set_value("embedding_dim", self._embedding_dim)
        cfg.set_value("backbone", self._backbone)

        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._output_directory = str(cfg.get_value("output_directory"))
        self._output_prefix = str(cfg.get_value("output_prefix"))
        self._pipeline_template = str(cfg.get_value("pipeline_template"))
        self._gpu_count = int(cfg.get_value("gpu_count"))
        self._max_epochs = str(cfg.get_value("max_epochs"))
        self._batch_size = str(cfg.get_value("batch_size"))
        self._learning_rate = str(cfg.get_value("learning_rate"))
        self._threshold = str(cfg.get_value("threshold"))
        self._timeout = str(cfg.get_value("timeout"))
        self._crop_size = str(cfg.get_value("crop_size"))
        self._embedding_dim = str(cfg.get_value("embedding_dim"))
        self._backbone = str(cfg.get_value("backbone"))

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

        if self._output_directory:
            if not os.path.exists(self._output_directory):
                os.makedirs(self._output_directory)

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

    def _prepare_reid_data(self):
        """
        Prepare Re-ID training data from track groundtruth.

        Creates a directory structure for training:
        - reid_data/train/{track_id}/{frame_id}.jpg
        - reid_data/test/{track_id}/{frame_id}.jpg

        Each track_id folder contains crops of the same object across frames.
        """
        import cv2
        import numpy as np

        crop_h, crop_w = map(int, self._crop_size.split('x'))

        reid_dir = Path(self._train_directory) / "reid_data"
        if reid_dir.exists():
            shutil.rmtree(reid_dir)

        train_dir = reid_dir / "train"
        test_dir = reid_dir / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        print("Preparing Re-ID training data...")

        # Build image file mapping
        image_map = {}
        for i, img_file in enumerate(self._train_image_files):
            image_map[i] = img_file

        # Process training tracks
        train_count = self._process_split_data(
            self._train_tracks, image_map, train_dir, crop_h, crop_w, "train"
        )

        # Process test tracks
        test_image_map = {}
        for i, img_file in enumerate(self._test_image_files):
            test_image_map[i] = img_file

        test_count = self._process_split_data(
            self._test_tracks, test_image_map, test_dir, crop_h, crop_w, "test"
        )

        print(f"  Train: {train_count} crops")
        print(f"  Test: {test_count} crops")

        return reid_dir

    def _process_split_data(self, track_sets, image_map, output_dir, crop_h, crop_w, split_name):
        """Process tracks for one split (train/test)."""
        import cv2
        import numpy as np

        total_crops = 0
        global_track_id = 0

        for seq_idx, track_set in enumerate(track_sets):
            if track_set is None:
                continue

            # Group detections by frame for efficient image loading
            frame_to_detections = {}

            for track in track_set.tracks():
                track_id = track.id()
                unique_track_id = f"seq{seq_idx:04d}_track{track_id:06d}"

                for state in track:
                    frame_id = state.frame()
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box()
                    x1 = int(bbox.min_x())
                    y1 = int(bbox.min_y())
                    x2 = int(bbox.max_x())
                    y2 = int(bbox.max_y())

                    if frame_id not in frame_to_detections:
                        frame_to_detections[frame_id] = []

                    frame_to_detections[frame_id].append({
                        'track_id': unique_track_id,
                        'bbox': (x1, y1, x2, y2),
                        'frame_id': frame_id
                    })

            # Process each frame
            for frame_id, detections in frame_to_detections.items():
                if frame_id not in image_map:
                    continue

                img_path = image_map[frame_id]
                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_h, img_w = img.shape[:2]

                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    track_id = det['track_id']

                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(x1 + 1, min(x2, img_w))
                    y2 = max(y1 + 1, min(y2, img_h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop and resize
                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (crop_w, crop_h))

                    # Save crop
                    track_dir = output_dir / track_id
                    track_dir.mkdir(exist_ok=True)

                    crop_path = track_dir / f"{det['frame_id']:06d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    total_crops += 1

        return total_crops

    def update_model(self):
        """
        Train the Re-ID model using triplet loss.
        """
        print("Starting DeepSORT Re-ID training...")

        # Prepare training data
        reid_dir = self._prepare_reid_data()

        # Train the model
        self._train_reid_model(reid_dir)

        print("\nDeepSORT Re-ID training complete!\n")

    def _train_reid_model(self, reid_dir):
        """
        Train Re-ID model using PyTorch.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            import torchvision.transforms as transforms
            from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
            import cv2
            from PIL import Image
        except ImportError as e:
            print(f"PyTorch not available: {e}")
            return

        crop_h, crop_w = map(int, self._crop_size.split('x'))
        embedding_dim = int(self._embedding_dim)
        batch_size = int(self._batch_size)
        max_epochs = int(self._max_epochs)
        lr = float(self._learning_rate)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Create Re-ID model
        class ReIDModel(nn.Module):
            def __init__(self, backbone_name, embedding_dim):
                super().__init__()
                if backbone_name == 'resnet50':
                    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
                    backbone_dim = 2048
                else:
                    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
                    backbone_dim = 512

                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                self.embedding = nn.Linear(backbone_dim, embedding_dim)
                self.bn = nn.BatchNorm1d(embedding_dim)

            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.embedding(x)
                x = self.bn(x)
                x = nn.functional.normalize(x, dim=1)
                return x

        # Create dataset
        class ReIDDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.data_dir = Path(data_dir)
                self.transform = transform
                self.samples = []
                self.labels = []
                self.label_to_idx = {}

                # Collect all samples
                for idx, track_dir in enumerate(sorted(self.data_dir.iterdir())):
                    if not track_dir.is_dir():
                        continue

                    self.label_to_idx[track_dir.name] = idx
                    for img_path in track_dir.glob("*.jpg"):
                        self.samples.append(str(img_path))
                        self.labels.append(idx)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img = Image.open(self.samples[idx]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, self.labels[idx]

        # Triplet loss
        class TripletLoss(nn.Module):
            def __init__(self, margin=0.3):
                super().__init__()
                self.margin = margin

            def forward(self, embeddings, labels):
                # Get pairwise distances
                dist_mat = torch.cdist(embeddings, embeddings, p=2)

                # For each anchor, find hardest positive and negative
                labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                labels = labels.to(embeddings.device)

                n = embeddings.size(0)
                mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
                mask_neg = ~mask_pos

                # Set diagonal to False for positives
                mask_pos.fill_diagonal_(False)

                loss = 0
                count = 0
                for i in range(n):
                    pos_dists = dist_mat[i][mask_pos[i]]
                    neg_dists = dist_mat[i][mask_neg[i]]

                    if len(pos_dists) == 0 or len(neg_dists) == 0:
                        continue

                    hardest_pos = pos_dists.max()
                    hardest_neg = neg_dists.min()

                    triplet_loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0)
                    loss += triplet_loss
                    count += 1

                return loss / max(count, 1)

        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((crop_h, crop_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((crop_h, crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets and loaders
        train_dataset = ReIDDataset(reid_dir / "train", transform)
        test_dataset = ReIDDataset(reid_dir / "test", transform_test)

        if len(train_dataset) == 0:
            print("No training data found!")
            return

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Number of identities (train): {len(train_dataset.label_to_idx)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create model and optimizer
        model = ReIDModel(self._backbone, embedding_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = TripletLoss(margin=0.3)

        # Training loop
        best_loss = float('inf')
        snapshot_dir = Path(self._train_directory) / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            num_batches = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)

                optimizer.zero_grad()
                embeddings = model(images)
                loss = criterion(embeddings, labels)

                if loss.item() > 0:
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_train_loss = train_loss / max(num_batches, 1)

            # Validation
            model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    embeddings = model(images)
                    loss = criterion(embeddings, labels)
                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / max(num_val_batches, 1)

            print(f"Epoch {epoch+1}/{max_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

            # Save checkpoint
            checkpoint_path = snapshot_dir / f"checkpoint_e{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_path = snapshot_dir / "best_model.pth"
                torch.save(model.state_dict(), best_path)

        # Save final model
        self._save_final_model(snapshot_dir)

    def _save_final_model(self, snapshot_dir):
        """Copy best model to output directory."""
        best_model = snapshot_dir / "best_model.pth"

        if best_model.exists():
            dst_model = Path(self._output_directory) / "deepsort_reid.pth"
            copyfile(best_model, dst_model)
            print(f"Copied model to {dst_model}")

            # Generate pipeline config
            if self._pipeline_template and os.path.exists(self._pipeline_template):
                with open(self._pipeline_template, 'r') as fin:
                    template = fin.read()

                pipeline = template.replace("[-MODEL-FILE-]", "deepsort_reid.pth")

                output_pipeline = Path(self._output_directory) / "tracker.pipe"
                with open(output_pipeline, 'w') as fout:
                    fout.write(pipeline)
                print(f"Generated pipeline: {output_pipeline}")
            else:
                # Generate default config
                config_content = f"""# DeepSORT tracker configuration
# Generated by deepsort_trainer

process tracker
  :: deepsort_tracker
  model_path = deepsort_reid.pth
  max_dist = 0.2
  min_confidence = 0.3
  max_iou_distance = 0.7
  max_age = 30
  n_init = 3
"""
                config_path = Path(self._output_directory) / "deepsort_tracker.pipe"
                with open(config_path, 'w') as f:
                    f.write(config_content)
                print(f"Generated config: {config_path}")
        else:
            print("Warning: No best model found")


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "deepsort"

    if algorithm_factory.has_algorithm_impl_name(
        DeepSORTTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "PyTorch DeepSORT Re-ID model training",
        DeepSORTTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
