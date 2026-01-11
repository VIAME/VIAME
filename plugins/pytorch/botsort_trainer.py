# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
BoT-SORT tracker training implementation.

BoT-SORT training involves:
1. Training a Re-ID model for appearance features (same as DeepSORT)
2. Estimating optimal tracking parameters from groundtruth
3. Configuring camera motion compensation settings

The trainer produces a Re-ID model and configuration file with
optimized parameters for the target domain.
"""

from __future__ import print_function
from __future__ import division

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
import json
import numpy as np


class BoTSORTTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for BoT-SORT training.

    Trains Re-ID model and estimates tracking parameters.
    """
    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-botsort-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "botsort_tracker"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._max_epochs = "50"
        self._batch_size = "32"
        self._learning_rate = "0.0003"
        self._threshold = "0.00"
        self._timeout = "604800"
        self._crop_size = "128x64"
        self._embedding_dim = "512"
        self._backbone = "resnet18"
        self._feat_ema_alpha = "0.9"
        self._use_cmc = True
        self._use_reid = True

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
        cfg.set_value("feat_ema_alpha", self._feat_ema_alpha)
        cfg.set_value("use_cmc", str(self._use_cmc))
        cfg.set_value("use_reid", str(self._use_reid))

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
        self._feat_ema_alpha = str(cfg.get_value("feat_ema_alpha"))
        self._use_cmc = strtobool(cfg.get_value("use_cmc"))
        self._use_reid = strtobool(cfg.get_value("use_reid"))

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

    def _extract_track_statistics(self):
        """Extract statistics for parameter estimation."""
        positions = []
        velocities = []
        confidences = []
        track_lengths = []
        gap_lengths = []

        all_tracks = self._train_tracks + self._test_tracks

        for track_set in all_tracks:
            if track_set is None:
                continue

            for track in track_set.tracks():
                states = list(track)
                track_lengths.append(len(states))

                prev_frame = None
                prev_cx, prev_cy = None, None

                for state in states:
                    frame_id = state.frame()
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box()
                    x1 = bbox.min_x()
                    y1 = bbox.min_y()
                    x2 = bbox.max_x()
                    y2 = bbox.max_y()
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    positions.append((cx, cy, w, h))

                    if det.confidence is not None:
                        confidences.append(det.confidence)

                    if prev_frame is not None and prev_cx is not None:
                        dt = frame_id - prev_frame
                        if dt > 0:
                            vx = (cx - prev_cx) / dt
                            vy = (cy - prev_cy) / dt
                            velocities.append((vx, vy, h, dt))

                            if dt > 1:
                                gap_lengths.append(dt - 1)

                    prev_frame = frame_id
                    prev_cx, prev_cy = cx, cy

        return {
            'positions': positions,
            'velocities': velocities,
            'confidences': confidences,
            'track_lengths': track_lengths,
            'gap_lengths': gap_lengths
        }

    def _estimate_parameters(self, stats):
        """Estimate tracking parameters."""
        params = {}

        # Kalman filter parameters
        velocities = stats['velocities']
        if len(velocities) >= 10:
            pos_variances = []
            for vx, vy, h, dt in velocities:
                if h > 0 and dt == 1:
                    pos_var = np.sqrt(vx**2 + vy**2) / h
                    pos_variances.append(pos_var)

            if len(pos_variances) > 0:
                params['std_weight_position'] = float(np.clip(np.median(pos_variances) * 2, 0.01, 0.5))
            else:
                params['std_weight_position'] = 1.0 / 20
        else:
            params['std_weight_position'] = 1.0 / 20

        params['std_weight_velocity'] = params['std_weight_position'] / 8

        # Confidence thresholds
        confidences = stats['confidences']
        if len(confidences) >= 10:
            confidences = np.array(confidences)
            params['high_thresh'] = float(np.clip(np.percentile(confidences, 30), 0.3, 0.9))
            params['low_thresh'] = float(np.clip(np.percentile(confidences, 10), 0.05, params['high_thresh'] - 0.1))
            params['new_track_thresh'] = params['high_thresh']
        else:
            params['high_thresh'] = 0.6
            params['low_thresh'] = 0.1
            params['new_track_thresh'] = 0.6

        # Track buffer
        gap_lengths = stats['gap_lengths']
        if len(gap_lengths) >= 5:
            params['track_buffer'] = int(np.clip(np.percentile(gap_lengths, 90) * 1.5 + 5, 10, 100))
        else:
            params['track_buffer'] = 30

        params['match_thresh'] = 0.8
        params['iou_weight'] = 0.5  # Balance between IOU and ReID
        params['feat_ema_alpha'] = float(self._feat_ema_alpha)

        return params

    def _prepare_reid_data(self):
        """Prepare Re-ID training data (same as DeepSORT)."""
        import cv2

        crop_h, crop_w = map(int, self._crop_size.split('x'))

        reid_dir = Path(self._train_directory) / "reid_data"
        if reid_dir.exists():
            shutil.rmtree(reid_dir)

        train_dir = reid_dir / "train"
        test_dir = reid_dir / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        print("Preparing Re-ID training data...")

        image_map = {}
        for i, img_file in enumerate(self._train_image_files):
            image_map[i] = img_file

        train_count = self._process_split_data(
            self._train_tracks, image_map, train_dir, crop_h, crop_w
        )

        test_image_map = {}
        for i, img_file in enumerate(self._test_image_files):
            test_image_map[i] = img_file

        test_count = self._process_split_data(
            self._test_tracks, test_image_map, test_dir, crop_h, crop_w
        )

        print(f"  Train: {train_count} crops")
        print(f"  Test: {test_count} crops")

        return reid_dir

    def _process_split_data(self, track_sets, image_map, output_dir, crop_h, crop_w):
        """Process tracks for one split."""
        import cv2

        total_crops = 0

        for seq_idx, track_set in enumerate(track_sets):
            if track_set is None:
                continue

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

                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(x1 + 1, min(x2, img_w))
                    y2 = max(y1 + 1, min(y2, img_h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (crop_w, crop_h))

                    track_dir = output_dir / track_id
                    track_dir.mkdir(exist_ok=True)

                    crop_path = track_dir / f"{det['frame_id']:06d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    total_crops += 1

        return total_crops

    def update_model(self):
        """Train Re-ID model and estimate parameters."""
        print("Starting BoT-SORT training...")

        # Extract statistics
        print("Extracting track statistics...")
        stats = self._extract_track_statistics()

        # Estimate parameters
        print("Estimating tracking parameters...")
        params = self._estimate_parameters(stats)

        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Train Re-ID model if enabled
        if self._use_reid:
            reid_dir = self._prepare_reid_data()
            self._train_reid_model(reid_dir)

        # Save parameters and config
        params['use_cmc'] = bool(self._use_cmc)
        params['use_reid'] = bool(self._use_reid)

        params_file = os.path.join(self._output_directory, "botsort_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved parameters to {params_file}")

        self._save_final_config(params)

        print("\nBoT-SORT training complete!\n")

    def _train_reid_model(self, reid_dir):
        """Train Re-ID model using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            import torchvision.transforms as transforms
            from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
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
        print(f"Training Re-ID model on {device}...")

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

        class ReIDDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.data_dir = Path(data_dir)
                self.transform = transform
                self.samples = []
                self.labels = []
                self.label_to_idx = {}

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

        class TripletLoss(nn.Module):
            def __init__(self, margin=0.3):
                super().__init__()
                self.margin = margin

            def forward(self, embeddings, labels):
                dist_mat = torch.cdist(embeddings, embeddings, p=2)
                labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                labels = labels.to(embeddings.device)

                n = embeddings.size(0)
                mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
                mask_neg = ~mask_pos
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

        train_dataset = ReIDDataset(reid_dir / "train", transform)
        test_dataset = ReIDDataset(reid_dir / "test", transform_test)

        if len(train_dataset) == 0:
            print("No training data found!")
            return

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = ReIDModel(self._backbone, embedding_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = TripletLoss(margin=0.3)

        best_loss = float('inf')
        snapshot_dir = Path(self._train_directory) / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)

        for epoch in range(int(max_epochs)):
            model.train()
            train_loss = 0
            num_batches = 0

            for images, labels in train_loader:
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

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), snapshot_dir / "best_model.pth")

        # Save final model
        best_model = snapshot_dir / "best_model.pth"
        if best_model.exists():
            dst_model = Path(self._output_directory) / "botsort_reid.pth"
            copyfile(best_model, dst_model)
            print(f"Copied Re-ID model to {dst_model}")

    def _save_final_config(self, params):
        """Generate pipeline configuration file."""
        model_file = "botsort_reid.pth" if self._use_reid else ""

        config_content = f"""# BoT-SORT tracker configuration
# Generated by botsort_trainer

process tracker
  :: botsort_tracker
  high_thresh = {params['high_thresh']:.3f}
  low_thresh = {params['low_thresh']:.3f}
  match_thresh = {params['match_thresh']:.3f}
  track_buffer = {params['track_buffer']}
  new_track_thresh = {params['new_track_thresh']:.3f}
  use_cmc = {str(params['use_cmc']).lower()}
  use_reid = {str(params['use_reid']).lower()}
  iou_weight = {params['iou_weight']:.3f}
  model_path = {model_file}
  feat_ema_alpha = {params['feat_ema_alpha']:.3f}
"""
        config_path = Path(self._output_directory) / "botsort_tracker.pipe"
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Generated config: {config_path}")


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "botsort"

    if algorithm_factory.has_algorithm_impl_name(
        BoTSORTTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "PyTorch BoT-SORT Re-ID model training and parameter estimation",
        BoTSORTTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
