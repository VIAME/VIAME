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
from viame.pytorch.utilities import report_cuda_errors


# The Re-ID dataset and batch sampler live at module scope rather than inside
# _train_reid_model because DataLoader worker processes have to be able to find
# them by name. Python 3.14 switched the default multiprocessing start method on
# Linux from fork to forkserver, which pickles the worker arguments; a class
# defined inside a method is a <locals> object and pickling it fails with
# "Can't pickle local object ... <locals>.ReIDDataset". Fork based Pythons never
# exercised this path, so the same code ran fine on 3.13 and earlier.
#
# torch is imported lazily elsewhere in this file so that the module still
# imports when torch is absent, which is how kwiver decides whether to register
# the trainer. The guard here keeps that property.
try:
    from torch.utils.data import Dataset as _TorchDataset, Sampler as _TorchSampler
except ImportError:
    _TorchDataset = object
    _TorchSampler = object


class ReIDDataset(_TorchDataset):
    """Crops on disk, one directory per track."""

    def __init__(self, data_dir, transform=None):
        from PIL import Image  # noqa: F401  (kept local, see module note)

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
        from PIL import Image

        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class PKSampler(_TorchSampler):
    """Yield batches of P identities with K crops each.

    Triplet loss can only produce a gradient from an anchor that has both a
    positive (same track) and a negative (different track) in the same batch.
    Drawing crops uniformly at random gives a same-identity collision with
    probability roughly B^2 / 2N for a batch of B over N identities, which for a
    track dataset this size is only a few percent -- so nearly every batch was a
    no-op and the Re-ID model never actually learned. Sampling K crops from each
    of P identities guarantees every sample has a positive.
    """

    def __init__(self, labels, p, k, num_batches=None):
        self.k = max(int(k), 2)

        self.by_id = {}
        for idx, label in enumerate(labels):
            self.by_id.setdefault(label, []).append(idx)

        # A track with a single crop can never supply a positive pair
        self.ids = [i for i, idxs in self.by_id.items() if len(idxs) >= 2]
        self.p = max(min(int(p), len(self.ids)), 1)

        if num_batches is None:
            num_batches = max(len(labels) // (self.p * self.k), 1)
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for track_id in random.sample(self.ids, self.p):
                pool = self.by_id[track_id]
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    # Short track, repeat crops to fill its slot
                    batch.extend(random.choices(pool, k=self.k))
            yield batch


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

    @report_cuda_errors("DeepSORTTrainer initialization")
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
                track_id = track.id
                unique_track_id = f"seq{seq_idx:04d}_track{track_id:06d}"

                for state in track:
                    frame_id = state.frame_id
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box
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

    @report_cuda_errors("DeepSORTTrainer training")
    def update_model(self):
        """
        Train the Re-ID model using triplet loss.

        Returns:
            dict: Map of template replacements and file copies
        """
        print("Starting DeepSORT Re-ID training...")

        # Prepare training data
        reid_dir = self._prepare_reid_data()

        # Train the model and get output map
        output = self._train_reid_model(reid_dir)

        print("\nDeepSORT Re-ID training complete!\n")

        return output if output else {}

    def _train_reid_model(self, reid_dir):
        """
        Train Re-ID model using PyTorch.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader, Sampler
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

                # Must stay a tensor: a batch with no valid anchor leaves this
                # untouched, and callers do loss.item() unconditionally
                loss = torch.zeros((), device=embeddings.device)
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

        # Batches are drawn identity-aware rather than uniformly at random, so
        # that every batch contains valid triplets. See PKSampler.
        crops_per_id = 4
        train_sampler = PKSampler(train_dataset.labels,
                                  max(batch_size // crops_per_id, 1),
                                  crops_per_id)

        if train_sampler.ids:
            print(f"PK sampling: {train_sampler.p} identities x "
                  f"{train_sampler.k} crops = {train_sampler.p * train_sampler.k} "
                  f"per batch, {len(train_sampler)} batches per epoch "
                  f"({len(train_sampler.ids)} identities with 2+ crops)")
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                      num_workers=4)
        else:
            print("Warning: no identity has more than one crop, falling back to "
                  "shuffled batches. Triplet loss cannot train on this data.")
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=4)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create model and optimizer
        model = ReIDModel(self._backbone, embedding_dim).to(device)

        # Batch-hard triplet loss on its own has a stable degenerate optimum:
        # map every crop to the same point, and hardest_pos == hardest_neg == 0
        # so the loss parks on the margin while the distance gradients vanish,
        # leaving an embedding that cannot tell anything apart. Earlier runs on
        # this dataset collapsed into exactly that, sitting at 0.3001 for 45 of
        # 50 epochs. An identity classifier alongside it removes the escape: a
        # constant embedding cannot classify, so cross entropy keeps a gradient
        # pointing away from collapse. The head is a training aid only and is
        # not part of the exported model.
        num_identities = max(len(train_dataset.label_to_idx), 1)
        classifier = nn.Linear(embedding_dim, num_identities).to(device)

        optimizer = optim.Adam(
            list(model.parameters()) + list(classifier.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = TripletLoss(margin=0.3)
        id_criterion = nn.CrossEntropyLoss()
        id_loss_weight = 1.0

        # Training loop
        best_loss = float('inf')
        snapshot_dir = Path(self._train_directory) / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)

        for epoch in range(max_epochs):
            model.train()
            classifier.train()
            train_loss = 0
            epoch_id_loss = 0
            num_batches = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)

                optimizer.zero_grad()
                embeddings = model(images)

                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                labels = labels.to(device)

                triplet = criterion(embeddings, labels)
                id_loss = id_criterion(classifier(embeddings), labels)
                loss = triplet + id_loss_weight * id_loss

                loss.backward()
                optimizer.step()

                train_loss += triplet.item()
                epoch_id_loss += id_loss.item()
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

            # Tracker training gets no validation split unless one is given
            # explicitly, so test_loader is usually empty and avg_val_loss is a
            # constant zero. Selecting on that pins best_model to epoch 1 and
            # silently throws away every later epoch, so fall back to the
            # training loss whenever there is nothing to validate against.
            selection_loss = avg_val_loss if num_val_batches else avg_train_loss

            avg_id_loss = epoch_id_loss / max(num_batches, 1)

            with torch.no_grad():
                spread = float(embeddings.std(0).mean())

            print(f"Epoch {epoch+1}/{max_epochs}: train_loss={avg_train_loss:.4f}, "
                  f"id_loss={avg_id_loss:.4f}, embed_spread={spread:.5f}, "
                  f"val_loss={avg_val_loss:.4f}")

            if spread < 1e-4:
                print("  Warning: embeddings have collapsed to a single point; "
                      "the resulting model cannot discriminate")

            # Save checkpoint
            checkpoint_path = snapshot_dir / f"checkpoint_e{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

            if selection_loss < best_loss:
                best_loss = selection_loss
                best_path = snapshot_dir / "best_model.pth"
                torch.save(model.state_dict(), best_path)

        # Get output map with model file
        return self._get_output_map(snapshot_dir)

    def _get_output_map(self, snapshot_dir):
        """Build output map with template replacements and file copies.

        Returns:
            dict: Map where keys with '[-' and '-]' are template replacements,
                  other keys are file copies (key=output filename, value=source path)
        """
        output = {}
        best_model = snapshot_dir / "best_model.pth"

        if best_model.exists():
            output_model_name = "deepsort_reid.pth"
            algo = "deepsort"

            output["type"] = algo

            # Config keys matching deepsort_tracker inference config
            output[algo + ":model_path"] = output_model_name

            # File copies (key=output filename, value=source path)
            output[output_model_name] = str(best_model)

            print(f"Model found at {best_model}")
        else:
            print("Warning: No best model found")

        return output


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
