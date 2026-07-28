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

from kwiver.vital.algo import TrainTracker

from kwiver.vital.types import (
    CategoryHierarchy,
    ObjectTrackSet, ObjectTrackState,
    BoundingBoxD, DetectedObjectType
)

from distutils.util import strtobool
from pathlib import Path

import os
import sys
import shutil
import json
import random
import numpy as np
from viame.pytorch.utilities import report_cuda_errors
from viame.core.training_data import (build_sequence_maps,
    load_computed_detections, match_to_groundtruth)


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

    def __init__(self, labels, p, k, num_batches=None, same_sequence=0.7,
                 names=None):
        """
        Args:
            labels: the dataset's per sample label, an integer index
            names: label index -> identity name, as ReIDDataset.label_to_idx
                holds it the other way round. Without it the sampler cannot
                tell which clip an identity came from and simply draws
                globally, which is what it always did.
        """
        self.k = max(int(k), 2)
        self.same_sequence = same_sequence

        self.by_id = {}
        for idx, label in enumerate(labels):
            self.by_id.setdefault(label, []).append(idx)

        # A track with a single crop can never supply a positive pair
        self.ids = [i for i, idxs in self.by_id.items() if len(idxs) >= 2]
        self.p = max(min(int(p), len(self.ids)), 1)

        # Identities grouped by the clip they came from. Names are written
        # seq{seq:04d}_track{id:06d}, so the clip is the part before _track.
        self.by_sequence = {}

        if names:
            for identity in self.ids:
                name = names.get(identity)

                if name is None:
                    continue

                sequence = str(name).split("_track")[0]
                self.by_sequence.setdefault(sequence, []).append(identity)

        # Only clips that can fill a batch on their own are worth drawing
        # from, otherwise the batch is mostly topped up from elsewhere and
        # the point is lost
        self.rich_sequences = [s for s, ids in self.by_sequence.items()
                               if len(ids) >= self.p]

        if num_batches is None:
            num_batches = max(len(labels) // (self.p * self.k), 1)
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def _pick_identities(self):
        """The P identities for one batch.

        Drawn from a single clip most of the time. Sampling identities
        uniformly puts each one in a batch with fish from other clips, other
        water and other lighting, and batch-hard mining will happily satisfy
        the margin on those cues rather than on what the fish looks like. A
        negative from the same clip is the one that forces an appearance
        comparison. The rest of the time the draw is global, so the embedding
        still has to separate identities across clips.
        """
        if self.rich_sequences and random.random() < self.same_sequence:
            sequence = random.choice(self.rich_sequences)
            return random.sample(self.by_sequence[sequence], self.p)

        return random.sample(self.ids, self.p)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for track_id in self._pick_identities():
                pool = self.by_id[track_id]
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    # Short track, repeat crops to fill its slot
                    batch.extend(random.choices(pool, k=self.k))
            yield batch


class BoTSORTTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for BoT-SORT training.

    Trains Re-ID model and estimates tracking parameters.
    """
    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-botsort-tracker"

        # Directory of detector output for the same clips, one VIAME
        # CSV per clip. Optional: left empty everything comes from the
        # groundtruth exactly as before.
        self._computed_detections = ""
        self._train_directory = "deep_training"
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
        cfg.set_value("computed_detections", self._computed_detections)
        cfg.set_value("train_directory", self._train_directory)
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

    @report_cuda_errors("BoTSORTTrainer initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._computed_detections = str(cfg.get_value("computed_detections"))
        self._train_directory = str(cfg.get_value("train_directory"))
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
                    frame_id = state.frame_id
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box
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

        # One image map per sequence. A frame id is a position within its
        # own sequence, so resolving it against the flat list of every
        # sequence's images only ever worked for the first one.
        train_maps, train_names = build_sequence_maps(
            self._train_image_files, len(self._train_tracks), "training"
        )

        train_count = self._process_split_data(
            self._train_tracks, train_maps, train_names, train_dir, crop_h, crop_w
        )

        test_maps, test_names = build_sequence_maps(
            self._test_image_files, len(self._test_tracks), "validation"
        )

        test_count = self._process_split_data(
            self._test_tracks, test_maps, test_names, test_dir, crop_h, crop_w
        )

        print(f"  Train: {train_count} crops")
        print(f"  Test: {test_count} crops")

        return reid_dir

    def _load_computed_by_sequence(self, image_maps, names, track_sets):
        """Detector output per sequence, keyed by track set index."""
        if not self._computed_detections:
            return None

        if not names or all(n is None for n in names):
            print("WARNING: computed_detections was given but the images "
                  "could not be split per sequence, so there is no clip name "
                  "to look a detection file up by. Using the groundtruth "
                  "boxes.")
            return None

        loaded = {}

        for seq_idx, name in enumerate(names):
            if name is None or seq_idx >= len(track_sets):
                continue

            detections = load_computed_detections(self._computed_detections,
                                                  name)

            if detections:
                loaded[seq_idx] = detections

        print(f"  computed detections found for {len(loaded)} of "
              f"{len(track_sets)} sequences")

        return loaded or None

    @staticmethod
    def _substitute_computed(frame_to_detections, computed, counters,
                             iou_threshold=0.5):
        """Swap groundtruth boxes for the detector boxes that matched them.

        A groundtruth box no detection reached is dropped rather than kept:
        keeping it would put a perfectly framed crop back into a set that is
        meant to look like detector output. A detection matching nothing is
        dropped too, having no identity to belong to.
        """
        replaced = {}

        for frame_id, truth in frame_to_detections.items():
            frame_computed = computed.get(frame_id, [])

            if not frame_computed:
                counters['frames_without'] += 1
                continue

            matches, unmatched, missed = match_to_groundtruth(
                frame_computed,
                [(*d['bbox'], d['track_id']) for d in truth],
                iou_threshold)

            counters['matched'] += len(matches)
            counters['false_positives'] += len(unmatched)
            counters['missed'] += len(missed)

            rows = []

            for c, t, _overlap in matches:
                rows.append({
                    'track_id': t[4],
                    'bbox': (int(c[0]), int(c[1]), int(c[2]), int(c[3])),
                    'frame_id': frame_id,
                })

            if rows:
                replaced[frame_id] = rows

        return replaced

    def _process_split_data(self, track_sets, image_maps, names, output_dir, crop_h, crop_w):
        """Process tracks for one split."""
        import cv2

        total_crops = 0

        computed_by_sequence = self._load_computed_by_sequence(
            image_maps, names, track_sets)
        counters = {'matched': 0, 'false_positives': 0, 'missed': 0,
                    'frames_without': 0}

        for seq_idx, track_set in enumerate(track_sets):
            if track_set is None:
                continue

            image_map = image_maps[seq_idx]

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

            # Where a detector's own output is supplied, crop its boxes
            # instead of the groundtruth's, taking the identity from the
            # groundtruth box each one matched. Re-ID at inference sees crops
            # cut by the detector, framed and padded however it frames them;
            # trained on exact truth boxes the embedding never meets that.
            if computed_by_sequence and seq_idx in computed_by_sequence:
                frame_to_detections = self._substitute_computed(
                    frame_to_detections, computed_by_sequence[seq_idx],
                    counters)

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

        if computed_by_sequence:
            print(f"  computed boxes: {counters['matched']} matched a "
                  f"groundtruth track, {counters['false_positives']} matched "
                  f"nothing and were dropped, {counters['missed']} truth "
                  f"boxes were not found")

        return total_crops

    @report_cuda_errors("BoTSORTTrainer training")
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

        output = self._get_output_map( params )

        print("\nBoT-SORT training complete!")

        return output

    def _train_reid_model(self, reid_dir):
        """Train Re-ID model using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader, Sampler
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

        # Batches are drawn identity-aware rather than uniformly at random, so
        # that every batch contains valid triplets. See PKSampler.
        crops_per_id = 4
        train_sampler = PKSampler(train_dataset.labels,
                                  max(batch_size // crops_per_id, 1),
                                  crops_per_id,
                                  names={v: k for k, v
                                         in train_dataset.label_to_idx.items()})

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

        best_loss = float('inf')
        snapshot_dir = Path(self._train_directory) / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)

        for epoch in range(int(max_epochs)):
            model.train()
            classifier.train()
            train_loss = 0
            epoch_id_loss = 0
            num_batches = 0

            for images, labels in train_loader:
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

            if selection_loss < best_loss:
                best_loss = selection_loss
                torch.save(model.state_dict(), snapshot_dir / "best_model.pth")

        print("Re-ID model training complete.")

    def _get_output_map( self, params ):
        """Build output map for process_trainer_output."""
        output = {}

        algo = "botsort"
        output["type"] = algo

        # Save params JSON to train directory
        params_name = "botsort_params.json"
        params_file = os.path.join( self._train_directory, params_name )
        with open( params_file, 'w' ) as f:
            json.dump( params, f, indent=2 )

        output[algo + ":params_file"] = params_name
        output[params_name] = params_file

        # Include Re-ID model if it exists
        if self._use_reid:
            snapshot_dir = Path( self._train_directory ) / "snapshot"
            reid_model = snapshot_dir / "best_model.pth"
            if reid_model.exists():
                output[algo + ":model_path"] = "botsort_reid.pth"
                output["botsort_reid.pth"] = str( reid_model )

        print( f"\nThe {self._train_directory} directory can now be deleted, "
               "unless you want to review training metrics first." )

        return output


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
