# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
MOTR-style track-query transformer training implementation.

Trains the learned association model used by the motr track_objects
implementation (see motr_tracker.py) end-to-end on groundtruth tracks:

1. Groundtruth tracks are sampled into clips of consecutive frames.
2. Per frame, detection proposals are simulated from the groundtruth
   boxes with jitter, random dropout (missed detections) and injected
   false positives, following the MOTRv2 proposal-based training setup.
3. Track queries are maintained per groundtruth identity across the clip
   (teacher forcing) and the association head is supervised with
   cross-entropy: each existing track query must select its own identity's
   detection, or the no-match bin when that identity was dropped or left.

The trainer produces a checkpoint consumed by the motr tracker via its
"model_path" configuration value.
"""

import os
import random
from pathlib import Path

import numpy as np

from kwiver.vital.algo import TrainTracker

from viame.pytorch.utilities import report_cuda_errors


class MOTRTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for MOTR-style tracker training.

    Trains a track-query transformer association model from groundtruth
    tracks.
    """

    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-motr-tracker"
        self._train_directory = "deep_training"
        self._gpu_count = -1
        self._timeout = "604800"

        # Model configuration
        self._d_model = "256"
        self._nhead = "8"
        self._num_layers = "3"
        self._crop_size = "64"

        # Training configuration
        self._max_epochs = "40"
        self._batch_size = "8"
        self._learning_rate = "1e-4"
        self._seed = "6969"

        # Clip sampling configuration
        self._clip_length = "10"
        self._clip_stride = "5"
        self._max_clips = "2000"

        # Detection proposal simulation
        self._box_jitter = "0.05"
        self._det_dropout = "0.15"
        self._fp_rate = "0.1"

        # Internal state
        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration(self):
        cfg = super(TrainTracker, self).get_configuration()

        cfg.set_value("identifier", self._identifier)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("gpu_count", str(self._gpu_count))
        cfg.set_value("timeout", self._timeout)
        cfg.set_value("d_model", self._d_model)
        cfg.set_value("nhead", self._nhead)
        cfg.set_value("num_layers", self._num_layers)
        cfg.set_value("crop_size", self._crop_size)
        cfg.set_value("max_epochs", self._max_epochs)
        cfg.set_value("batch_size", self._batch_size)
        cfg.set_value("learning_rate", self._learning_rate)
        cfg.set_value("seed", self._seed)
        cfg.set_value("clip_length", self._clip_length)
        cfg.set_value("clip_stride", self._clip_stride)
        cfg.set_value("max_clips", self._max_clips)
        cfg.set_value("box_jitter", self._box_jitter)
        cfg.set_value("det_dropout", self._det_dropout)
        cfg.set_value("fp_rate", self._fp_rate)

        return cfg

    @report_cuda_errors("MOTRTrainer initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._gpu_count = int(cfg.get_value("gpu_count"))
        self._timeout = str(cfg.get_value("timeout"))
        self._d_model = str(cfg.get_value("d_model"))
        self._nhead = str(cfg.get_value("nhead"))
        self._num_layers = str(cfg.get_value("num_layers"))
        self._crop_size = str(cfg.get_value("crop_size"))
        self._max_epochs = str(cfg.get_value("max_epochs"))
        self._batch_size = str(cfg.get_value("batch_size"))
        self._learning_rate = str(cfg.get_value("learning_rate"))
        self._seed = str(cfg.get_value("seed"))
        self._clip_length = str(cfg.get_value("clip_length"))
        self._clip_stride = str(cfg.get_value("clip_stride"))
        self._max_clips = str(cfg.get_value("max_clips"))
        self._box_jitter = str(cfg.get_value("box_jitter"))
        self._det_dropout = str(cfg.get_value("det_dropout"))
        self._fp_rate = str(cfg.get_value("fp_rate"))

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

    def _extract_clips(self, image_files, track_sets):
        """
        Sample clips of consecutive annotated frames.

        Each clip is a list of frames; each frame is a dict with keys
        'image' (path) and 'objects' (list of (identity, tlbr) tuples).
        Frame IDs index into the split's image file list.
        """
        clip_length = int(self._clip_length)
        clip_stride = int(self._clip_stride)
        max_clips = int(self._max_clips)

        clips = []

        for seq_idx, track_set in enumerate(track_sets):
            if track_set is None:
                continue

            # frame_id -> list of (identity, tlbr)
            frame_objects = {}
            for track in track_set.tracks():
                identity = f"seq{seq_idx}_trk{track.id}"
                for state in track:
                    det = state.detection()
                    if det is None:
                        continue
                    frame_id = state.frame_id
                    if frame_id >= len(image_files):
                        continue
                    bbox = det.bounding_box
                    tlbr = [bbox.min_x(), bbox.min_y(),
                            bbox.max_x(), bbox.max_y()]
                    if tlbr[2] - tlbr[0] <= 0 or tlbr[3] - tlbr[1] <= 0:
                        continue
                    frame_objects.setdefault(frame_id, []).append(
                        (identity, tlbr))

            if not frame_objects:
                continue

            frames_sorted = sorted(frame_objects.keys())

            for start_pos in range(0, len(frames_sorted), clip_stride):
                window = frames_sorted[start_pos:start_pos + clip_length]
                if len(window) < 2:
                    break

                clip = []
                for frame_id in window:
                    img_path = image_files[frame_id]
                    if not os.path.exists(img_path):
                        continue
                    clip.append({
                        'image': img_path,
                        'objects': frame_objects[frame_id],
                    })

                if len(clip) >= 2:
                    clips.append(clip)

                if len(clips) >= max_clips:
                    return clips

        return clips

    def _simulate_detections(self, objects, img_w, img_h, rng,
                             augment=True):
        """
        Turn groundtruth objects into simulated detector proposals.

        Returns (identities, boxes, scores) where dropped groundtruth
        objects are omitted and injected false positives have identity
        None.
        """
        jitter = float(self._box_jitter)
        dropout = float(self._det_dropout)
        fp_rate = float(self._fp_rate)

        identities = []
        boxes = []
        scores = []

        for identity, tlbr in objects:
            if augment and rng.random() < dropout:
                continue

            x1, y1, x2, y2 = tlbr
            w = x2 - x1
            h = y2 - y1

            if augment:
                x1 += rng.gauss(0, jitter) * w
                y1 += rng.gauss(0, jitter) * h
                x2 += rng.gauss(0, jitter) * w
                y2 += rng.gauss(0, jitter) * h

            x1 = max(0.0, min(x1, img_w - 2))
            y1 = max(0.0, min(y1, img_h - 2))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            identities.append(identity)
            boxes.append([x1, y1, x2, y2])
            scores.append(rng.uniform(0.5, 1.0) if augment else 1.0)

        if augment and boxes:
            avg_w = np.mean([b[2] - b[0] for b in boxes])
            avg_h = np.mean([b[3] - b[1] for b in boxes])
            for _ in range(len(boxes)):
                if rng.random() < fp_rate:
                    fx = rng.uniform(0, max(img_w - avg_w, 1))
                    fy = rng.uniform(0, max(img_h - avg_h, 1))
                    identities.append(None)
                    boxes.append([fx, fy, fx + avg_w, fy + avg_h])
                    scores.append(rng.uniform(0.1, 0.7))

        return identities, boxes, scores

    def _run_clip(self, model, clip, device, rng, augment):
        """
        Run one clip through the model with teacher-forced track queries.

        Returns (total loss tensor, number of supervised rows).
        """
        import cv2
        import torch
        import torch.nn.functional as F

        from viame.pytorch.motr_tracker import (
            crop_detections, boxes_tlbr_to_norm_cxcywh)

        crop_size = int(self._crop_size)

        queries = {}       # identity -> query tensor
        last_boxes = {}    # identity -> normalized cxcywh
        last_frames = {}   # identity -> frame index within clip

        total_loss = 0
        supervised = 0

        for t, frame in enumerate(clip):
            img = cv2.imread(frame['image'])
            if img is None:
                continue
            # Runtime images arrive RGB from ImageContainer.asarray()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]

            identities, boxes, scores = self._simulate_detections(
                frame['objects'], img_w, img_h, rng, augment=augment)

            D = len(boxes)

            det_tokens = None
            boxes_norm_t = None
            if D > 0:
                crops = crop_detections(img, boxes, crop_size)
                crops_t = torch.from_numpy(crops).to(device)
                boxes_norm = boxes_tlbr_to_norm_cxcywh(
                    boxes, img_w, img_h)
                boxes_norm_t = torch.from_numpy(boxes_norm).to(device)
                scores_t = torch.tensor(
                    scores, dtype=torch.float32, device=device)
                det_tokens = model.encode_detections(
                    crops_t, boxes_norm_t, scores_t)

            # Supervise association of existing queries
            active = list(queries.keys())
            if len(active) > 0:
                query_stack = torch.stack([queries[k] for k in active])

                if D > 0:
                    refined = model.decode_queries(query_stack, det_tokens)
                else:
                    refined = query_stack

                qboxes = torch.stack(
                    [last_boxes[k] for k in active]).to(device)
                dt = torch.tensor(
                    [t - last_frames[k] for k in active], device=device)

                if D > 0:
                    logits = model.association_logits(
                        refined, qboxes, det_tokens, boxes_norm_t, dt)
                else:
                    logits = model.association_logits(
                        refined, qboxes,
                        torch.zeros((0, model.d_model), device=device),
                        torch.zeros((0, 4), device=device), dt)

                targets = []
                for k in active:
                    if k in identities:
                        targets.append(identities.index(k))
                    else:
                        targets.append(logits.shape[1] - 1)  # no-match bin

                targets_t = torch.tensor(targets, device=device)
                total_loss = total_loss + F.cross_entropy(
                    logits, targets_t, reduction='sum')
                supervised += len(active)

                # Teacher forcing: update queries with their true tokens
                for row, k in enumerate(active):
                    if k in identities:
                        col = identities.index(k)
                        queries[k] = model.update_query(
                            refined[row], det_tokens[col])
                        last_boxes[k] = boxes_norm_t[col].detach()
                        last_frames[k] = t

            # Spawn queries for identities appearing this frame
            for col, k in enumerate(identities):
                if k is None or k in queries:
                    continue
                queries[k] = model.init_query(det_tokens[col])
                last_boxes[k] = boxes_norm_t[col].detach()
                last_frames[k] = t

        return total_loss, supervised

    @report_cuda_errors("MOTRTrainer training")
    def update_model(self):
        """Main training entry point."""
        import torch
        import torch.optim as optim

        from viame.pytorch.motr_tracker import build_track_query_model

        print("Starting MOTR track-query transformer training...")

        rng = random.Random(int(self._seed))
        torch.manual_seed(int(self._seed))

        train_clips = self._extract_clips(
            self._train_image_files, self._train_tracks)
        val_clips = self._extract_clips(
            self._test_image_files, self._test_tracks)

        print(f"  Training clips: {len(train_clips)}")
        print(f"  Validation clips: {len(val_clips)}")

        if len(train_clips) == 0:
            print("No training clips could be sampled from track data!")
            return {}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model_config = {
            'd_model': int(self._d_model),
            'nhead': int(self._nhead),
            'num_layers': int(self._num_layers),
            'crop_size': int(self._crop_size),
        }
        model = build_track_query_model(**model_config).to(device)

        trainable = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {trainable:,}")

        max_epochs = int(self._max_epochs)
        batch_size = int(self._batch_size)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(self._learning_rate),
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs)

        checkpoint_dir = Path(self._train_directory) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        best_val_loss = float('inf')

        for epoch in range(max_epochs):
            model.train()
            rng.shuffle(train_clips)

            train_loss = 0
            train_rows = 0
            optimizer.zero_grad()

            for clip_idx, clip in enumerate(train_clips):
                loss, supervised = self._run_clip(
                    model, clip, device, rng, augment=True)

                if supervised == 0:
                    continue

                loss = loss / supervised
                (loss / batch_size).backward()

                # Step once every batch_size clips (gradient accumulation)
                if (clip_idx + 1) % batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                train_rows += 1

            scheduler.step()
            avg_train_loss = train_loss / max(train_rows, 1)

            model.eval()
            val_loss = 0
            val_rows = 0
            with torch.no_grad():
                for clip in val_clips:
                    loss, supervised = self._run_clip(
                        model, clip, device, rng, augment=False)
                    if supervised == 0:
                        continue
                    val_loss += (loss / supervised).item()
                    val_rows += 1

            avg_val_loss = val_loss / max(val_rows, 1)

            print(f"Epoch {epoch+1}/{max_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint,
                       checkpoint_dir / f"checkpoint_e{epoch+1}.pth")

            # With no validation clips fall back to tracking train loss
            selection_loss = avg_val_loss if val_rows > 0 \
                else avg_train_loss
            if selection_loss < best_val_loss:
                best_val_loss = selection_loss
                torch.save(
                    {'model_state_dict': model.state_dict(),
                     'config': model_config},
                    checkpoint_dir / "best_model.pth")

        output = self._get_output_map()

        print("\nMOTR training complete!")

        return output

    def _get_output_map(self):
        """Build output map for process_trainer_output."""
        output = {}
        output_model_name = "motr_tracker.pth"

        checkpoint_dir = Path(self._train_directory) / "checkpoints"
        best_model = checkpoint_dir / "best_model.pth"

        if not best_model.exists():
            print("\n[MOTRTrainer] No best model found, "
                  "training may have failed")
            return output

        algo = "motr"

        output["type"] = algo
        output[algo + ":model_path"] = output_model_name
        output[output_model_name] = str(best_model)

        print(f"\nModel found at: {best_model}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "motr"

    if algorithm_factory.has_algorithm_impl_name(
            MOTRTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "MOTR-style track-query transformer training for learned association",
        MOTRTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
