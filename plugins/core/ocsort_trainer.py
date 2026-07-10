# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OC-SORT / Deep OC-SORT tracker training implementation.

OC-SORT uses Kalman filtering with observation-centric improvements (no
learned components), so "training" primarily estimates optimal parameters
from groundtruth tracks:
- Kalman filter noise weights, from observed motion patterns
- Detection confidence thresholds and track buffer
- Velocity direction consistency (VDC) weight
- Observation-centric velocity window (delta_t)

When ``use_reid`` is enabled (Deep OC-SORT), an appearance Re-ID model is
additionally trained, reusing the BoT-SORT Re-ID training recipe. That
path is imported lazily so a default (motion-only) run pulls in no
torch/pytorch dependency.

The emitted parameters are consumed by the ocsort track_objects
implementation via its ``params_file`` (and, for Deep OC-SORT, its
``model_path``) configuration values.
"""

import os
import json
import numpy as np

from kwiver.vital.algo import TrainTracker


class OCSORTTrainer(TrainTracker):
    """
    Implementation of TrainTracker class for OC-SORT parameter estimation
    and optional Deep OC-SORT Re-ID model training.
    """
    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-ocsort-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "ocsort_tracker"
        self._pipeline_template = ""
        self._threshold = "0.00"
        self._delta_t = "3"

        # Deep OC-SORT (appearance Re-ID) training. Disabled by default;
        # when off, no torch/pytorch code is imported.
        self._use_reid = "false"
        self._crop_size = "128x64"
        self._embedding_dim = "512"
        self._backbone = "resnet18"
        self._max_epochs = "50"
        self._batch_size = "32"
        self._learning_rate = "0.0003"
        self._gpu_count = -1

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
        cfg.set_value("threshold", self._threshold)
        cfg.set_value("delta_t", self._delta_t)
        cfg.set_value("use_reid", self._use_reid)
        cfg.set_value("crop_size", self._crop_size)
        cfg.set_value("embedding_dim", self._embedding_dim)
        cfg.set_value("backbone", self._backbone)
        cfg.set_value("max_epochs", self._max_epochs)
        cfg.set_value("batch_size", self._batch_size)
        cfg.set_value("learning_rate", self._learning_rate)
        cfg.set_value("gpu_count", str(self._gpu_count))

        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._output_directory = str(cfg.get_value("output_directory"))
        self._output_prefix = str(cfg.get_value("output_prefix"))
        self._pipeline_template = str(cfg.get_value("pipeline_template"))
        self._threshold = str(cfg.get_value("threshold"))
        self._delta_t = str(cfg.get_value("delta_t"))
        self._use_reid = str(cfg.get_value("use_reid"))
        self._crop_size = str(cfg.get_value("crop_size"))
        self._embedding_dim = str(cfg.get_value("embedding_dim"))
        self._backbone = str(cfg.get_value("backbone"))
        self._max_epochs = str(cfg.get_value("max_epochs"))
        self._batch_size = str(cfg.get_value("batch_size"))
        self._learning_rate = str(cfg.get_value("learning_rate"))
        self._gpu_count = int(cfg.get_value("gpu_count"))

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

    def _reid_enabled(self):
        return str(self._use_reid).lower() in ('true', '1', 'yes')

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
        """
        Extract statistics including velocity direction changes for VDC tuning.
        """
        positions = []
        velocities = []
        confidences = []
        track_lengths = []
        gap_lengths = []
        direction_changes = []  # Angle changes in velocity direction

        all_tracks = self._train_tracks + self._test_tracks

        for track_set in all_tracks:
            if track_set is None:
                continue

            for track in track_set.tracks():
                states = list(track)
                track_lengths.append(len(states))

                prev_frame = None
                prev_cx, prev_cy = None, None
                prev_vx, prev_vy = None, None

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

                            # Track velocity direction changes
                            if prev_vx is not None and dt == 1:
                                v1 = np.array([prev_vx, prev_vy])
                                v2 = np.array([vx, vy])
                                n1 = np.linalg.norm(v1)
                                n2 = np.linalg.norm(v2)

                                if n1 > 1e-5 and n2 > 1e-5:
                                    cos_angle = np.dot(v1, v2) / (n1 * n2)
                                    cos_angle = np.clip(cos_angle, -1, 1)
                                    angle = np.arccos(cos_angle)
                                    direction_changes.append(angle)

                            if dt > 1:
                                gap_lengths.append(dt - 1)

                            prev_vx, prev_vy = vx, vy

                    prev_frame = frame_id
                    prev_cx, prev_cy = cx, cy

        return {
            'positions': positions,
            'velocities': velocities,
            'confidences': confidences,
            'track_lengths': track_lengths,
            'gap_lengths': gap_lengths,
            'direction_changes': direction_changes
        }

    def _estimate_kalman_parameters(self, stats):
        """Estimate Kalman filter parameters (same as ByteTrack)."""
        velocities = stats['velocities']

        if len(velocities) < 10:
            return 1.0 / 20, 1.0 / 160

        pos_variances = []
        for vx, vy, h, dt in velocities:
            if h > 0 and dt == 1:
                pos_var = np.sqrt(vx**2 + vy**2) / h
                pos_variances.append(pos_var)

        if len(pos_variances) > 0:
            median_pos_var = np.median(pos_variances)
            std_weight_position = np.clip(median_pos_var * 2, 0.01, 0.5)
        else:
            std_weight_position = 1.0 / 20

        vel_variances = []
        prev_vx, prev_vy, prev_h = None, None, None
        for vx, vy, h, dt in velocities:
            if prev_vx is not None and dt == 1 and h > 0:
                ax = vx - prev_vx
                ay = vy - prev_vy
                vel_var = np.sqrt(ax**2 + ay**2) / h
                vel_variances.append(vel_var)
            prev_vx, prev_vy, prev_h = vx, vy, h

        if len(vel_variances) > 0:
            median_vel_var = np.median(vel_variances)
            std_weight_velocity = np.clip(median_vel_var * 2, 0.001, 0.1)
        else:
            std_weight_velocity = 1.0 / 160

        return std_weight_position, std_weight_velocity

    def _estimate_thresholds(self, stats):
        """Estimate detection confidence thresholds."""
        confidences = stats['confidences']

        if len(confidences) < 10:
            return 0.6, 0.1, 0.6

        confidences = np.array(confidences)

        high_thresh = np.percentile(confidences, 30)
        low_thresh = np.percentile(confidences, 10)
        new_track_thresh = high_thresh

        high_thresh = np.clip(high_thresh, 0.3, 0.9)
        low_thresh = np.clip(low_thresh, 0.05, high_thresh - 0.1)
        new_track_thresh = np.clip(new_track_thresh, 0.3, 0.9)

        return float(high_thresh), float(low_thresh), float(new_track_thresh)

    def _estimate_track_buffer(self, stats):
        """Estimate track buffer from gap statistics."""
        gap_lengths = stats['gap_lengths']

        if len(gap_lengths) < 5:
            return 30

        gap_90 = np.percentile(gap_lengths, 90)
        track_buffer = int(gap_90 * 1.5) + 5
        track_buffer = max(10, min(100, track_buffer))

        return track_buffer

    def _estimate_delta_t(self, stats):
        """
        Estimate the observation-centric velocity window (OCM delta_t).

        A window spanning the typical annotation gap keeps velocity
        estimated from a real prior observation across short misses.
        """
        gap_lengths = stats['gap_lengths']
        if len(gap_lengths) < 5:
            return int(self._delta_t)
        return int(np.clip(np.percentile(gap_lengths, 75) + 1, 1, 5))

    def _estimate_vdc_weight(self, stats):
        """
        Estimate velocity direction consistency weight.

        Higher weight is warranted when motion is mostly linear (so a
        direction reversal is strong evidence against an association);
        lower when motion is erratic.
        """
        direction_changes = stats['direction_changes']

        if len(direction_changes) < 10:
            return 0.2  # Default

        direction_changes_deg = np.array(direction_changes) * 180 / np.pi
        mean_change = np.mean(direction_changes_deg)
        std_change = np.std(direction_changes_deg)

        if mean_change < 20:
            vdc_weight = 0.3
        elif mean_change < 45:
            vdc_weight = 0.2
        else:
            vdc_weight = 0.1

        print(f"  Mean direction change: {mean_change:.1f} degrees")
        print(f"  Std direction change: {std_change:.1f} degrees")

        return vdc_weight

    def _train_reid_model(self):
        """
        Train the Deep OC-SORT appearance Re-ID model.

        Reuses the BoT-SORT Re-ID training recipe (imported lazily so torch
        is only pulled in when appearance training is requested). Returns
        the path to the trained model, or None on failure.
        """
        try:
            from viame.pytorch.botsort_trainer import BoTSORTTrainer
        except ImportError as e:
            print(f"[OCSORT] Re-ID training unavailable (pytorch plugin "
                  f"not importable): {e}")
            return None

        print("Training Deep OC-SORT appearance Re-ID model...")

        helper = BoTSORTTrainer()
        helper._train_directory = self._train_directory
        helper._crop_size = self._crop_size
        helper._embedding_dim = self._embedding_dim
        helper._backbone = self._backbone
        helper._max_epochs = self._max_epochs
        helper._batch_size = self._batch_size
        helper._learning_rate = self._learning_rate
        helper._gpu_count = self._gpu_count
        helper._train_image_files = self._train_image_files
        helper._train_tracks = self._train_tracks
        helper._test_image_files = self._test_image_files
        helper._test_tracks = self._test_tracks

        reid_dir = helper._prepare_reid_data()
        helper._train_reid_model(reid_dir)

        model_path = os.path.join(self._train_directory, "snapshot",
                                  "best_model.pth")
        if os.path.exists(model_path):
            return model_path

        print("[OCSORT] Re-ID training produced no model")
        return None

    def update_model(self):
        """
        Analyze track groundtruth and estimate OC-SORT parameters (and,
        for Deep OC-SORT, train the appearance Re-ID model).

        Returns:
            dict: Map of template replacements and file copies
        """
        print("Starting OC-SORT parameter estimation...")

        print("Extracting track statistics...")
        stats = self._extract_track_statistics()

        print(f"  Found {len(stats['positions'])} detections")
        print(f"  Found {len(stats['velocities'])} velocity measurements")
        print(f"  Found {len(stats['track_lengths'])} tracks")
        print(f"  Found {len(stats['direction_changes'])} direction changes")

        print("Estimating Kalman filter parameters...")
        std_weight_position, std_weight_velocity = self._estimate_kalman_parameters(stats)
        print(f"  std_weight_position: {std_weight_position:.6f}")
        print(f"  std_weight_velocity: {std_weight_velocity:.6f}")

        print("Estimating detection thresholds...")
        high_thresh, low_thresh, new_track_thresh = self._estimate_thresholds(stats)
        print(f"  high_thresh: {high_thresh:.3f}")
        print(f"  low_thresh: {low_thresh:.3f}")
        print(f"  new_track_thresh: {new_track_thresh:.3f}")

        print("Estimating track buffer...")
        track_buffer = self._estimate_track_buffer(stats)
        print(f"  track_buffer: {track_buffer}")

        print("Estimating VDC weight...")
        vdc_weight = self._estimate_vdc_weight(stats)
        print(f"  vdc_weight: {vdc_weight:.3f}")

        delta_t = self._estimate_delta_t(stats)
        print(f"  delta_t: {delta_t}")

        use_reid = self._reid_enabled()

        params = {
            'std_weight_position': std_weight_position,
            'std_weight_velocity': std_weight_velocity,
            'high_thresh': high_thresh,
            'low_thresh': low_thresh,
            'match_thresh': 0.8,
            'new_track_thresh': new_track_thresh,
            'track_buffer': track_buffer,
            'delta_t': delta_t,
            'vdc_weight': vdc_weight,
            'ocr_iou_thresh': 0.3,
            'use_vdc': True,
            'use_oru': True,
            'use_byte': True,
            'use_ocr': True,
            'use_reid': use_reid,
        }

        # Save parameter JSON to train directory (copied out by the caller)
        params_file = os.path.join(self._train_directory, "ocsort_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved parameters to {params_file}")

        # Optional Deep OC-SORT appearance model
        reid_model_path = None
        if use_reid:
            reid_model_path = self._train_reid_model()

        output = self._get_output_map(params, params_file, reid_model_path)

        print("\nOC-SORT parameter estimation complete!\n")

        return output

    def _get_output_map(self, params, params_file, reid_model_path=None):
        """Build output map with template replacements and file copies.

        Returns:
            dict: Map where file paths are file copies, other values are
            template replacements.
        """
        output = {}
        algo = "ocsort"

        output["type"] = algo

        # Config keys matching ocsort inference config
        for key, value in params.items():
            if isinstance(value, float):
                output[algo + ":" + key] = f"{value:.3f}"
            else:
                output[algo + ":" + key] = str(value)

        # File copies
        output["ocsort_params.json"] = params_file

        # Deep OC-SORT appearance model, if trained
        if reid_model_path is not None:
            output[algo + ":model_path"] = "ocsort_reid.pth"
            output["ocsort_reid.pth"] = reid_model_path

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "ocsort"

    if algorithm_factory.has_algorithm_impl_name(
        OCSORTTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "OC-SORT parameter estimation and optional Deep OC-SORT Re-ID training",
        OCSORTTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
