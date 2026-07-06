# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
OC-SORT / Deep OC-SORT tracker training implementation.

OC-SORT training involves:
1. Estimating tracking parameters from groundtruth tracks (Kalman noise,
   confidence thresholds, track buffer, OCM delta_t / velocity-direction
   weight from observed motion nonlinearity).
2. Optionally training a Re-ID model for appearance features
   (Deep OC-SORT), reusing the BoT-SORT Re-ID training recipe.

The trainer produces a parameter JSON file (and optionally a Re-ID model)
consumed by the ocsort track_objects implementation via its "params_file"
and "model_path" configuration values.
"""

import json
import os
from pathlib import Path

import numpy as np

from viame.pytorch.botsort_trainer import BoTSORTTrainer
from viame.pytorch.utilities import report_cuda_errors


class OCSORTTrainer(BoTSORTTrainer):
    """
    Implementation of TrainTracker class for OC-SORT training.

    Estimates observation-centric tracking parameters from groundtruth
    tracks and optionally trains a Re-ID model for Deep OC-SORT.
    """

    def __init__(self):
        super().__init__()

        self._identifier = "viame-ocsort-tracker"

        # OC-SORT is usually run motion-only; enable Re-ID (Deep OC-SORT)
        # explicitly when appearance is discriminative in the target domain
        self._use_reid = False
        self._use_cmc = False

        self._delta_t = "3"
        self._vdc_weight = "0.2"

    def get_configuration(self):
        cfg = super().get_configuration()
        cfg.set_value("delta_t", self._delta_t)
        cfg.set_value("vdc_weight", self._vdc_weight)
        return cfg

    @report_cuda_errors("OCSORTTrainer initialization")
    def set_configuration(self, cfg_in):
        result = super().set_configuration(cfg_in)
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self._delta_t = str(cfg.get_value("delta_t"))
        self._vdc_weight = str(cfg.get_value("vdc_weight"))
        return result

    def _extract_direction_statistics(self):
        """
        Measure motion direction stability from groundtruth tracks.

        Returns per-step angular changes (radians) of the center-motion
        direction, used to decide how much weight the velocity direction
        consistency (OCM) term deserves.
        """
        angular_changes = []

        all_tracks = self._train_tracks + self._test_tracks

        for track_set in all_tracks:
            if track_set is None:
                continue

            for track in track_set.tracks():
                prev_dir = None
                prev_frame = None
                prev_cx, prev_cy = None, None

                for state in track:
                    det = state.detection()
                    if det is None:
                        continue

                    frame_id = state.frame_id
                    bbox = det.bounding_box
                    cx = (bbox.min_x() + bbox.max_x()) / 2.0
                    cy = (bbox.min_y() + bbox.max_y()) / 2.0

                    if prev_cx is not None and frame_id > prev_frame:
                        dx = cx - prev_cx
                        dy = cy - prev_cy
                        norm = np.sqrt(dx * dx + dy * dy)
                        if norm > 1e-3:
                            direction = np.array([dx, dy]) / norm
                            if prev_dir is not None:
                                cos_sim = np.clip(
                                    np.dot(direction, prev_dir), -1.0, 1.0)
                                angular_changes.append(np.arccos(cos_sim))
                            prev_dir = direction

                    prev_frame = frame_id
                    prev_cx, prev_cy = cx, cy

        return angular_changes

    def _estimate_parameters(self, stats):
        """Estimate OC-SORT parameters from groundtruth statistics."""
        params = super()._estimate_parameters(stats)

        # BoT-SORT specific values not used by ocsort
        params.pop('iou_weight', None)

        # OCM velocity window: span the typical annotation gap so velocity
        # is computed from a real prior observation
        gap_lengths = stats['gap_lengths']
        if len(gap_lengths) >= 5:
            delta_t = int(np.clip(
                np.percentile(gap_lengths, 75) + 1, 1, 5))
        else:
            delta_t = int(self._delta_t)
        params['delta_t'] = delta_t

        # OCM weight: direction consistency is only informative when motion
        # direction is stable frame-to-frame. Scale the configured weight
        # down as observed angular volatility grows (erratic fish motion).
        angular_changes = self._extract_direction_statistics()
        base_weight = float(self._vdc_weight)
        if len(angular_changes) >= 10:
            median_change = float(np.median(angular_changes))
            # 0 rad change -> full weight; >= pi/2 typical change -> ~0
            stability = max(0.0, 1.0 - median_change / (np.pi / 2))
            params['vdc_weight'] = float(
                np.clip(base_weight * stability, 0.02, base_weight))
            print(f"  Median direction change: "
                  f"{np.degrees(median_change):.1f} deg "
                  f"(stability {stability:.2f})")
        else:
            params['vdc_weight'] = base_weight

        params['ocr_iou_thresh'] = 0.3
        params['use_byte'] = True

        return params

    def _get_output_map(self, params):
        """Build output map for process_trainer_output."""
        output = {}

        algo = "ocsort"
        output["type"] = algo

        params_name = "ocsort_params.json"
        params_file = os.path.join(self._train_directory, params_name)
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

        output[algo + ":params_file"] = params_name
        output[params_name] = params_file

        # Include Re-ID model if trained (Deep OC-SORT)
        if self._use_reid:
            snapshot_dir = Path(self._train_directory) / "snapshot"
            reid_model = snapshot_dir / "best_model.pth"
            if reid_model.exists():
                output[algo + ":model_path"] = "ocsort_reid.pth"
                output["ocsort_reid.pth"] = str(reid_model)

        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output

    @report_cuda_errors("OCSORTTrainer training")
    def update_model(self):
        """Estimate parameters and optionally train the Re-ID model."""
        print("Starting OC-SORT training...")

        print("Extracting track statistics...")
        stats = self._extract_track_statistics()

        print("Estimating tracking parameters...")
        params = self._estimate_parameters(stats)

        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        if self._use_reid:
            reid_dir = self._prepare_reid_data()
            self._train_reid_model(reid_dir)

        params['use_reid'] = bool(self._use_reid)
        params['use_cmc'] = bool(self._use_cmc)
        params['feat_ema_alpha'] = float(self._feat_ema_alpha)

        output = self._get_output_map(params)

        print("\nOC-SORT training complete!")

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "ocsort"

    if algorithm_factory.has_algorithm_impl_name(
            OCSORTTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "OC-SORT parameter estimation and Deep OC-SORT Re-ID training",
        OCSORTTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
