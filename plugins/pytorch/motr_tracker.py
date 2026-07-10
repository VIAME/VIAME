# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
MOTR-style track-query transformer multi-object tracker.

End-to-end learned association in the spirit of MOTRv2 / MeMOTR: each
active track is represented by a persistent "track query" embedding which
is updated every frame by a transformer decoder attending over the current
frame's detection tokens. Association between tracks and detections is
predicted by the network (appearance + geometry + motion context) rather
than by hand-crafted IoU/Kalman rules, which is particularly effective
when instances are nearly identical in appearance (schooling fish) and
appearance Re-ID fails.

Following MOTRv2, detections come from an external detector (e.g. RF-DETR
via a detector process upstream in the pipeline) and act as proposals;
the transformer only learns the association and track memory update.

Track queries are updated with a GRU cell (MeMOT/MeMOTR-style memory)
using the matched detection's token, so a track's embedding accumulates
appearance and motion history over time.

References:
  Zeng et al., "MOTR: End-to-End Multiple-Object Tracking with
  TRansformer" (ECCV 2022)
  Zhang et al., "MOTRv2: Bootstrapping End-to-End Multi-Object Tracking
  by Pretrained Object Detectors" (CVPR 2023)
  Gao & Wang, "MeMOTR: Long-Term Memory-Augmented Transformer for
  Multi-Object Tracking" (ICCV 2023)

This implementation uses the vital track_objects algorithm interface;
the matching motr train_tracker implementation produces the model file.
"""

import logging
import os

import numpy as np
import scipy.optimize
import scriptconfig as scfg

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import ObjectTrackSet

from viame.pytorch.botsort_tracker import (
    to_DetectedObject_list, get_DetectedObject_bbox_tlbr,
    get_DetectedObject_score, to_ObjectTrackSet,
)
from viame.pytorch.utilities import report_cuda_errors

logger = logging.getLogger(__name__)


# =============================================================================
# Model
# =============================================================================

def build_track_query_model(d_model=256, nhead=8, num_layers=3,
                            crop_size=64, dropout=0.1):
    """
    Build the track-query transformer association model.

    Defined as a function returning an nn.Module subclass instance so that
    torch is only imported when the model is actually needed.
    """
    import torch
    import torch.nn as nn

    class TrackQueryModel(nn.Module):
        """
        Track-query transformer for learned multi-object association.

        Components:
        - appearance encoder: small CNN over per-detection image crops
        - box encoder: MLP over normalized box geometry + score
        - transformer decoder: track queries attend to detection tokens
          (and to each other via self-attention)
        - association head: scaled dot-product between decoded queries and
          detection tokens, plus a pairwise geometry/motion bias and a
          learned per-track no-match logit
        - GRU memory: matched detection tokens update track queries
        """

        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.num_layers = num_layers
            self.crop_size = crop_size

            # Appearance encoder (trained from scratch; no downloads)
            self.appearance = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.appearance_proj = nn.Linear(256, d_model)

            # Box geometry encoder: (cx, cy, w, h, score) normalized
            self.box_encoder = nn.Sequential(
                nn.Linear(5, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )
            self.det_norm = nn.LayerNorm(d_model)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_layers)

            # Association head
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.no_match = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1),
            )
            # Pairwise geometry/motion bias:
            # (dx/w, dy/h, log w-ratio, log h-ratio, iou, log dt)
            self.geom_bias = nn.Sequential(
                nn.Linear(6, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

            # Query lifecycle
            self.query_init = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )
            self.query_update = nn.GRUCell(d_model, d_model)

        def encode_detections(self, crops, boxes_norm, scores):
            """
            Encode per-frame detections into tokens.

            Args:
                crops: [D, 3, crop, crop] float tensor (0-1 RGB) or None
                boxes_norm: [D, 4] (cx, cy, w, h) in 0-1 image coordinates
                scores: [D] detection confidences

            Returns:
                [D, d_model] detection tokens
            """
            geom = torch.cat([boxes_norm, scores.unsqueeze(-1)], dim=-1)
            tokens = self.box_encoder(geom)
            if crops is not None:
                app = self.appearance(crops).flatten(1)
                tokens = tokens + self.appearance_proj(app)
            return self.det_norm(tokens)

        def decode_queries(self, queries, det_tokens):
            """
            Refine track queries against the frame's detection tokens.

            Args:
                queries: [T, d_model]
                det_tokens: [D, d_model]

            Returns:
                [T, d_model] refined queries
            """
            if det_tokens.shape[0] == 0:
                return queries
            return self.decoder(
                queries.unsqueeze(0), det_tokens.unsqueeze(0)
            ).squeeze(0)

        def association_logits(self, queries, query_boxes_norm,
                               det_tokens, det_boxes_norm, dt):
            """
            Compute association logits between tracks and detections.

            Args:
                queries: [T, d_model] refined track queries
                query_boxes_norm: [T, 4] last observed box per track
                det_tokens: [D, d_model]
                det_boxes_norm: [D, 4]
                dt: [T] frames since each track's last observation

            Returns:
                [T, D + 1] logits; the final column is the no-match bin
            """
            T = queries.shape[0]
            D = det_tokens.shape[0]

            no_match = self.no_match(queries)  # [T, 1]

            if D == 0:
                return no_match

            q = self.q_proj(queries)
            k = self.k_proj(det_tokens)
            sim = q @ k.t() / (self.d_model ** 0.5)  # [T, D]

            # Pairwise geometry features
            qb = query_boxes_norm.unsqueeze(1).expand(T, D, 4)
            db = det_boxes_norm.unsqueeze(0).expand(T, D, 4)
            eps = 1e-6
            dx = (db[..., 0] - qb[..., 0]) / (qb[..., 2] + eps)
            dy = (db[..., 1] - qb[..., 1]) / (qb[..., 3] + eps)
            dw = torch.log((db[..., 2] + eps) / (qb[..., 2] + eps))
            dh = torch.log((db[..., 3] + eps) / (qb[..., 3] + eps))
            iou = self._box_iou_cxcywh(qb, db)
            dtf = torch.log(dt.float() + 1.0)
            dtf = dtf.unsqueeze(1).expand(T, D)

            geom = torch.stack([dx, dy, dw, dh, iou, dtf], dim=-1)
            bias = self.geom_bias(geom).squeeze(-1)  # [T, D]

            return torch.cat([sim + bias, no_match], dim=1)

        @staticmethod
        def _box_iou_cxcywh(a, b):
            """IoU between [..., 4] cxcywh box tensors."""
            ax1 = a[..., 0] - a[..., 2] / 2
            ay1 = a[..., 1] - a[..., 3] / 2
            ax2 = a[..., 0] + a[..., 2] / 2
            ay2 = a[..., 1] + a[..., 3] / 2
            bx1 = b[..., 0] - b[..., 2] / 2
            by1 = b[..., 1] - b[..., 3] / 2
            bx2 = b[..., 0] + b[..., 2] / 2
            by2 = b[..., 1] + b[..., 3] / 2
            ix1 = torch.maximum(ax1, bx1)
            iy1 = torch.maximum(ay1, by1)
            ix2 = torch.minimum(ax2, bx2)
            iy2 = torch.minimum(ay2, by2)
            iw = (ix2 - ix1).clamp(min=0)
            ih = (iy2 - iy1).clamp(min=0)
            inter = iw * ih
            union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter
            return inter / union.clamp(min=1e-6)

        def init_query(self, det_token):
            """Spawn a new track query from a detection token."""
            return self.query_init(det_token)

        def update_query(self, query, det_token):
            """GRU memory update of a track query with its matched token."""
            return self.query_update(
                det_token.unsqueeze(0), query.unsqueeze(0)).squeeze(0)

    return TrackQueryModel()


def crop_detections(np_image, boxes_tlbr, crop_size):
    """
    Crop and resize detection regions from an RGB/BGR image.

    Returns a [D, 3, crop, crop] float32 numpy array scaled to 0-1, or
    None when no image is available.
    """
    import cv2

    if np_image is None:
        return None

    img_h, img_w = np_image.shape[:2]
    crops = []
    for box in boxes_tlbr:
        x1 = int(max(0, min(box[0], img_w - 1)))
        y1 = int(max(0, min(box[1], img_h - 1)))
        x2 = int(max(x1 + 1, min(box[2], img_w)))
        y2 = int(max(y1 + 1, min(box[3], img_h)))

        crop = np_image[y1:y2, x1:x2]
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)
        crop = cv2.resize(crop, (crop_size, crop_size))
        crops.append(crop.astype(np.float32).transpose(2, 0, 1) / 255.0)

    return np.stack(crops) if crops else None


def boxes_tlbr_to_norm_cxcywh(boxes_tlbr, img_w, img_h):
    """Convert [D, 4] tlbr pixel boxes to normalized cxcywh."""
    boxes = np.asarray(boxes_tlbr, dtype=np.float32).reshape(-1, 4)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / img_w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / img_h
    w = (boxes[:, 2] - boxes[:, 0]) / img_w
    h = (boxes[:, 3] - boxes[:, 1]) / img_h
    return np.stack([cx, cy, w, h], axis=1)


# =============================================================================
# Configuration
# =============================================================================

class MOTRTrackerConfig(scfg.DataConfig):
    """Configuration for the MOTR-style track-query tracker."""
    model_path = scfg.Value('', help='Path to trained track-query model checkpoint')
    device = scfg.Value('cuda', help='Device to run the model on (cuda, cpu, auto)')
    det_thresh = scfg.Value(0.1, help='Minimum detection confidence to consider at all')
    match_thresh = scfg.Value(0.5, help='Minimum association probability to accept a match')
    new_track_thresh = scfg.Value(0.6, help='Minimum detection confidence to start a new track')
    track_buffer = scfg.Value(30, help='Frames a track survives without a match')
    min_hits = scfg.Value(1, help='Number of associations before a track is output')


# =============================================================================
# Runtime tracker
# =============================================================================

class _QueryTrack:
    """Runtime state for one tracked object."""

    _count = 0

    def __init__(self, query, box_tlbr, score, frame_id, timestamp,
                 detected_object):
        _QueryTrack._count += 1
        self.track_id = _QueryTrack._count
        self.query = query
        self.box_tlbr = np.asarray(box_tlbr, dtype=np.float64)
        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits = 1
        self.history = []
        if detected_object is not None and timestamp is not None:
            self.history.append((timestamp, detected_object))

    @staticmethod
    def reset_id():
        _QueryTrack._count = 0


class MOTRTracker(TrackObjects):
    """
    MOTR-style track-query transformer tracker.

    Consumes detections from an upstream detector and associates them
    across frames with a learned transformer association model.
    """

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = MOTRTrackerConfig()
        self._model = None
        self._device = None
        self._crop_size = 64
        self._tracks = []
        self._frame_id = 0

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("MOTRTracker initialization")
    def set_configuration(self, cfg_in):
        import torch
        from viame.pytorch.utilities import vital_config_update, \
            resolve_device

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        c = self._config
        c.model_path = cfg.get_value('model_path')
        c.device = cfg.get_value('device')
        c.det_thresh = float(cfg.get_value('det_thresh'))
        c.match_thresh = float(cfg.get_value('match_thresh'))
        c.new_track_thresh = float(cfg.get_value('new_track_thresh'))
        c.track_buffer = int(cfg.get_value('track_buffer'))
        c.min_hits = int(cfg.get_value('min_hits'))

        self._device = resolve_device(c.device)

        model_cfg = {}
        state_dict = None
        if c.model_path and os.path.exists(c.model_path):
            checkpoint = torch.load(
                c.model_path, map_location='cpu', weights_only=True)
            model_cfg = checkpoint.get('config', {})
            state_dict = checkpoint.get(
                'model_state_dict', checkpoint)
        else:
            logger.warning(
                "No trained model_path provided for motr tracker; "
                "using randomly initialized association weights")

        self._model = build_track_query_model(
            d_model=int(model_cfg.get('d_model', 256)),
            nhead=int(model_cfg.get('nhead', 8)),
            num_layers=int(model_cfg.get('num_layers', 3)),
            crop_size=int(model_cfg.get('crop_size', 64)),
        )
        self._crop_size = self._model.crop_size

        if state_dict is not None:
            self._model.load_state_dict(state_dict)
            print(f"[MOTR] Loaded track-query model from {c.model_path}")

        self._model = self._model.to(self._device)
        self._model.eval()

        return True

    def check_configuration(self, cfg):
        return True

    @report_cuda_errors("MOTRTracker tracking")
    def track(self, ts, image, detections):
        """Track objects in the current frame."""
        import torch

        self._frame_id += 1
        c = self._config

        np_image = image.asarray() if image is not None else None
        if np_image is not None:
            img_h, img_w = np_image.shape[:2]
        else:
            img_h, img_w = 1, 1

        # Gather detections
        det_objects = []
        det_boxes = []
        det_scores = []
        for do in to_DetectedObject_list(detections):
            score = get_DetectedObject_score(do)
            if score < c.det_thresh:
                continue
            det_objects.append(do)
            det_boxes.append(get_DetectedObject_bbox_tlbr(do))
            det_scores.append(score)

        D = len(det_objects)
        T = len(self._tracks)

        with torch.no_grad():
            det_tokens = None
            det_boxes_norm_t = None
            if D > 0:
                crops = crop_detections(
                    np_image, det_boxes, self._crop_size)
                crops_t = None if crops is None else \
                    torch.from_numpy(crops).to(self._device)
                boxes_norm = boxes_tlbr_to_norm_cxcywh(
                    det_boxes, img_w, img_h)
                det_boxes_norm_t = torch.from_numpy(
                    boxes_norm).to(self._device)
                scores_t = torch.tensor(
                    det_scores, dtype=torch.float32, device=self._device)
                det_tokens = self._model.encode_detections(
                    crops_t, det_boxes_norm_t, scores_t)

            matched_track = {}
            matched_det = set()

            if T > 0 and D > 0:
                queries = torch.stack([t.query for t in self._tracks])
                refined = self._model.decode_queries(queries, det_tokens)

                track_boxes_norm = boxes_tlbr_to_norm_cxcywh(
                    [t.box_tlbr for t in self._tracks], img_w, img_h)
                track_boxes_norm_t = torch.from_numpy(
                    track_boxes_norm).to(self._device)
                dt = torch.tensor(
                    [self._frame_id - t.frame_id for t in self._tracks],
                    device=self._device)

                logits = self._model.association_logits(
                    refined, track_boxes_norm_t,
                    det_tokens, det_boxes_norm_t, dt)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Hungarian on negative probability over real detections
                cost = 1.0 - probs[:, :D]
                rows, cols = scipy.optimize.linear_sum_assignment(cost)
                for r, col in zip(rows, cols):
                    if probs[r, col] >= c.match_thresh:
                        matched_track[r] = col
                        matched_det.add(col)

                # Matched tracks: update memory and state
                for r, col in matched_track.items():
                    trk = self._tracks[r]
                    trk.query = self._model.update_query(
                        refined[r], det_tokens[col])
                    trk.box_tlbr = np.asarray(det_boxes[col])
                    trk.score = det_scores[col]
                    trk.frame_id = self._frame_id
                    trk.time_since_update = 0
                    trk.hits += 1
                    trk.history.append((ts, det_objects[col]))

            # Unmatched tracks age
            for r, trk in enumerate(self._tracks):
                if r not in matched_track:
                    trk.time_since_update += 1

            # New tracks from unmatched confident detections
            for j in range(D):
                if j in matched_det:
                    continue
                if det_scores[j] >= c.new_track_thresh:
                    query = self._model.init_query(det_tokens[j])
                    self._tracks.append(_QueryTrack(
                        query, det_boxes[j], det_scores[j],
                        self._frame_id, ts, det_objects[j]))

        # Drop dead tracks
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= c.track_buffer]

        output = [t for t in self._tracks
                  if t.time_since_update == 0 and t.hits >= c.min_hits
                  and len(t.history) > 0]
        return to_ObjectTrackSet(output)

    @report_cuda_errors("MOTRTracker tracking")
    def initialize(self, ts, image, seed_detections):
        """Initialize tracking with optional seed detections."""
        self.reset()
        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        return ObjectTrackSet([])

    @report_cuda_errors("MOTRTracker finalization")
    def finalize(self):
        """Finalize tracking and return all tracks."""
        output = [t for t in self._tracks if len(t.history) > 0]
        return to_ObjectTrackSet(output)

    def reset(self):
        """Reset tracker state for a new sequence."""
        _QueryTrack.reset_id()
        self._tracks = []
        self._frame_id = 0


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "motr"

    if algorithm_factory.has_algorithm_impl_name(
            MOTRTracker.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "MOTR-style track-query transformer tracker with learned association",
        MOTRTracker
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
