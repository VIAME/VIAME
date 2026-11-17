# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Learned-association multi-object tracker.

One assignment pass per frame. The cost of linking a track to a detection is
1 - P(same object), where P comes from a gradient-boosted tree ensemble over
box-geometry and motion features of the pair. The ensemble is read from an
.npz of flattened trees, so only numpy and scipy are needed at run time.
"""

import json
import logging
import os

import numpy as np
import scipy.optimize
import scriptconfig as scfg

from kwiver.vital.algo import TrackObjects
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "dx", "dy", "dist", "dist_per_gap", "res", "res_x", "res_y", "lw", "lh", "la",
    "iou_last", "iou_pred", "iou_buf", "det_score", "last_score", "gap", "nobs",
    "has_vel", "speed", "cosang", "lsize", "aspect_t", "aspect_d",
    "resL", "iou_predL", "speedL", "dvel", "res_min", "gap_x_speed"]


# =============================================================================
# Pair features
# =============================================================================

def track_desc(obs, frame_now):
    """(box, score, vel, has_vel, nobs, gap, vel_long) from [(frame, box, score), ...]."""
    f, b, sc = obs[-1][:3]
    n = len(obs)
    if n > 1:
        pf, pb = obs[-2][:2]
        v = np.array([(b[0] + b[2] - pb[0] - pb[2]) / 2,
                      (b[1] + b[3] - pb[1] - pb[3]) / 2]) / max(f - pf, 1)
        qf, qb = obs[max(0, n - 5)][:2]
        vl = np.array([(b[0] + b[2] - qb[0] - qb[2]) / 2,
                       (b[1] + b[3] - qb[1] - qb[3]) / 2]) / max(f - qf, 1)
    else:
        v = vl = np.zeros(2)
    return b, sc, v, float(n > 1), n, frame_now - f, vl


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _buf(x, r):
    w = (x[:, 2] - x[:, 0]) * r
    h = (x[:, 3] - x[:, 1]) * r
    return np.stack([x[:, 0] - w, x[:, 1] - h, x[:, 2] + w, x[:, 3] + h], 1)


def pair_features(tbox, tscore, tvel, thas, tnobs, tgap, dets, tvelL):
    """(T, D, F) float32 features and the (T, D) mask of plausible pairs.

    tbox (T,4) last observed boxes, tvel/tvelL (T,2) px/frame, tgap (T,) frames
    since the last observation, dets (D,5) x1,y1,x2,y2,score."""
    T, D = len(tbox), len(dets)
    tw = np.maximum(tbox[:, 2] - tbox[:, 0], 1.0)
    th = np.maximum(tbox[:, 3] - tbox[:, 1], 1.0)
    ts = np.sqrt(tw * th)
    tc = np.stack([(tbox[:, 0] + tbox[:, 2]) / 2, (tbox[:, 1] + tbox[:, 3]) / 2], 1)
    dw = np.maximum(dets[:, 2] - dets[:, 0], 1.0)
    dh = np.maximum(dets[:, 3] - dets[:, 1], 1.0)
    dc = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)
    g = np.maximum(tgap, 1).astype(float)
    d = (dc[None] - tc[:, None]) / ts[:, None, None]
    dist = np.hypot(d[..., 0], d[..., 1])
    pc = tc + tvel * g[:, None]
    r = (dc[None] - pc[:, None]) / ts[:, None, None]
    res = np.hypot(r[..., 0], r[..., 1])
    shift = pc - tc
    pbox = tbox + np.concatenate([shift, shift], 1)
    sp = np.hypot(tvel[:, 0], tvel[:, 1]) / ts
    disp = dc[None] - tc[:, None]
    cos = (disp[..., 0] * tvel[:, None, 0] + disp[..., 1] * tvel[:, None, 1]) / np.maximum(
        np.hypot(disp[..., 0], disp[..., 1]) * np.hypot(tvel[:, 0], tvel[:, 1])[:, None], 1e-6)
    cos = cos * thas[:, None]
    one = np.ones((T, D))
    pcL = tc + tvelL * g[:, None]
    rL = (dc[None] - pcL[:, None]) / ts[:, None, None]
    resL = np.hypot(rL[..., 0], rL[..., 1])
    shL = pcL - tc
    pboxL = tbox + np.concatenate([shL, shL], 1)
    spL = np.hypot(tvelL[:, 0], tvelL[:, 1]) / ts
    dvel = np.hypot(tvel[:, 0] - tvelL[:, 0], tvel[:, 1] - tvelL[:, 1]) / ts
    F = np.stack([
        d[..., 0], d[..., 1], dist, dist / g[:, None], res, r[..., 0], r[..., 1],
        np.log(dw[None] / tw[:, None]), np.log(dh[None] / th[:, None]),
        np.log((dw * dh)[None] / (tw * th)[:, None]),
        _iou(tbox, dets[:, :4]), _iou(pbox, dets[:, :4]),
        _iou(_buf(tbox, 0.5), _buf(dets[:, :4], 0.5)),
        one * dets[None, :, 4], one * tscore[:, None], one * g[:, None],
        one * np.minimum(tnobs, 20)[:, None], one * thas[:, None], one * sp[:, None], cos,
        one * np.log(ts)[:, None], one * np.log(tw / th)[:, None], one * np.log(dw / dh)[None],
        resL, _iou(pboxL, dets[:, :4]), one * spL[:, None], one * dvel[:, None],
        np.minimum(np.minimum(res, resL), dist), one * (g * sp)[:, None],
    ], -1).astype(np.float32)
    gate = np.minimum(np.minimum(dist, res), resL) <= 2.0 + 1.5 * g[:, None]
    return F, gate


# =============================================================================
# Tree ensemble (flattened gradient-boosted trees, binary log-odds)
# =============================================================================

class TreeEnsemble:
    """Leaves point at themselves, so a fixed number of descent steps suffices."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.feat, self.thr = z["feat"], z["thr"]
        self.left, self.right = z["left"], z["right"]
        self.miss_left, self.value = z["miss_left"], z["value"]
        self.roots = z["roots"]
        self.baseline = float(z["baseline"])
        self.depth = int(z["depth"])
        self.names = [str(n) for n in z["names"]]

    def predict(self, X):
        """P(same object) for rows of X (N, F)."""
        X = np.asarray(X, np.float64)
        rows = np.arange(len(X))[:, None]
        node = np.repeat(self.roots[None, :], len(X), 0)
        for _ in range(self.depth):
            x = X[rows, self.feat[node]]
            go_left = (x <= self.thr[node]) | (np.isnan(x) & self.miss_left[node])
            node = np.where(go_left, self.left[node], self.right[node])
        raw = self.baseline + self.value[node].sum(1)
        return 1.0 / (1.0 + np.exp(-raw))


# =============================================================================
# Track
# =============================================================================

class TreeLinkTrack:
    __slots__ = ("track_id", "obs", "history", "frame", "activated", "lost")

    def __init__(self, track_id, frame, box, score, ts, do, activated):
        self.track_id = track_id
        self.obs = [(frame, box, score)]     # tracker frame index, x1y1x2y2, score
        self.history = [(ts, do)]
        self.frame = frame
        self.activated = activated
        self.lost = False

    def update(self, frame, box, score, ts, do):
        self.obs.append((frame, box, score))
        self.history.append((ts, do))
        self.frame = frame
        self.activated = True
        self.lost = False


def to_ObjectTrackSet(tracks):
    result = []
    for track in tracks:
        t = Track(id=track.track_id)
        for ts, do in track.history:
            if not t.append(ObjectTrackState(ts.get_frame(), ts.get_time_usec(), do)):
                logger.warning("Unsorted input in to_ObjectTrackSet for track %d", track.track_id)
        result.append(t)
    return ObjectTrackSet(result)


# =============================================================================
# Configuration
# =============================================================================

class TreeLinkTrackerConfig(scfg.DataConfig):
    """Configuration for the learned-association tracker."""
    model_file = scfg.Value('', help='Flattened tree ensemble (.npz) scoring track-detection pairs')
    low_thresh = scfg.Value(0.27, help='Detections scoring below this are ignored')
    new_track_thresh = scfg.Value(0.8, help='Minimum score for an unmatched detection to start a track')
    p_match = scfg.Value(0.05, help='Minimum P(same object) for a track-detection link')
    track_buffer = scfg.Value(15, help='Frames a confirmed track survives without a detection')
    params_file = scfg.Value('', help='Optional JSON of trained parameters overriding the above')


# =============================================================================
# Algorithm
# =============================================================================

class TreeLinkTracker(TrackObjects):
    """Single-pass tracker whose association cost is a learned link probability."""

    def __init__(self):
        TrackObjects.__init__(self)
        self._config = TreeLinkTrackerConfig()
        self._model = None
        self.reset()

    def get_configuration(self):
        cfg = super(TrackObjects, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        from viame.pytorch.utilities import vital_config_update

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        params = {k: self._config[k] for k in
                  ('model_file', 'low_thresh', 'new_track_thresh', 'p_match', 'track_buffer')}
        params_file = str(self._config.params_file)
        if params_file and os.path.exists(params_file):
            with open(params_file, 'r') as f:
                trained = json.load(f)
            params.update({k: v for k, v in trained.items() if k in params})
            # A relative model path in the params file is relative to that file
            mf = str(params['model_file'])
            if mf and not os.path.isabs(mf) and not os.path.exists(mf):
                params['model_file'] = os.path.join(os.path.dirname(params_file), mf)
            print(f"[TreeLinkTracker] Loaded trained parameters from {params_file}")

        self._low_thresh = float(params['low_thresh'])
        self._new_track_thresh = float(params['new_track_thresh'])
        self._p_match = float(params['p_match'])
        self._track_buffer = int(float(params['track_buffer']))
        model_file = str(params['model_file'])
        self._model = None
        if model_file:
            self._model = TreeEnsemble(model_file)
            if self._model.names != FEATURE_NAMES:
                raise ValueError("link model features do not match this tracker: %s" % model_file)
        self.reset()
        return True

    def check_configuration(self, cfg):
        return True

    def track(self, ts, image, detections):
        """Associate one frame of detections; returns the tracks updated in it."""
        if self._model is None:
            raise RuntimeError("TreeLinkTracker needs model_file")
        self._frame_id += 1
        frame = self._frame_id
        dos = [do for do in (list(detections) if detections else [])
               if do.confidence >= self._low_thresh]
        dets = np.zeros((len(dos), 5))
        for i, do in enumerate(dos):
            bb = do.bounding_box
            dets[i] = (bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y(), do.confidence)

        trks = self._tracks
        used = set()
        if trks and len(dets):
            D = [track_desc(t.obs, frame) for t in trks]
            F, gate = pair_features(
                np.array([x[0] for x in D]), np.array([x[1] for x in D]),
                np.array([x[2] for x in D]), np.array([x[3] for x in D]),
                np.array([x[4] for x in D]), np.array([x[5] for x in D]), dets,
                np.array([x[6] for x in D]))
            P = np.zeros(gate.shape)
            if gate.any():
                P[gate] = self._model.predict(F[gate])
            C = np.where(P >= self._p_match, 1 - P, 1e3)
            for i, j in zip(*scipy.optimize.linear_sum_assignment(C)):
                if C[i, j] < 1e3:
                    trks[i].update(frame, dets[j, :4].copy(), dets[j, 4], ts, dos[j])
                    used.add(j)

        # Single-hit tracks die on their first miss; confirmed ones wait out the buffer
        live = []
        for t in trks:
            if t.frame != frame:
                if not t.activated or frame - t.frame > self._track_buffer:
                    continue
                t.lost = True
            live.append(t)
        for j in range(len(dets)):
            if j not in used and dets[j, 4] >= self._new_track_thresh:
                self._count += 1
                live.append(TreeLinkTrack(self._count, frame, dets[j, :4].copy(), dets[j, 4],
                                      ts, dos[j], activated=(frame == 1)))
        # Updated tracks first, then lost, as the association order depends on it
        self._tracks = [t for t in live if not t.lost] + [t for t in live if t.lost]
        return to_ObjectTrackSet([t for t in self._tracks if t.activated and not t.lost])

    def initialize(self, ts, image, seed_detections):
        self.reset()
        if seed_detections is not None and len(seed_detections) > 0:
            return self.track(ts, image, seed_detections)
        return ObjectTrackSet([])

    def finalize(self):
        return to_ObjectTrackSet([t for t in self._tracks if t.activated])

    def reset(self):
        self._tracks = []
        self._frame_id = 0
        self._count = 0


# =============================================================================
# Algorithm Registration
# =============================================================================

def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        TreeLinkTracker,
        "tree_link",
        "Single-pass tracker with a learned track-detection link probability",
    )
