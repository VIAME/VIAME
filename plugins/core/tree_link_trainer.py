# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
TreeLink tracker training.

Fits the link scorer the tree_link tracker associates with: a gradient-boosted
tree ensemble giving P(same object) for a (track, detection) pair. Pairs come
from a detector's output over the training clips, labelled with groundtruth
track ids; without detector output the groundtruth boxes stand in for it.

The gates (low_thresh, new_track_thresh, confirm_thresh, p_match, track_buffer)
are then searched by running the tracker. To keep that honest every clip is tracked
with a scorer that never saw it: the clips are split in two by a hash of their
names and each half is tracked with the scorer fitted on the other. The
objective is by default the mean of MOTA and IDF1 pooled over all clips. The
shipped scorer is refitted on every clip. Fitting needs scikit-learn; running
does not.
"""

import hashlib
import json
import os

import numpy as np
import scipy.optimize

from kwiver.vital.algo import TrainTracker

from viame.core.tree_link_tracker import (
    FEATURE_NAMES, TreeEnsemble, _iou, pair_features, track_desc,
)

MODEL_NAME = "tree_link_model.npz"
PARAMS_NAME = "tree_link_params.json"

# Searched one key at a time, in this order, from the seed values
GATE_GRID = (
    ('new_track_thresh', (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
    ('low_thresh', (0.05, 0.1, 0.2, 0.27)),
    ('p_match', (0.02, 0.05, 0.1, 0.15, 0.3)),
    ('track_buffer', (5, 10, 15, 30)),
)

SEED_GATES = dict(low_thresh=0.27, new_track_thresh=0.8, confirm_thresh=0.0, p_match=0.05,
                  track_buffer=15)

# Second pass: the start gate found above becomes the confirmation gate and starts may go lower
START_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)


# =============================================================================
# Training pairs
# =============================================================================

def label_detections(computed, truth, min_score, iou_threshold=0.5):
    """frame -> (dets (n,5), groundtruth id per det, -1 where unmatched)."""
    frames = {}
    for f, rows in computed.items():
        d = np.array(rows, float).reshape(-1, 5)
        d = d[d[:, 4] >= min_score]
        ids = -np.ones(len(d), int)
        g = np.array(truth.get(f, []), float).reshape(-1, 5)
        if len(d) and len(g):
            iou = _iou(d[:, :4], g[:, :4])
            for i, j in zip(*scipy.optimize.linear_sum_assignment(-iou)):
                if iou[i, j] >= iou_threshold:
                    ids[i] = int(g[j, 4])
        frames[f] = (d, ids)
    return frames


def link_pairs(frames, max_gap=15):
    """(X, y) for one clip of labelled detections: gated (track, detection) pairs
    and same-object labels.

    A track is the detections sharing a groundtruth id; each of its prefixes is
    paired with every frame up to max_gap ahead, stopping at its next sighting."""
    obs = {}
    for f in sorted(frames):
        d, ids = frames[f]
        for i in np.where(ids >= 0)[0]:
            obs.setdefault(ids[i], []).append((f, d[i, :4], d[i, 4]))
    by_frame = {}
    for gid, o in obs.items():
        for k in range(len(o)):
            f = o[k][0]
            nxt = o[k + 1][0] if k + 1 < len(o) else None
            desc = track_desc(o[:k + 1], f)
            for gap in range(1, max_gap + 1):
                t = f + gap
                if nxt is not None and t > nxt:
                    break
                if t in frames and len(frames[t][0]):
                    by_frame.setdefault(t, []).append((gid, desc, gap))
    X, Y = [], []
    for t, L in by_frame.items():
        d, ids = frames[t]
        D = [x[1] for x in L]
        F, gate = pair_features(
            np.array([x[0] for x in D]), np.array([x[1] for x in D]),
            np.array([x[2] for x in D]), np.array([x[3] for x in D]),
            np.array([x[4] for x in D]), np.array([x[2] for x in L]), d,
            np.array([x[6] for x in D]))
        same = np.array([x[0] for x in L])[:, None] == ids[None, :]
        X.append(F[gate])
        Y.append(same[gate])
    if not X:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32), np.zeros(0, bool)
    return np.concatenate(X), np.concatenate(Y)


# =============================================================================
# Scorer
# =============================================================================

def fit_scorer(X, y, max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
               l2_regularization=1.0):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        raise RuntimeError("Training the tree_link tracker needs scikit-learn")
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization, random_state=0).fit(X, y)


def save_scorer(model, path):
    """Flatten a fitted HistGradientBoostingClassifier into the tracker's .npz."""
    feat, thr, left, right, miss, value, roots = [], [], [], [], [], [], []
    depth = 0
    for (tree,) in model._predictors:
        nodes = tree.nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported")
        off = len(feat)
        roots.append(off)
        depth = max(depth, int(nodes["depth"].max()))
        for i, n in enumerate(nodes):
            leaf = bool(n["is_leaf"])
            # Leaves point at themselves so descent can run a fixed depth
            feat.append(0 if leaf else int(n["feature_idx"]))
            thr.append(np.inf if leaf else float(n["num_threshold"]))
            left.append(off + (i if leaf else int(n["left"])))
            right.append(off + (i if leaf else int(n["right"])))
            miss.append(True if leaf else bool(n["missing_go_to_left"]))
            value.append(float(n["value"]) if leaf else 0.0)
    np.savez_compressed(
        path, feat=np.array(feat, np.int32), thr=np.array(thr, np.float64),
        left=np.array(left, np.int32), right=np.array(right, np.int32),
        miss_left=np.array(miss, bool), value=np.array(value, np.float64),
        roots=np.array(roots, np.int32),
        baseline=float(np.ravel(model._baseline_prediction)[0]),
        depth=depth, names=np.array(FEATURE_NAMES))
    return len(roots), len(feat)


def check_scorer(model, path, X):
    """Largest gap between the saved ensemble and the model it came from."""
    if not len(X):
        return 0.0
    X = X[:100000]
    return float(np.abs(TreeEnsemble(path).predict(X) - model.predict_proba(X)[:, 1]).max())


# =============================================================================
# Trainer
# =============================================================================

class TreeLinkTrainer(TrainTracker):
    """Fits the tree_link tracker's link scorer and gates from track groundtruth."""

    def __init__(self):
        TrainTracker.__init__(self)

        self._identifier = "viame-tree-link-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "trained_model"
        self._output_prefix = "tree_link_tracker"
        self._pipeline_template = ""
        self._threshold = "0.00"

        # Detector output for the same clips, one VIAME CSV per clip
        self._computed_detections = ""
        # Written by the training tool: which frames belong to which track set
        self._sequence_manifest = ""
        self._validation_manifest = ""

        self._min_score = 0.1
        self._max_gap = 15
        self._max_iter = 400
        self._learning_rate = 0.08
        self._max_leaf_nodes = 31
        self._l2_regularization = 1.0

        self._search_parameters = True
        # 0 tracks every held-out clip
        self._search_max_sequences = 0
        self._search_confidence = 0.0
        # Objective: mean of these motmetrics names, pooled over clips or per clip
        self._search_metrics = "mota,idf1"
        self._search_pooled = True
        # A gate moves only for a gain above this, so ties keep the seed
        self._search_min_gain = 0.0005
        # Processes tracking clips during the search, 0 picks from the CPU count
        self._search_workers = 0

        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    _STR = ("identifier", "train_directory", "output_directory", "output_prefix",
            "pipeline_template", "threshold", "computed_detections",
            "sequence_manifest", "validation_manifest", "search_metrics")
    _FLOAT = ("min_score", "learning_rate", "l2_regularization", "search_confidence",
              "search_min_gain")
    _INT = ("max_gap", "max_iter", "max_leaf_nodes", "search_max_sequences",
            "search_workers")
    _BOOL = ("search_parameters", "search_pooled")

    def get_configuration(self):
        cfg = super(TrainTracker, self).get_configuration()
        for key in self._STR + self._FLOAT + self._INT + self._BOOL:
            cfg.set_value(key, str(getattr(self, "_" + key)))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        for key in self._STR:
            setattr(self, "_" + key, str(cfg.get_value(key)))
        for key in self._FLOAT:
            setattr(self, "_" + key, float(cfg.get_value(key)))
        for key in self._INT:
            setattr(self, "_" + key, int(float(cfg.get_value(key))))
        for key in self._BOOL:
            value = str(cfg.get_value(key)).lower()
            setattr(self, "_" + key, value in ('true', '1', 'yes'))

        for folder in (self._train_directory, self._output_directory):
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
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

        self._train_image_files = list(train_files)
        self._train_tracks = list(train_tracks)
        self._test_image_files = list(test_files)
        self._test_tracks = list(test_tracks)

    def _clips(self):
        """[(detections, truth)] per clip, both keyed by frame id."""
        from viame.core.tracker_param_search import _groundtruth_by_frame
        from viame.core.training_data import (
            build_sequence_maps, load_computed_detections, read_sequence_manifest,
        )

        track_sets = self._train_tracks + self._test_tracks
        image_files = self._train_image_files + self._test_image_files
        names = None
        if self._computed_detections:
            if self._sequence_manifest:
                _maps, names = read_sequence_manifest(
                    self._sequence_manifest, image_files, len(track_sets))
            if names is None:
                _maps, names = build_sequence_maps(
                    image_files, len(track_sets), "tree_link training")

        clips, missing = [], 0
        self._clip_names = []
        for i, track_set in enumerate(track_sets):
            if track_set is None:
                continue
            truth = _groundtruth_by_frame(track_set)
            if not truth:
                continue
            if self._computed_detections:
                name = names[i] if names else None
                computed = load_computed_detections(self._computed_detections, name)
                if not computed:
                    missing += 1
                    continue
            else:
                computed = {f: [g[:4] + (1.0,) for g in rows] for f, rows in truth.items()}
            clips.append((computed, truth))
            self._clip_names.append(names[i] if names else None)
        if missing:
            print("  {} clips had no detection file and were skipped".format(missing))
        return clips

    def _fit(self, clips, pairs):
        X = np.concatenate([pairs[i][0] for i in clips])
        y = np.concatenate([pairs[i][1] for i in clips])
        if len(X) < 100 or y.all() or not y.any():
            raise RuntimeError("Too few training pairs ({}, {} positive)".format(len(X), int(y.sum())))
        model = fit_scorer(X, y, self._max_iter, self._learning_rate,
                           self._max_leaf_nodes, self._l2_regularization)
        return model, X, y

    def _halves(self, count):
        """Two halves of the clips, by name hash so a clip stays in its half
        when others are added; alternating when the names are not known."""
        names = getattr(self, "_clip_names", [])
        if len(names) == count and all(names):
            side = [int(hashlib.md5(n.encode()).hexdigest(), 16) % 2 for n in names]
        else:
            side = [i % 2 for i in range(count)]
        return [[i for i in range(count) if side[i] == h] for h in (0, 1)]

    def _search_gates(self, clips, pairs, gates):
        """Gates picked by tracking each half of the clips with a scorer fitted on the other."""
        from viame.core.tracker_param_search import search_parameters

        halves = self._halves(len(clips))
        if min(len(h) for h in halves) < 2:
            print("  too few clips to hold any out, keeping the default gates")
            return gates
        scorer_for = {}
        for h in (0, 1):
            model, _X, _y = self._fit(halves[h], pairs)
            path = os.path.join(self._train_directory, "tree_link_search_model_%d.npz" % h)
            save_scorer(model, path)
            for i in halves[1 - h]:
                scorer_for[i] = path
        print("  halves of {} and {} clips".format(len(halves[0]), len(halves[1])))

        # Sample evenly across clip size, and feed empty frames so tracks age
        held = sorted(range(len(clips)), key=lambda i: sum(len(v) for v in clips[i][0].values()))
        if self._search_max_sequences and len(held) > self._search_max_sequences:
            step = len(held) / float(self._search_max_sequences)
            held = [held[int(i * step)] for i in range(self._search_max_sequences)]
        sequences = []
        for i in held:
            computed, truth = clips[i]
            last = max(max(computed), max(truth))
            sequences.append(({f: computed.get(f, []) for f in range(last + 1)}, truth,
                              {"model_file": scorer_for[i]}))

        workers = self._search_workers or max(1, min(8, (os.cpu_count() or 2) // 2))
        options = dict(
            confidence_threshold=self._search_confidence,
            metrics=[m.strip() for m in self._search_metrics.split(",") if m.strip()],
            pooled=self._search_pooled, min_gain=self._search_min_gain, workers=workers)
        found, _report = search_parameters(
            'tree_link', dict(gates), sequences, grid=GATE_GRID, **options)

        # Weak detections may start a track that is only output once a strong one joins it
        bar = found['new_track_thresh']
        lower = tuple(v for v in START_GRID if found['low_thresh'] <= v < bar)
        if lower:
            print("  searching a lower start gate under a confirmation gate of {}".format(bar))
            found, _report = search_parameters(
                'tree_link', dict(found, confirm_thresh=bar), sequences,
                grid=(('new_track_thresh', lower),), **options)
            if found['new_track_thresh'] >= bar:
                found['confirm_thresh'] = 0.0
        return {k: found[k] for k in gates}

    def update_model(self):
        print("Starting tree_link tracker training...")
        clips = self._clips()
        if not self._computed_detections:
            print("  no computed_detections given: fitting on groundtruth boxes, "
                  "so the scorer has not seen detector noise or false alarms")
        print("  {} clips".format(len(clips)))

        labelled = [label_detections(c, t, self._min_score) for c, t in clips]
        boxes = sum(len(rows) for _c, t in clips for rows in t.values())
        dets = sum(len(d) for clip in labelled for d, _ids in clip.values())
        hits = sum(int((ids >= 0).sum()) for clip in labelled for _d, ids in clip.values())
        # A low match rate usually means the detection frame ids are off
        print("  {} groundtruth boxes, {} detections, {} of them on a groundtruth box".format(
            boxes, dets, hits))
        pairs = [link_pairs(frames, self._max_gap) for frames in labelled]
        everything = list(range(len(clips)))

        gates = dict(SEED_GATES)
        if self._search_parameters and self._computed_detections:
            print("Searching gates on held-out clips...")
            try:
                gates = self._search_gates(clips, pairs, gates)
            except Exception as e:
                # A failed search must not lose the run: the defaults still track
                print("  gate search failed ({}), keeping the default gates".format(e))

        print("Fitting the link scorer on every clip...")
        model, X, y = self._fit(everything, pairs)
        model_file = os.path.join(self._train_directory, MODEL_NAME)
        trees, nodes = save_scorer(model, model_file)
        print("  {} pairs, {:.1f}% positive, {} trees, {} nodes".format(
            len(X), 100 * y.mean(), trees, nodes))
        print("  saved ensemble differs from the fitted model by at most {:.2g}".format(
            check_scorer(model, model_file, X)))

        params = dict(gates, model_file=MODEL_NAME)
        params_file = os.path.join(self._train_directory, PARAMS_NAME)
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        print("Saved parameters to {}".format(params_file))
        print("\ntree_link tracker training complete!\n")

        return self._get_output_map(params, params_file, model_file)

    def _get_output_map(self, params, params_file, model_file):
        """File paths are copied next to the pipeline; other values fill its template."""
        algo = "tree_link"
        output = {"type": algo}
        output[algo + ":low_thresh"] = "{:.3f}".format(params['low_thresh'])
        output[algo + ":new_track_thresh"] = "{:.3f}".format(params['new_track_thresh'])
        output[algo + ":confirm_thresh"] = "{:.3f}".format(params['confirm_thresh'])
        output[algo + ":p_match"] = "{:.3f}".format(params['p_match'])
        output[algo + ":track_buffer"] = str(int(params['track_buffer']))
        output[algo + ":model_file"] = MODEL_NAME
        output[MODEL_NAME] = model_file
        output[PARAMS_NAME] = params_file
        return output


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        TreeLinkTrainer,
        "tree_link",
        "TreeLink link scorer and gate fitting from track groundtruth",
    )
