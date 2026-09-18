# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
"""Curved stereo length of annotated centerlines (head, spine_NNN..., tail).

Left centerline vertices are transferred to the right camera through a dense
disparity backend on the rectified grid and, in bidirectional mode, checked on
the way back through an independent right-reference disparity. The length is
the sum of 3D segment lengths along the resampled centerline.

A detection without spine vertices gets its centerline from the fish mask: the
ridge path between head and tail, written back as head, spine_NNN, tail
keypoints. A mask made of several polygons is bridged into one shape for the
path while every vertex stays inside the polygons. Annotated head/tail anchor
the path as drawn; model keypoints must be plausible for the mask, and
otherwise each end is where the trunk's tangent leaves the mask. An optional nested
refine_detections algorithm supplies masks and endpoints to boxes that lack
them.
"""
import numpy as np

from kwiver.sprokit.pipeline import datum, process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

from viame.core.curved_measurement import measure_curve, sample_map

SPINE_PREFIX = 'spine_'
MEASURE_OPTIONS = ('samples', 'smoothing', 'consistency_px', 'centerline_tolerance_px',
                   'max_length_disagreement', 'max_depth_ratio')
GRID_OPTIONS = ('rectification_alpha', 'refine_keypoints_disparity_window',
                'refine_keypoints_disparity_percentile')


def is_spine_key(name):
    return name.startswith(SPINE_PREFIX) and name[len(SPINE_PREFIX):].isdigit()


def centerline_names(keypoints):
    """head, spine_NNN in index order, tail; None without both endpoints."""
    if 'head' not in keypoints or 'tail' not in keypoints:
        return None
    spine = sorted((int(k[len(SPINE_PREFIX):]), k) for k in keypoints if is_spine_key(k))
    return ['head'] + [k for _, k in spine] + ['tail']


def _xy(point):
    value = getattr(point, 'value', point)
    return float(value[0]), float(value[1])


def centerline_from_keypoints(keypoints):
    names = centerline_names(keypoints)
    if names is None:
        return None
    points = np.array([_xy(keypoints[n]) for n in names], dtype=float)
    return points if np.isfinite(points).all() else None


def centerline_keypoint_names(count):
    return ['head'] + ['%s%03d' % (SPINE_PREFIX, i) for i in range(1, count - 1)] + ['tail']


def transfer_vertices(vertices, forward, reverse, grid, consistency_px):
    """Right-grid matches of left-grid vertices; NaN rows where no valid match
    exists or, given a right-reference map, where the round trip disagrees."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
    matched = np.array(grid.match_grid_points(forward, vertices), dtype=float).reshape(-1, 2)
    disp = vertices[:, 0] - matched[:, 0]
    h, w = forward.shape
    ok = np.isfinite(matched).all(axis=1) & (disp > 0)
    ok &= (matched[:, 0] >= 0) & (matched[:, 0] <= w - 1)
    ok &= (matched[:, 1] >= 0) & (matched[:, 1] <= h - 1)
    if reverse is not None:
        valid = np.where(np.isfinite(reverse) & (reverse > 0), reverse, np.nan)
        back = sample_map(valid, np.nan_to_num(matched))
        ok &= np.isfinite(back) & (np.abs(back - disp) <= consistency_px)
    matched = matched.copy()
    matched[~ok] = np.nan
    return matched


def detection_mask(det, shape):
    """Full-image boolean mask from the detection's mask crop (anchored at the
    floored box origin, as refiners write it) or from its polygons."""
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    mask = det.mask
    if mask is not None:
        crop = np.asarray(mask.asarray())
        if crop.ndim == 3:
            crop = crop[:, :, 0]
        box = det.bounding_box
        x0, y0 = int(np.floor(box.min_x())), int(np.floor(box.min_y()))
        ch, cw = crop.shape
        xs, ys = slice(max(x0, 0), min(x0 + cw, w)), slice(max(y0, 0), min(y0 + ch, h))
        if xs.stop > xs.start and ys.stop > ys.start:
            out[ys, xs] = crop[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] > 0
    else:
        import cv2
        polygons = [np.asarray(p, dtype=float).reshape(-1, 2) for p in det.get_flattened_polygons()]
        polygons = [np.rint(p).astype(np.int32) for p in polygons if len(p) >= 3]
        if not polygons:
            return None
        raster = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raster, polygons, 1)
        out = raster > 0
    return out if out.any() else None


def endpoints_on_mask(union, endpoints, span):
    from scipy import ndimage
    away = ndimage.distance_transform_edt(~union)
    h, w = union.shape
    for x, y in endpoints:
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < w and 0 <= py < h) or away[py, px] > max(3.0, 0.15 * span):
            return False
    return True


def _inside(union, points):
    h, w = union.shape
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)
    return union[y, x]


def mask_polyline(mask, endpoints, count, harmonics, trusted=False):
    """Head-to-tail centerline of a mask as `count` vertices.

    The ridge path between head and tail is reduced to a smooth midline of
    `harmonics` sine terms pinned to both ends. Several polygons are bridged
    into one shape for the path, but only path points inside the polygons
    shape the curve and every interior vertex lands inside them. endpoints (head, tail; may be None) anchor the path as drawn
    when trusted. Model endpoints must sit on the mask, or the two disagree and
    nothing is written; ones spanning too little of it (a head placed
    mid-body) only orient ends computed from the mask itself."""
    from viame.core.curved_measurement import (
        fit_midline, mask_centerline, mask_end_seeds, merge_components)
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if endpoints is not None:
        endpoints = np.asarray(endpoints, dtype=float).reshape(2, 2)
        xs, ys = np.r_[xs, endpoints[:, 0]], np.r_[ys, endpoints[:, 1]]
    margin = 2
    x0, y0 = max(int(xs.min()) - margin, 0), max(int(ys.min()) - margin, 0)
    x1, y1 = int(xs.max()) + margin + 1, int(ys.max()) + margin + 1
    origin = np.array([x0, y0], dtype=float)
    merged, union = merge_components(mask[y0:y1, x0:x1])
    seeds = mask_end_seeds(union)
    anchored = False
    if endpoints is not None:
        endpoints = endpoints - origin
        span = np.linalg.norm(seeds[0] - seeds[1])
        if not trusted and not endpoints_on_mask(union, endpoints, span):
            raise ValueError('model head/tail lie off the mask')
        anchored = trusted or np.linalg.norm(endpoints[0] - endpoints[1]) >= 0.6 * span
        if not anchored and (np.linalg.norm(endpoints[0] - seeds[1])
                             < np.linalg.norm(endpoints[0] - seeds[0])):
            seeds = seeds[::-1]
    path = mask_centerline(merged, endpoints if anchored else seeds, anchored)
    dense = fit_midline(path, _inside(union, path), harmonics)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    curve = np.column_stack([np.interp(np.linspace(0, arc[-1], count), arc, dense[:, i])
                             for i in range(2)])
    inside = np.nonzero(_inside(union, dense))[0]
    for i in range(1, count - 1):
        nearest = np.argmin(np.linalg.norm(dense - curve[i], axis=1))
        curve[i] = dense[inside[np.argmin(np.abs(inside - nearest))]]
    return curve + origin


class CurvedStereoMeasurer:
    """One stereo frame at a time: dense disparity on the rectified grid, then
    per-centerline transfer and curved length."""

    def __init__(self, stereo_algo, rectifier, options):
        self._algo = stereo_algo
        self._rectifier = rectifier
        self._mode = options['mode']
        self._options = {k: options[k] for k in MEASURE_OPTIONS}
        self._forward = self._reverse = None

    @property
    def bidirectional(self):
        return self._mode == 'bidirectional'

    def set_frame(self, left, right):
        self._rectifier.prepare(left.shape[1], left.shape[0])
        left = self._rectifier.rectify_image(left, False)
        right = self._rectifier.rectify_image(right, True)
        self._forward = self._disparity(left, right)
        self._reverse = None
        if self.bidirectional:
            self._reverse = self._disparity(right[:, ::-1], left[:, ::-1])[:, ::-1]
            if self._reverse.shape != self._forward.shape:
                raise RuntimeError('Reverse disparity does not match the forward grid')

    def _disparity(self, left, right):
        from kwiver.vital.types import Image, ImageContainer
        result = self._algo.compute(ImageContainer(Image(np.ascontiguousarray(left))),
                                    ImageContainer(Image(np.ascontiguousarray(right))))
        if result is None:
            raise RuntimeError('Stereo disparity computation failed')
        disparity = result.image().asarray()
        if disparity.ndim == 3 and disparity.shape[2] == 1:
            disparity = disparity[:, :, 0]
        scale = 256.0 if disparity.dtype == np.uint16 else 1.0
        return disparity.astype(np.float64) / scale

    def calibration(self):
        i = self._rectifier.intrinsics()
        return dict(rectified=True, **{k: float(i[k]) for k in
                                       ('fx', 'fy', 'cx_left', 'cx_right', 'cy', 'baseline')})

    def measure(self, left_vertices, right_vertices=None):
        """Original-coordinate centerlines. Returns (measure_curve result, right
        vertices in original coordinates or None when any vertex failed)."""
        if self._forward is None:
            raise RuntimeError('set_frame must precede measure')
        left = self._rectifier.rectify_points(left_vertices, False)
        right = None
        if right_vertices is not None:
            right = self._rectifier.rectify_points(right_vertices, True)
        matched = transfer_vertices(left, self._forward, self._reverse, self._rectifier,
                                    self._options['consistency_px'])
        mapped = None
        if np.isfinite(matched).all():
            mapped = np.asarray(self._rectifier.unrectify_points(matched, True), dtype=float)
        curve = right if right is not None else matched
        if self.bidirectional and not np.isfinite(curve).all():
            return dict(success=False, error='No round-trip consistent right centerline'), None
        try:
            result = measure_curve(left, self._forward, self.calibration(), right_curve=curve,
                                   right_disparity=self._reverse, mode=self._mode, **self._options)
        except ValueError as error:
            result = dict(success=False, error=str(error))
        return result, mapped


def _bool(value):
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def _block_files(cfg, prefix):
    """(configured, missing): whether any key sits under the block, and the
    absolute paths in it that do not exist (relativepath model entries of an
    add-on that is not installed)."""
    import os
    configured, missing = False, []
    for key in cfg.available_values():
        if not key.startswith(prefix):
            continue
        configured = True
        value = str(cfg.get_value(key))
        if os.path.isabs(value) and not os.path.exists(value):
            missing.append(value)
    return configured, missing


def _first_available_refiner(cfg, block, candidates, log):
    """First refine_detections candidate whose model files exist and that
    configures without error; (name, algorithm) or (None, None)."""
    from kwiver.vital.algo import RefineDetections
    for name in candidates:
        configured, missing = _block_files(cfg, '%s:%s:' % (block, name))
        if not configured:
            log('%s skipped: no %s:%s block' % (name, block, name))
            continue
        if missing:
            log('%s unavailable: missing %s' % (name, ', '.join(missing)))
            continue
        cfg.set_value(block + ':type', name)
        try:
            algo = RefineDetections.set_nested_algo_configuration(block, cfg)
        except Exception as error:
            log('%s failed to load: %s' % (name, error))
            continue
        if algo is None:
            log('%s is not a registered refine_detections algorithm' % name)
            continue
        return name, algo
    return None, None


def _three_channel(image):
    """Byte RGB copy of a frame for segmenters that reject gray or alpha input."""
    from kwiver.vital.types import Image, ImageContainer
    data = image.image().asarray()
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    elif data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 4:
        data = data[:, :, :3]
    return ImageContainer(Image(np.ascontiguousarray(data)))


def _match_refined(inputs, refined):
    """Refined detection for each input, paired by identity then by box: some
    refiners clone, others drop detections they could not segment."""
    matched = [None] * len(inputs)
    unused = list(refined)
    for i, det in enumerate(inputs):
        for j, out in enumerate(unused):
            if out is det:
                matched[i] = unused.pop(j)
                break
    for i, det in enumerate(inputs):
        if matched[i] is not None:
            continue
        box = det.bounding_box
        for j, out in enumerate(unused):
            other = out.bounding_box
            if all(abs(a - b) < 1e-3 for a, b in (
                    (box.min_x(), other.min_x()), (box.min_y(), other.min_y()),
                    (box.max_x(), other.max_x()), (box.max_y(), other.max_y()))):
                matched[i] = unused.pop(j)
                break
    return matched


def _bbox_from_points(points, scale):
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max((x1 - x0) * scale / 2, 1.0), max((y1 - y0) * scale / 2, 1.0)
    return cx - hw, cy - hh, cx + hw, cy + hh


def _set_centerline(det, vertices):
    from kwiver.vital.types import Point2d
    kept = {k: v for k, v in det.keypoints.items()
            if k not in ('head', 'tail') and not is_spine_key(k)}
    det.clear_keypoints()
    for name, point in kept.items():
        det.add_keypoint(name, point)
    for name, (x, y) in zip(centerline_keypoint_names(len(vertices)), vertices):
        det.add_keypoint(name, Point2d(float(x), float(y)))


def _failure_reason(result):
    for part in (result, result.get('left', {}), result.get('right', {})):
        if part.get('error'):
            return part['error']
    return 'measurement rejected'


def _annotate(det, result, method):
    length = float(result['curved_length'])
    det.set_attribute('length', length)
    det.add_note(':length=%f' % length)
    det.add_note(':curved_length=%f' % length)
    det.add_note(':straight_length=%f' % result['straight_length'])
    det.add_note(':curvature_ratio=%f' % result['curvature_ratio'])
    if method:
        det.add_note(':stereo_method=' + method)


CONFIG = (
    ('calibration_file', '', 'Stereo rig calibration (npz, json, yaml or OpenCV directory)'),
    ('mode', 'bidirectional', 'left: transfer left vertices through left-reference '
     'disparity. bidirectional: also compute right-reference disparity from the swapped, '
     'flipped pair, require every vertex to round-trip and both curve lengths to agree.'),
    ('samples', '32', 'Centerline samples used for the 3D length'),
    ('smoothing', '0.5', 'Spline fit tolerance in pixels; endpoints are retained'),
    ('consistency_px', '1.5', 'Maximum round-trip disparity disagreement per sample'),
    ('centerline_tolerance_px', '5.0', 'Maximum distance of a transferred sample from an '
     'existing right centerline'),
    ('max_length_disagreement', '0.1', 'Maximum relative left/right length disagreement'),
    ('max_depth_ratio', '2.0', 'Maximum far/near depth ratio along one centerline'),
    ('right_keypoint_policy', 'keep_existing', 'keep_existing: an annotated right centerline '
     'constrains the transfer and is left untouched. refine_all: replace it with the '
     'transferred vertices.'),
    ('update_right_keypoints', 'true', 'Write transferred vertices to right detections '
     'lacking a centerline (and to all of them with refine_all)'),
    ('create_synthetic_detections', 'true', 'Create a right detection when the left track '
     'has no partner at this frame'),
    ('box_scale_factor', '1.1', 'Synthetic right box size relative to its vertex extent'),
    ('length_aggregation_method', 'median', 'none, average, average_iqr or median per track'),
    ('length_iqr_factor', '1.5', 'Outlier factor for average_iqr'),
    ('record_stereo_method', 'true', 'Add a :stereo_method= note to measured detections'),
    ('mask_refiners', '', 'Ordered refine_detections candidates (e.g. sam2,rf_detr,sam3,'
     'ocv_watershed) that add a mask to left boxes still lacking one; the first whose '
     'model files exist and that loads is used. Configured under mask_refiner:<name>.'),
    ('centerline_source', 'auto', 'auto: drawn spine vertices, else the mask skeleton, else the '
     'head/tail segment. keypoints: only annotated keypoints. mask: only mask skeletons.'),
    ('centerline_vertices', '8', 'Vertices written for a mask-derived centerline (head to tail)'),
    ('centerline_harmonics', '3', 'Sine terms of the smooth midline fitted through a mask '
     'ridge between head and tail: 1 is a single arc, more follow S-bends'),
    ('rectification_alpha', '-1.0', 'OpenCV stereoRectify alpha; -1 keeps every source pixel'),
    ('refine_keypoints_disparity_window', '3', 'Neighbourhood radius sampled at each vertex'),
    ('refine_keypoints_disparity_percentile', '0.9', 'Percentile of the neighbourhood '
     'disparity; high favours the nearer (object) surface'),
)


class MeasureCurvedObjects(KwiverProcess):
    """Ports mirror compute_measurements: image1/2 and object_track_set1/2 in,
    object_track_set1/2 out. The stereo_disparity nested algorithm supplies
    left-reference pixel disparity on rectified input; an optional refiner
    nested algorithm (refine_detections) adds masks and head/tail to left
    detections that lack them before the centerline is derived; the first
    available of mask_refiners then covers boxes still without a shape."""

    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)
        for key, default, description in CONFIG:
            self.add_config_trait(key, key, default, description)
            self.declare_config_using_trait(key)

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)
        for i in (1, 2):
            self.add_port_trait('image%d' % i, 'image', 'Camera %d image' % i)
            self.add_port_trait('object_track_set%d' % i, 'object_track_set',
                                'Camera %d tracks' % i)
            self.declare_input_port_using_trait('image%d' % i, required)
            self.declare_input_port_using_trait('object_track_set%d' % i, required)
            self.declare_output_port_using_trait('object_track_set%d' % i, optional)
        self.declare_input_port_using_trait('timestamp', required)

    def _configure(self):
        from kwiver.vital.algo import ComputeStereoDepthMap
        from viame.core.interactive_stereo import DenseStereoRectifier

        cfg = self.get_config()
        algo = ComputeStereoDepthMap.set_nested_algo_configuration('stereo_disparity', cfg)
        if algo is None:
            raise RuntimeError('stereo_disparity:type must name a dense disparity algorithm')
        if not ComputeStereoDepthMap.check_nested_algo_configuration('stereo_disparity', cfg):
            raise RuntimeError('Invalid stereo_disparity configuration')
        self._refiner = None
        if cfg.has_value('refiner:type') and cfg.get_value('refiner:type'):
            from kwiver.vital.algo import RefineDetections
            self._refiner = RefineDetections.set_nested_algo_configuration('refiner', cfg)
            if self._refiner is None:
                raise RuntimeError('refiner:type names an unknown refine_detections algorithm')
        candidates = [c.strip() for c in self.config_value('mask_refiners').split(',') if c.strip()]
        self._mask_refiner_name, self._mask_refiner = _first_available_refiner(
            cfg, 'mask_refiner', candidates, self._log)
        if candidates and self._mask_refiner is None:
            self._log('no mask refiner available; boxes without a shape are not measured')
        elif self._mask_refiner is not None:
            self._log('mask refiner: ' + self._mask_refiner_name)
        calibration = self.config_value('calibration_file')
        if not calibration:
            raise RuntimeError('calibration_file is required')
        mode = self.config_value('mode')
        if mode not in ('left', 'bidirectional'):
            raise RuntimeError('mode must be left or bidirectional')
        policy = self.config_value('right_keypoint_policy')
        if policy not in ('keep_existing', 'refine_all'):
            raise RuntimeError('right_keypoint_policy must be keep_existing or refine_all')
        method = self.config_value('length_aggregation_method')
        if method not in ('none', 'average', 'average_iqr', 'median'):
            raise RuntimeError('length_aggregation_method must be none, average, '
                               'average_iqr or median')

        options = dict(mode=mode, samples=int(self.config_value('samples')))
        for key in MEASURE_OPTIONS[1:]:
            options[key] = float(self.config_value(key))
        grid = {key: self.config_value(key) for key in GRID_OPTIONS}
        self._measurer = CurvedStereoMeasurer(
            algo, DenseStereoRectifier(calibration, grid), options)
        self._policy = policy
        self._update_right = _bool(self.config_value('update_right_keypoints'))
        self._synthetic = _bool(self.config_value('create_synthetic_detections'))
        self._box_scale = float(self.config_value('box_scale_factor'))
        self._aggregation = method
        self._iqr_factor = float(self.config_value('length_iqr_factor'))
        self._record_method = _bool(self.config_value('record_stereo_method'))
        self._source = self.config_value('centerline_source')
        if self._source not in ('auto', 'keypoints', 'mask'):
            raise RuntimeError('centerline_source must be auto, keypoints or mask')
        self._vertices = int(self.config_value('centerline_vertices'))
        self._harmonics = int(self.config_value('centerline_harmonics'))
        if self._vertices < 4 or self._harmonics < 1:
            raise RuntimeError('centerline_vertices must be >= 4 and centerline_harmonics >= 1')
        self._tracks = ({}, {})
        self._lengths = {}
        self._finalized = False
        self._base_configure()

    # ------------------------------------------------------------------ step
    def _step(self):
        if self.peek_at_datum_on_port('object_track_set1').type() == datum.DatumType.complete:
            try:
                self._finalize()
            finally:
                self.mark_process_as_complete()
            return

        timestamp = self.grab_input_using_trait('timestamp')
        images = [self.grab_input_using_trait('image%d' % i) for i in (1, 2)]
        sets = [self.grab_input_using_trait('object_track_set%d' % i) for i in (1, 2)]
        frame = timestamp.get_frame() if timestamp.has_valid_frame() else -1

        states = [self._states_at(s, frame) for s in sets]
        if frame >= 0 and any(self._could_measure(st.detection()) for st in states[0].values()):
            self._measurer.set_frame(*[im.image().asarray() for im in images])
            self._measure_frame(frame, states, images[0])
        for camera, frame_states in enumerate(states):
            for tid, state in frame_states.items():
                self._append(camera, tid, state.frame_id, state.time_usec, state.detection())
        self._aggregate()
        self._push()
        self._base_step()

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        self._aggregate(final=True)
        self._push()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _states_at(track_set, frame):
        states = {}
        if track_set is None:
            return states
        for track in track_set.tracks():
            for state in track:
                if state.frame_id == frame and state.detection() is not None:
                    states[track.id] = state
        return states

    @staticmethod
    def _has_shape(det):
        return det.mask is not None or bool(det.get_flattened_polygons())

    def _could_measure(self, det):
        if self._refiner is not None or self._mask_refiner is not None:
            return True
        if self._source != 'mask' and centerline_from_keypoints(det.keypoints) is not None:
            return True
        return self._source != 'keypoints' and self._has_shape(det)

    def _wants_refinement(self, det):
        curve = centerline_from_keypoints(det.keypoints)
        if curve is not None and len(curve) > 2:
            return False
        return curve is None or not self._has_shape(det)

    def _refine(self, frame, left_states, image):
        """Nested refiner over the left detections still missing a mask or
        endpoints, then the mask refiner over those still without a shape."""
        from kwiver.vital.types import DetectedObjectSet, ObjectTrackState

        def run(refiner, label, wanted):
            todo = [(tid, st) for tid, st in left_states.items() if wanted(st.detection())]
            if not todo:
                return
            try:
                refined = list(refiner.refine(
                    image, DetectedObjectSet([st.detection() for _, st in todo])))
            except Exception as error:
                self._log('frame %d: %s failed: %s' % (frame, label, error))
                return
            for (tid, state), det in zip(todo, _match_refined([st.detection() for _, st in todo],
                                                              refined)):
                if det is not None and det is not state.detection():
                    left_states[tid] = ObjectTrackState(state.frame_id, state.time_usec, det)

        if self._refiner is not None:
            run(self._refiner, 'refiner', self._wants_refinement)
        if self._mask_refiner is not None:
            image = _three_channel(image)
            run(self._mask_refiner, self._mask_refiner_name, self._wants_mask)

    def _wants_mask(self, det):
        curve = centerline_from_keypoints(det.keypoints)
        return (curve is None or len(curve) <= 2) and not self._has_shape(det)

    def _left_centerline(self, det, shape, trusted):
        """Vertices to measure, deriving and writing them from the mask when
        the detection has no drawn spine. trusted: head/tail were annotated
        rather than added by the refiner. None when nothing usable exists."""
        curve = centerline_from_keypoints(det.keypoints)
        if curve is not None and (len(curve) > 2 or self._source == 'keypoints'):
            return curve
        if self._source != 'keypoints':
            mask = detection_mask(det, shape)
            if mask is not None:
                try:
                    path = mask_polyline(mask, curve, self._vertices, self._harmonics, trusted)
                    _set_centerline(det, path)
                    det.add_note(':centerline_source=mask')
                    return path
                except (ValueError, ImportError) as error:
                    self._log('mask centerline failed: %s' % error)
                    if not trusted:
                        return None
        if curve is not None and not trusted:
            box = det.bounding_box
            if np.linalg.norm(curve[0] - curve[-1]) < 0.5 * max(box.width(), box.height()):
                self._log('model head/tail span too little of the box; not measured')
                return None
        return None if self._source == 'mask' else curve

    def _measure_frame(self, frame, states, left_image):
        from kwiver.vital.types import (
            BoundingBoxD, DetectedObject, ObjectTrackState)
        left_states, right_states = states
        annotated = {tid for tid, st in left_states.items()
                     if centerline_names(st.detection().keypoints) is not None}
        if self._refiner is not None or self._mask_refiner is not None:
            self._refine(frame, left_states, left_image)
        shape = left_image.image().asarray().shape[:2]
        method = ('curved_' + ('bidirectional' if self._measurer.bidirectional else 'left')
                  if self._record_method else '')
        for tid, state in left_states.items():
            left_det = state.detection()
            left_curve = self._left_centerline(left_det, shape, tid in annotated)
            if left_curve is None:
                continue
            right_state = right_states.get(tid)
            right_det = right_state.detection() if right_state is not None else None
            right_curve = None
            if right_det is not None and self._policy == 'keep_existing':
                right_curve = centerline_from_keypoints(right_det.keypoints)
            try:
                result, mapped = self._measurer.measure(left_curve, right_curve)
            except Exception as error:
                self._log('track %d frame %d: %s' % (tid, frame, error))
                continue
            if not result.get('success'):
                self._log('track %d frame %d: %s' % (tid, frame, _failure_reason(result)))
                continue
            _annotate(left_det, result, method)
            self._lengths.setdefault(tid, []).append(float(result['curved_length']))
            if right_det is None:
                if not self._synthetic or mapped is None:
                    continue
                right_det = DetectedObject(BoundingBoxD(*_bbox_from_points(mapped, self._box_scale)),
                                           left_det.confidence, left_det.type)
                right_states[tid] = ObjectTrackState(state.frame_id, state.time_usec, right_det)
            if mapped is not None and self._update_right and (
                    right_curve is None or self._policy == 'refine_all'):
                _set_centerline(right_det, mapped)
            _annotate(right_det, result, method)

    def _append(self, camera, tid, frame, time_usec, det):
        from kwiver.vital.types import ObjectTrackState, Track
        track = self._tracks[camera].get(tid)
        if track is None:
            track = self._tracks[camera][tid] = Track(id=tid)
        if track.size and track.last_frame >= frame:
            return
        track.append(ObjectTrackState(frame, time_usec, det))

    def _aggregate(self, final=False):
        if self._aggregation == 'none':
            return
        from viame.core import _measurement
        for tid, lengths in self._lengths.items():
            value = _measurement.aggregate_lengths(lengths, self._aggregation, self._iqr_factor)
            if value <= 0:
                continue
            for tracks in self._tracks:
                track = tracks.get(tid)
                if track is None:
                    continue
                track.set_attribute('length', float(value))
                if final:
                    for state in track:
                        det = state.detection()
                        if det is not None and det.has_attribute('length'):
                            det.add_note(':avg_length=%f' % value)

    def _push(self):
        from kwiver.vital.types import ObjectTrackSet
        for camera in (0, 1):
            self.push_to_port_using_trait(
                'object_track_set%d' % (camera + 1),
                ObjectTrackSet(list(self._tracks[camera].values())))

    def _log(self, message):
        print('[compute_curved_measurements] ' + message, flush=True)


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:viame.core.measure_curved_process'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'compute_curved_measurements',
        'Curved stereo length of annotated centerlines via dense disparity',
        MeasureCurvedObjects)
    process_factory.mark_process_module_as_loaded(module_name)
