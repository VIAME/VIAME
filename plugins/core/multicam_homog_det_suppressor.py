# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import logging
import os

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
)

from .multicam_homog_tracker import MultiHomographyF2F, diff_homogs
from .simple_homog_tracker import (
    Homography, Transformer, add_declare_config,
    get_DetectedObject_bbox, to_DetectedObject_list, wrap_F2FHomography,
)
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)

logger = logging.getLogger(__name__)

# (homogs, sizes, names): the source frames that suppress a camera. `names`
# runs parallel to `sizes` (one source-image filename per suppressing frame) so
# each suppression-region polygon can be traced back to the frame it came from.
zero_homog_and_size = (np.empty((0, 3, 3)), np.empty((0, 2), dtype=int),
                       np.empty(0, dtype=object))

def get_self_suppression_homogs_and_sizes(multihomog, sizes, names):
    """Get suppression homographies, sizes and source names within a timestep

    Returns a value suitable for supplying as arg_suppress_boxes's
    suppression_homogs_and_sizes argument.

    """
    return [
        zero_homog_and_size if cam == hc else (to_curr[s], sizes[s], names[s])
        for cam, to_curr in enumerate(diff_homogs(
                multihomog.homogs, multihomog.homogs,
        ))
        for hc in [len(multihomog) // 2]
        # XXX This choice of suppression is very tied to how
        # stabilize_many_images works
        for s in [np.s_[min(cam + 1, hc):max(cam, hc + 1)]]
    ]

def concat_suppression_homogs_and_sizes(*args):
    """Combine multiple values suitable for supplying as
    arg_suppress_boxes's suppression_homogs_and_sizes argument into
    one

    """
    if not args:
        raise ValueError("At least one argument required")
    ncam = len(args[0])
    if any(len(arg) != ncam for arg in args):
        raise ValueError("Arguments must have a consistent number of cameras")
    result = []
    for args_single_cam in zip(*args):
        homogs, sizes, names = zip(*args_single_cam)
        result.append((np.concatenate(homogs), np.concatenate(sizes),
                       np.concatenate(names)))
    return result

def arg_suppress_boxes(box_lists, suppression_homogs_and_sizes):
    """Compute whether bounding boxes should be kept after suppression

    Arguments:
    - box_lists (list[list[.simple_homog_tracker.BBox]]): bounding
      boxes, where the elements of the outer list correspond to the
      different cameras
    - suppression_homogs_and_sizes (list[tuple[ndarray, ndarray]]):
      transformations to previous frames and those frames' sizes.  The
      elements of the list correspond to the different cameras.  Each
      pair holds:
      - homogs: Nx3x3 ndarray whose first dimension corresponds to
        the previous frames that suppress detections for this camera.
        Each element is a homography that warps camera coordinates to
        previous-frame coordinates.
      - sizes: Nx2 ndarray with the image sizes of the previous frames

    Returns list[list[bool]], False when the corresponding BBox's
    center should have been in a previous frame.

    """
    def center_in_bounds(box, homogs, sizes, _names=None):
        transform = Homography.matrix_transform
        tc = np.squeeze(transform(homogs, box.center[:, np.newaxis]), -1)
        return ((0 <= tc) & (tc < sizes)).all(-1).any(-1)
    # XXX This could perhaps be vectorized with Numpy
    return [[not center_in_bounds(b, *shs) for b in boxes]
            for boxes, shs in zip(box_lists, suppression_homogs_and_sizes)]

def get_all_other_current_homogs_and_sizes(multihomog, sizes, names):
    """Get every other camera of the current timestep, in both directions

    Unlike get_self_suppression_homogs_and_sizes, which only pairs cameras in
    the direction used for suppression, this returns, for each camera, the
    homographies to all other current-timestep cameras.  Used for the
    boundary-cutoff check, where coverage by any related frame matters
    regardless of which of the pair suppresses the other.

    """
    return [
        (np.concatenate([to_curr[:cam], to_curr[cam + 1:]]),
         np.concatenate([sizes[:cam], sizes[cam + 1:]]),
         np.concatenate([names[:cam], names[cam + 1:]]))
        for cam, to_curr in enumerate(diff_homogs(
                multihomog.homogs, multihomog.homogs,
        ))
    ]

def get_detection_points(do):
    """Get an Nx2 ndarray outlining a detection: its polygon if it has one,
    else the corners of its bounding box"""
    flat = do.get_flattened_polygon()
    if flat:
        points = np.asarray(flat, dtype=float).reshape(-1, 2)
        if len(points) >= 3:
            return points
    bbox = get_DetectedObject_bbox(do)
    return np.array([
        [bbox.xmin, bbox.ymin], [bbox.xmax, bbox.ymin],
        [bbox.xmax, bbox.ymax], [bbox.xmin, bbox.ymax],
    ], dtype=float)

def get_boundary_points(points, size, threshold):
    """Of an Nx2 ndarray of points, those within threshold pixels of the
    boundary of an image of the given (width, height) size"""
    width, height = size
    near = (
        (points[:, 0] <= threshold) | (points[:, 1] <= threshold)
        | (points[:, 0] >= width - 1 - threshold)
        | (points[:, 1] >= height - 1 - threshold)
    )
    return points[near]

def mark_boundary_suppressed(do_lists, sizes, boundary_homogs_and_sizes,
                             threshold, attribute):
    """Mark detections cut off by the image boundary as suppressed

    A detection whose outline (polygon if present, else bounding box) comes
    within `threshold` pixels of the image boundary is considered cut off by
    it.  If any of those near-boundary points also falls inside another
    related frame -- i.e. the boundary area appears in another image, where
    this image's field of view is drawn as a suppression region -- the
    detection is marked with a `:<attribute>=true` note, which serializes to
    a `(atr) <attribute> true` detection attribute in VIAME CSV.  Downstream
    (DIVE) displays such detections as the suppression type and excludes
    them from its own type's counts without hiding them.

    When some related frame contains the WHOLE detection clear of its own
    boundary band -- i.e. the object appears uncut there -- the frame where
    it sits most interior is linked with the same `:source_image=` note the
    suppression regions carry, pointing reviewers at the image most likely
    to show the object clearly.

    """
    for dos, size, (homogs, src_sizes, names) in zip(
            do_lists, sizes, boundary_homogs_and_sizes):
        if len(homogs) == 0:
            continue
        for do in dos:
            all_points = get_detection_points(do)
            points = get_boundary_points(all_points, size, threshold)
            if len(points) == 0:
                continue
            # (n_homogs, 2, n_points) positions of the near-boundary points
            # in each related frame
            transformed = Homography.matrix_transform(homogs, points.T)
            in_bounds = (
                (0 <= transformed) & (transformed < src_sizes[:, :, np.newaxis])
            ).all(-2).any()
            if in_bounds:
                note = ':' + attribute + '=true'
                if note not in do.notes:
                    do.add_note(note)
                # Best alternative view: transform the FULL outline into each
                # related frame and score by its margin from that frame's
                # boundary; link the frame where the detection sits deepest
                # inside, provided it clears the boundary band there (the
                # object is genuinely uncut in that image).
                t_all = Homography.matrix_transform(homogs, all_points.T)
                margins = np.minimum(
                    t_all.min(axis=-1),                      # left/top
                    (src_sizes - 1) - t_all.max(axis=-1),    # right/bottom
                ).min(axis=-1)
                best = int(np.argmax(margins))
                if margins[best] > threshold and names[best] is not None:
                    src = ':source_image=' + os.path.basename(str(names[best]))
                    if src not in do.notes:
                        do.add_note(src)

def clip_poly(poly, scores):
    """Clip a poly, only keeping points with a nonnegative score"""
    result = []
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        s1, s2 = scores[i], scores[(i + 1) % len(poly)]
        if s1 == 0:
            result.append(p1)
        else:
            if s1 > 0:
                result.append(p1)
            if s2 != 0 and (s1 > 0) != (s2 > 0):
                result.append((s2 * p1 - s1 * p2) / (s2 - s1))
    return result and np.stack(result)

def transform_poly_to_polys(poly, homog, clip_size):
    """Transform a convex polygon into zero to two clipped polygons

    Polygons are represented as Nx2 array-likes

    """
    # Convert to homogeneous coordinates
    poly = np.concatenate([poly, np.ones((len(poly), 1))], axis=1)
    # Transform (ensuring homog has a non-negative determinant)
    tpoly = poly @ (-homog if np.linalg.det(homog) < 0 else homog).T
    # Cut polygon if needed
    tpolys = [
        clip_poly(tpoly, tpoly[:, 2]),
        clip_poly(-tpoly, -tpoly[:, 2])[::-1],
    ]
    tpolys = [poly for poly in tpolys if len(poly) and (poly[:, 2] != 0).any()]
    if not tpolys:
        raise ValueError("All transformed points lie in camera ground plane")
    # Clip to size
    def clip(poly, scorers):
        for scorer in scorers:
            poly = clip_poly(poly, poly @ scorer)
            if len(poly) == 0:
                break
        return poly
    cpolys = [clip(poly, [
        [1, 0, 0], [0, 1, 0], [-1, 0, clip_size[0]], [0, -1, clip_size[1]],
    ]) for poly in tpolys]
    # Convert from homogeneous coordinates
    return [poly[:, :2] / poly[:, 2:] for poly in cpolys if len(poly) > 2]

def suppression_polys(suppression_homogs_and_sizes, sizes):
    """Per camera, the suppression-region polygons paired with the filename of
    the source frame each came from: list[list[(poly, source_name)]]."""
    def size_to_poly(size):
        w, h = size
        return [[0, 0], [0, h], [w, h], [w, 0]]
    return [[
        (r, name)
        for h, s, name in zip(homogs, src_sizes, names)
        for r in transform_poly_to_polys(size_to_poly(s), np.linalg.inv(h), size)
    ] for (homogs, src_sizes, names), size
        in zip(suppression_homogs_and_sizes, sizes)]

def wrap_poly(poly, class_, source_name=None, merged_count=0):
    result = DetectedObject(
        bbox=BoundingBoxD(*poly.min(0), *poly.max(0)),
        classifications=DetectedObjectType(class_, 1),
    )
    result.set_flattened_polygon(poly.reshape(-1))
    if source_name:
        # Record which source frame's field of view produced this region, so
        # downstream (e.g. DIVE) can trace a suppression region to its image.
        result.add_note(':source_image=' + os.path.basename(str(source_name)))
    if merged_count:
        # This region is the union of several overlapping source regions (see
        # max_overlap_suppr_regions).
        result.add_note(':merged_regions=' + str(merged_count))
    return result


def _polys_overlap(a, b):
    """True when convex polygons a and b (N,2) intersect (shapely when
    available, else cv2.intersectConvexConvex)."""
    try:
        from shapely.geometry import Polygon
        return Polygon(a).intersects(Polygon(b))
    except ImportError:
        import cv2
        area, _ = cv2.intersectConvexConvex(
            a.astype(np.float32), b.astype(np.float32))
        return area > 0


def _union_poly(polys):
    """Single polygon covering the union of overlapping polygons: the union's
    exterior ring via shapely (holes are dropped - the flattened-polygon
    representation cannot carry them), else the convex hull."""
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        u = unary_union([Polygon(p) for p in polys])
        if u.geom_type == 'MultiPolygon':   # numeric slivers: take largest
            u = max(u.geoms, key=lambda g: g.area)
        pts = np.array(u.exterior.coords[:-1])
    except ImportError:
        import cv2
        pts = cv2.convexHull(
            np.concatenate(polys).astype(np.float32)).reshape(-1, 2)
    return np.clip(pts, 0, None)


def merge_overlapping_suppression_polys(polys, max_regions):
    """Reduce clutter on frames covered by many past frames: when one frame
    carries more than ``max_regions`` suppression regions, merge each group of
    mutually-overlapping regions into a single union polygon (disjoint groups
    stay separate regions). ``polys`` is [(poly, source_name)]; returns
    [(poly, source_name, merged_count)] with merged_count 0 for untouched
    regions. ``max_regions`` <= 0 disables merging."""
    if max_regions <= 0 or len(polys) <= max_regions:
        return [(p, name, 0) for p, name in polys]
    # overlap-connected components via union-find
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if find(i) != find(j) and _polys_overlap(polys[i][0], polys[j][0]):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            p, name = polys[members[0]]
            out.append((p, name, 0))
        else:
            out.append((_union_poly([polys[k][0] for k in members]),
                        None, len(members)))
    return out

@Transformer.decorate
def find_prev_suppression_homogs_and_sizes():
    """Get suppressing frames from the previous timestep

    Only cameras with the same or an adjacent index are used to
    suppress a given camera.

    The .step call expects parameters two parameters, multihomog and
    sizes, and returns a value suitable for supplying as
    arg_suppress_boxes's suppression_homogs_and_sizes argument.

    """
    prev_multihomog = prev_sizes = prev_names = None
    output = None
    while True:
        multihomog, sizes, names = yield output
        if prev_multihomog is None or multihomog.to_id != prev_multihomog.to_id:
            output = len(multihomog) * [zero_homog_and_size]
        else:
            output = [
                (to_prev[s], prev_sizes[s], prev_names[s])
                for cam, to_prev in enumerate(diff_homogs(
                        multihomog.homogs, prev_multihomog.homogs,
                ))
                for s in [np.s_[max(0, cam - 1)
                                : min(cam + 2, len(prev_multihomog))]]
            ]
        prev_multihomog, prev_sizes, prev_names = multihomog, sizes, names

@Transformer.decorate
def find_all_suppression_homogs_and_sizes():
    frames_by_ref = {}
    output = None
    while True:
        multihomog, sizes, names = yield output
        try:
            prev_homogs, prev_sizes, prev_names = frames_by_ref[multihomog.to_id]
        except KeyError:
            prev_homogs = []
            prev_sizes, prev_names = zero_homog_and_size[1], zero_homog_and_size[2]
            output = len(multihomog) * [zero_homog_and_size]
        else:
            output = [(
                to_prev, prev_sizes, prev_names,
            ) for to_prev in diff_homogs(multihomog.homogs, prev_homogs)]
        prev_homogs.extend(multihomog.homogs)
        prev_sizes = np.concatenate([prev_sizes, sizes])
        prev_names = np.concatenate([prev_names, names])
        frames_by_ref[multihomog.to_id] = prev_homogs, prev_sizes, prev_names

@Transformer.decorate
def suppress(suppression_poly_class=None, *, past_frames,
             remove_suppressed=False, boundary_threshold=1.0,
             max_overlapping_regions=5, full_homogs=None):
    if past_frames == 'prev_neighbors':
        fshs = find_prev_suppression_homogs_and_sizes()
    elif past_frames == 'all':
        fshs = find_all_suppression_homogs_and_sizes()
    else:
        raise ValueError("Invalid value for past_frames")
    output = None
    while True:
        dhss, = yield output
        do_sets, homogs, sizes, names = zip(*dhss)
        sizes = np.array(sizes)
        names = np.array(names, dtype=object)
        multihomog = MultiHomographyF2F.from_homographyf2fs(map(wrap_F2FHomography, homogs))
        do_lists = list(map(to_DetectedObject_list, do_sets))
        prev_shs = fshs.step(multihomog, sizes, names)
        curr_shs = get_self_suppression_homogs_and_sizes(multihomog, sizes, names)
        shs = concat_suppression_homogs_and_sizes(prev_shs, curr_shs)
        if remove_suppressed:
            # Historical behavior: drop detections whose center lies in a
            # suppressing frame.  Off by default now that display-time
            # suppression is handled in DIVE via the emitted regions.
            boxes = (map(get_DetectedObject_bbox, dos) for dos in do_lists)
            keep_its = arg_suppress_boxes(boxes, shs)
        else:
            keep_its = [[True] * len(dos) for dos in do_lists]
        if suppression_poly_class is not None and boundary_threshold >= 0:
            # Detections cut off by the image boundary where another frame
            # covers that boundary are flagged with a detection attribute
            # named after the suppression class.  Coverage by any related
            # frame counts, so pair current-timestep cameras in both
            # directions, not just the suppression direction.
            all_curr_shs = get_all_other_current_homogs_and_sizes(
                multihomog, sizes, names)
            boundary_shs = concat_suppression_homogs_and_sizes(
                prev_shs, all_curr_shs)
            fh = full_homogs() if full_homogs is not None else None
            if fh:
                # The registration is a whole-survey batch, so the exported
                # map holds EVERY frame's homography - including future ones
                # a streaming process has not yet seen. Use it as the
                # boundary-check candidate set (it is a superset of past +
                # current cameras), so a detection cut off at the leading
                # edge is flagged when a FUTURE frame covers it. Frame sizes
                # are taken from the current frame of each camera (constant
                # per camera on these surveys).
                boundary_shs = []
                for cam in range(len(multihomog)):
                    self_name = os.path.basename(str(names[cam] or ''))
                    H_cam = multihomog.homogs[cam].matrix
                    hs, nm = [], []
                    for name, H_o in fh.items():
                        if name == self_name:
                            continue
                        try:
                            hs.append(np.linalg.inv(H_o) @ H_cam)
                        except np.linalg.LinAlgError:
                            continue
                        nm.append(name)
                    if hs:
                        boundary_shs.append((
                            np.stack(hs),
                            np.tile(sizes[cam], (len(hs), 1)),
                            np.array(nm, dtype=object)))
                    else:
                        boundary_shs.append(zero_homog_and_size)
            mark_boundary_suppressed(do_lists, sizes, boundary_shs,
                                     boundary_threshold,
                                     suppression_poly_class)
        if suppression_poly_class is None:
            poly_dets = [()] * len(do_lists)
        else:
            def n(p):
                # Normalize poly.  This works around DIVE issue #993
                # (https://github.com/Kitware/dive/issues/993)
                assert (p >= 0).all()
                return np.where(p, p, 0)  # Replace -0 with 0
            poly_dets = (
                (wrap_poly(n(p), suppression_poly_class, name, merged)
                 for p, name, merged in merge_overlapping_suppression_polys(
                     ps, max_overlapping_regions))
                for ps in suppression_polys(shs, sizes)
            )
        output = [
            DetectedObjectSet([*(do for k, do in zip(keep, dos) if k), *pd])
            for keep, dos, pd in zip(keep_its, do_lists, poly_dets)
        ]

class MulticamHomogDetSuppressor(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of inputs')
        add_declare_config(self, 'suppression_poly_class', '',
                           'If not empty, include polygons indicating the'
                           ' suppressed area with this class')
        add_declare_config(self, 'past_frames', 'prev_neighbors', (
            'Which past frames to use for suppression.  Valid values are'
            ' "prev_neighbors" (previous frame and same and neighboring'
            ' cameras only; this is the default) and "all" (all past frames'
            ' and cameras)'
        ))
        add_declare_config(self, 'remove_suppressed', 'false', (
            'If true, remove detections whose center lies in a suppressing'
            ' frame from the output (the historical behavior).  Off by'
            ' default: suppressed detections are left in the output and'
            ' hidden at display time (e.g. by DIVE) using the emitted'
            ' suppression regions'
        ))
        add_declare_config(self, 'boundary_threshold', '1.0', (
            'A detection whose polygon (or bounding box, if it has no'
            ' polygon) comes within this many pixels of the image boundary'
            ' is considered cut off by it.  If another frame covers that'
            ' boundary area, the detection is marked with a detection'
            ' attribute named after suppression_poly_class (set to true).'
            ' Negative disables the check'
        ))
        add_declare_config(self, 'full_homogs_file', '', (
            'Optional path to the registration node\'s export_homogs npz'
            ' (basename -> 3x3 map of EVERY frame). When set, the boundary-'
            ' cutoff check considers all frames - including FUTURE ones -'
            ' so a detection cut off at the leading edge is flagged (and'
            ' source_image-linked) when a later frame covers it. The file is'
            ' loaded lazily on first use; registration writes it during its'
            ' first step, which precedes this process\'s first step'
        ))
        add_declare_config(self, 'max_overlap_suppr_regions', '5', (
            'When a frame carries more than this many suppression regions,'
            ' each group of mutually-overlapping regions is merged into a'
            ' single Suppression detection whose polygon is their union'
            ' (disjoint groups stay separate; the merged detection carries a'
            ' merged_regions attribute with the source count). Keeps frames'
            ' covered by many past frames (past_frames=all) readable.'
            ' 0 or negative disables merging'
        ))

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        # XXX work around insufficient wrapping
        self._n_input = int(self.config_value('n_input'))
        for i in range(1, self._n_input + 1):
            add_declare_input_port(self, 'det_objs_' + str(i), 'detected_object_set',
                                   required, 'Input detected object set #' + str(i))
            add_declare_input_port(self, 'homog' + str(i), 'homography_src_to_ref',
                                   required, 'Input homography (source-to-ref) #' + str(i))
            add_declare_input_port(self, 'image' + str(i), 'image', required,
                                   'Input image #' + str(i))
            add_declare_input_port(self, 'file_name' + str(i), 'file_name',
                                   optional, 'Input image path #' + str(i)
                                   + ' (source_image attribute on regions)')
            add_declare_output_port(self, 'det_objs_' + str(i), 'detected_object_set',
                                    optional, 'Output detected object set #' + str(i))

    def _configure(self):
        # XXX actually use this
        self._n_input = int(self.config_value('n_input'))
        spc = self.config_value('suppression_poly_class') or None
        pf = self.config_value('past_frames')
        rs = self.config_value('remove_suppressed').lower() in (
            'true', '1', 'yes', 'on')
        bt = float(self.config_value('boundary_threshold'))
        try:
            mor = int(self.config_value('max_overlap_suppr_regions'))
        except (TypeError, ValueError):
            mor = 5
        fhf = self.config_value('full_homogs_file') or None
        self._full_homogs_cache = None

        def _load_full_homogs(_self=self, _path=fhf):
            if _path is None:
                return None
            if _self._full_homogs_cache is None:
                if not os.path.exists(_path):
                    return None
                try:
                    with np.load(_path) as z:
                        _self._full_homogs_cache = {
                            n: z[n] for n in z.files}
                except (OSError, ValueError):
                    return None
            return _self._full_homogs_cache

        self._suppressor = suppress(spc, past_frames=pf, remove_suppressed=rs,
                                    boundary_threshold=bt,
                                    max_overlapping_regions=mor,
                                    full_homogs=_load_full_homogs)
        # Determined on the first step (edges are present by then): which
        # optional file_name ports are wired, for the source_image attribute.
        self._fn_connected = None
        self._base_configure()

    def _step(self):
        def get_image_size(im):
            return im.width(), im.height()
        if self._fn_connected is None:
            self._fn_connected = [
                self.has_input_port_edge_using_trait('file_name' + str(i))
                for i in range(1, self._n_input + 1)]
        entries = []
        for i in range(1, self._n_input + 1):
            name = (self.grab_input_using_trait('file_name' + str(i))
                    if self._fn_connected[i - 1] else '')
            entries.append((
                self.grab_input_using_trait('det_objs_' + str(i)),
                self.grab_input_using_trait('homog' + str(i)),
                get_image_size(self.grab_input_using_trait('image' + str(i))),
                name,
            ))
        dos_list = self._suppressor.step(entries)
        for i, dos in enumerate(dos_list, 1):
            self.push_to_port_using_trait('det_objs_' + str(i), dos)
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.MulticamHomogDetSuppressor'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'multicam_homog_det_suppressor',
        'Multi-camera homography-based detection suppressor',
        MulticamHomogDetSuppressor,
    )
    process_factory.mark_process_module_as_loaded(module_name)
