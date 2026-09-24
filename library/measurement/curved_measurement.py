# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
"""Opt-in curved length measurement for calibrated, rectified stereo pairs.

Disparities here are floating point PIXELS: left x - right x, including for
right-reference disparity. Curves and masks must use the disparity image grid.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def resample_curve(points, count=32, smoothing=0.5):
    """Smooth a head-to-tail polyline in pixels, retaining its endpoints."""
    from scipy.interpolate import splprep, splev
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('A curve needs at least two finite [x, y] points')
    if not 4 <= count <= 512 or not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError('samples must be 4..512 and smoothing must be nonnegative')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if len(points) < 2:
        raise ValueError('Curve has zero length')
    tck, _ = splprep(points.T, s=len(points) * smoothing ** 2, k=min(3, len(points) - 1))
    dense = np.array(splev(np.linspace(0, 1, max(1024, count * 8)), tck)).T
    dense[0], dense[-1] = points[0], points[-1]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    return np.column_stack([np.interp(np.linspace(0, distance[-1], count), distance, dense[:, i])
                            for i in range(2)])


def resample_polyline(points, count=32):
    """Sample an editable polyline without smoothing away vertices or corners."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 512 or not np.isfinite(points).all():
        raise ValueError('A centerline needs 2..512 finite [x,y] vertices')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] <= 0:
        raise ValueError('Centerline has zero length')
    positions = np.unique(np.r_[distance, np.linspace(0, distance[-1], count)])
    return np.column_stack([np.interp(positions, distance, points[:, i]) for i in range(2)])


def centerline_keypoints(points):
    """Named markers accepted by VIAME CSV and DIVE's existing line editor."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Invalid centerline vertices')
    return {('head' if i == 0 else 'tail' if i == len(points) - 1 else 'spine_%03d' % i): p.tolist()
            for i, p in enumerate(points)}


def merge_components(mask, min_fraction=0.05):
    """(merged, union) of a mask that came as several polygons. union keeps the
    components holding at least min_fraction of the largest one's area; merged
    fills the gap between the facing edges of nearest neighbours so one path
    can cross them. Bridges exist for connectivity only and are absent from
    union."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from viame import image_kernels
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        raise ValueError('Mask is empty')
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    keep = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] >= min_fraction * sizes.max()]
    union = ndimage.binary_fill_holes(np.isin(labels, keep))
    merged = union.copy()
    if len(keep) == 1:
        return merged, union
    pixels = {i: np.column_stack(np.nonzero(labels == i))[:, ::-1] for i in keep}
    joined, pending = [keep[0]], keep[1:]
    raster = merged.astype(np.uint8)
    while pending:
        tree = cKDTree(np.vstack([pixels[i] for i in joined]))
        gaps = {i: tree.query(pixels[i]) for i in pending}
        i = min(pending, key=lambda k: gaps[k][0].min())
        distance, index = gaps[i]
        facing = distance <= 1.25 * distance.min() + 1
        hull = image_kernels.convex_hull(
            np.vstack([pixels[i][facing], tree.data[index[facing]]]))
        # `fill_polygon` fills the outline too, which on a convex hull is
        # exactly what `cv2.fillConvexPoly` does.
        image_kernels.fill_polygon(raster, hull, 1)
        joined.append(i)
        pending.remove(i)
    return raster > 0, union


def mask_end_seeds(mask):
    """Rough head and tail of a mask: the pixels furthest out along its major
    axis, head on the larger x as the keypoint pipelines order them."""
    xy = np.column_stack(np.nonzero(np.asarray(mask, dtype=bool)))[:, ::-1].astype(float)
    if len(xy) < 2:
        raise ValueError('Mask is empty')
    centered = xy - xy.mean(axis=0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    along = centered @ axis
    low, high = np.percentile(along, [1, 99])
    ends = np.array([xy[along <= low].mean(axis=0), xy[along >= high].mean(axis=0)])
    return ends if ends[0, 0] >= ends[1, 0] else ends[::-1]


def _ridge_path(mask, distance, start, end):
    from scipy.spatial import cKDTree
    from skimage.graph import route_through_array
    xy = np.column_stack(np.nonzero(mask))[:, ::-1]
    a, b = xy[cKDTree(xy).query([start, end])[1]]
    if (a == b).all():
        raise ValueError('Head and tail do not define a connected mask path')
    cost = np.where(mask, (distance.max() / np.maximum(distance, 0.5)) ** 2, -1.0)
    try:
        path, _ = route_through_array(cost, (a[1], a[0]), (b[1], b[0]),
                                      fully_connected=True, geometric=True)
    except ValueError:
        raise ValueError('Head and tail do not define a connected mask path')
    return np.asarray(path, dtype=float)[:, ::-1]


def _trim(path, distance, anchor, limit):
    """Leading path points drop while the local half width still reaches the
    anchor: there the path is swinging from the mask edge onto the ridge."""
    i = 0
    while i < limit:
        x, y = path[i].astype(int)
        if np.linalg.norm(path[i] - anchor) >= 1.5 * distance[y, x]:
            break
        i += 1
    return i


def _exit_point(mask, trunk, distance):
    """Where the tangent at the start of trunk leaves the mask."""
    x, y = trunk[0].astype(int)
    reach = max(3.0 * distance[y, x], 10.0)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(trunk, axis=0), axis=1))]
    near = trunk[arc <= reach]
    if len(near) < 2:
        return trunk[0]
    direction = near[0] - near[-1]
    direction /= np.linalg.norm(direction)
    h, w = mask.shape
    point = trunk[0]
    for step in np.arange(0.5, 4 * max(h, w), 0.5):
        probe = trunk[0] + step * direction
        px, py = int(round(probe[0])), int(round(probe[1]))
        if not (0 <= px < w and 0 <= py < h) or not mask[py, px]:
            break
        point = probe
    return point


def mask_centerline(mask, endpoints, anchored=True):
    """Head-to-tail path along the ridge of a mask's distance transform; fins
    are side branches the minimal path never enters.

    endpoints say where the path starts and stops. anchored keeps them as the
    first and last points; otherwise they only seed the path and each end is
    where the trunk's own tangent leaves the mask. The mask must be connected
    between them: see merge_components for masks made of several polygons.
    """
    from scipy import ndimage
    mask = np.asarray(mask, dtype=bool)
    endpoints = np.asarray(endpoints, dtype=float)
    if mask.ndim != 2 or endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
        raise ValueError('mask must be 2D and endpoints must be head/tail [x,y]')
    if mask.sum() < 2:
        raise ValueError('Mask has no usable centerline')
    distance = ndimage.distance_transform_edt(mask)
    path = _ridge_path(mask, distance, endpoints[0], endpoints[1])
    limit = len(path) // 4
    head = _trim(path, distance, path[0], limit)
    tail = _trim(path[::-1], distance, path[-1], limit)
    trunk = path[head:len(path) - tail]
    if len(trunk) < 2:
        trunk = path
    if not anchored:
        endpoints = np.array([_exit_point(mask, trunk, distance),
                              _exit_point(mask, trunk[::-1], distance)])
    return np.vstack([endpoints[0], trunk, endpoints[1]])


def fit_midline(path, keep=None, harmonics=3, count=512):
    """Smooth curve pinned to the first and last path points: the straight
    line between them plus a short sine series fitted to the body of the path.

    A sine series vanishes at both ends, so head and tail are met exactly, and
    its few terms cannot follow the swing a ridge path makes from a tail tip or
    snout onto the ridge. The outer tenth of the path at each end and points
    excluded by keep do not enter the fit."""
    path = np.asarray(path, dtype=float)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if len(path) < 2 or arc[-1] <= 0:
        raise ValueError('Curve has zero length')
    u = arc / arc[-1]
    use = (u > 0.1) & (u < 0.9)
    if keep is not None:
        use &= np.asarray(keep, dtype=bool)
    harmonics = min(harmonics, int(use.sum()) // 2)
    k = np.arange(1, harmonics + 1)
    line = lambda t: path[0] + np.outer(t, path[-1] - path[0])
    dense = np.linspace(0, 1, count)
    if harmonics < 1:
        return line(dense)
    coefficients = np.linalg.lstsq(np.sin(np.pi * np.outer(u[use], k)),
                                   path[use] - line(u[use]), rcond=None)[0]
    return line(dense) + np.sin(np.pi * np.outer(dense, k)) @ coefficients


def sample_map(array, points):
    """Bilinear sampling; outside pixels and any invalid contributing value fail."""
    from scipy.ndimage import map_coordinates
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError('Disparity/mask must be 2D')
    return map_coordinates(array, np.asarray(points).T[::-1], order=1,
                           mode='constant', cval=np.nan, prefilter=False)


def measure_curve(left_curve, left_disparity, calibration, *, right_curve=None,
                  right_disparity=None, left_mask=None, right_mask=None,
                  mode='left', samples=32, smoothing=0.5, consistency_px=1.5,
                  centerline_tolerance_px=5.0, max_length_disagreement=0.1,
                  max_depth_ratio=2.0):
    """Measure one fish; never bridge failed samples or silently fall back.

    calibration: rectified=true, fx, fy, cx_left, cx_right, cy, baseline (>0).
    Both rectified cameras share fx/fy/cy; baseline units determine length units.
    Bidirectional mode averages independently sampled curve LENGTHS, only after
    round-trip and length agreement checks. It does not average image coordinates.
    """
    if mode not in ('left', 'bidirectional'):
        raise ValueError('mode must be left or bidirectional')
    if calibration.get('rectified') is not True:
        raise ValueError('Explicit rectified calibration is required')
    fx, fy, cx, cr, cy, baseline = [float(calibration[k]) for k in
                                   ('fx', 'fy', 'cx_left', 'cx_right', 'cy', 'baseline')]
    if not np.isfinite([fx, fy, cx, cr, cy, baseline]).all() or min(fx, fy, baseline) <= 0:
        raise ValueError('Invalid rectified intrinsics or baseline')
    limits = [consistency_px, centerline_tolerance_px, max_length_disagreement, max_depth_ratio]
    if not np.isfinite(limits).all() or min(limits) <= 0 or max_depth_ratio < 1:
        raise ValueError('Quality thresholds must be finite and positive; depth ratio >= 1')
    left_disparity = np.asarray(left_disparity, dtype=float)
    if left_disparity.ndim != 2:
        raise ValueError('Disparity must be 2D')
    for value in (right_disparity, left_mask, right_mask):
        if value is not None and np.asarray(value).shape != left_disparity.shape:
            raise ValueError('All disparity maps and masks must use the same image grid')
    if mode == 'bidirectional' and (right_curve is None or right_disparity is None):
        raise ValueError('bidirectional requires right_curve and independent right_disparity')
    left = resample_curve(left_curve, samples, smoothing)
    right = None if right_curve is None else resample_curve(right_curve, samples, smoothing)

    def direction(source, disparity, reverse, source_mask, target_mask, target, from_right):
        from scipy.spatial import cKDTree
        disp = sample_map(np.where(np.isfinite(disparity) & (np.asarray(disparity) > 0),
                                   disparity, np.nan), source)
        mapped = source.copy()
        mapped[:, 0] += disp if from_right else -disp
        valid = np.isfinite(disp) & (disp > 0)
        h, w = left_disparity.shape
        valid &= (mapped[:, 0] >= 0) & (mapped[:, 0] <= w - 1)
        valid &= (mapped[:, 1] >= 0) & (mapped[:, 1] <= h - 1)
        for mask, points in ((source_mask, source), (target_mask, mapped)):
            if mask is not None:
                valid &= sample_map(np.asarray(mask, dtype=bool), points) >= 0.5
        cycle_error = None
        if reverse is not None:
            backward = sample_map(np.where(np.isfinite(reverse) & (np.asarray(reverse) > 0),
                                               reverse, np.nan), mapped)
            cycle_error = np.abs(disp - backward)
            valid &= np.isfinite(backward) & (backward > 0) & (cycle_error <= consistency_px)
        if target is not None:
            # Compare to a densely sampled target curve, not equal index/fraction.
            dense_target = resample_curve(target, 512, 0)
            finite = np.isfinite(mapped).all(axis=1)
            distances = np.full(len(source), np.inf)
            distances[finite] = cKDTree(dense_target).query(mapped[finite])[0]
            valid &= distances <= centerline_tolerance_px
        denominator = disp - (cx - cr)
        valid &= denominator > 0
        if not valid.all():
            return dict(success=False, valid_samples=int(valid.sum()), total_samples=len(source),
                        error='Incomplete or inconsistent correspondence; length withheld')
        lp = mapped if from_right else source
        z = fx * baseline / denominator
        if z.max() / z.min() > max_depth_ratio:
            return dict(success=False, error='Depth range exceeds max_depth_ratio')
        xyz = np.column_stack(((lp[:, 0] - cx) * z / fx, (lp[:, 1] - cy) * z / fy, z))
        # An explicit polyline estimate avoids spline overshoot in 3D. Image
        # curve smoothing is controlled above; raw depth noise remains visible.
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
        if not np.isfinite(length) or length <= 0 or chord <= 0:
            return dict(success=False, error='Degenerate reconstructed curve')
        return dict(success=True, length=length, straight_length=chord,
                    points_3d=xyz.tolist(), source_points=source.tolist(),
                    matched_points=mapped.tolist(), valid_samples=len(source),
                    max_cycle_error_px=None if cycle_error is None else float(cycle_error.max()))

    forward = direction(left, left_disparity, right_disparity, left_mask, right_mask, right, False)
    result = dict(success=forward['success'], method=mode, left=forward)
    if not forward['success']:
        return result
    length = forward['length']
    if mode == 'bidirectional':
        backward = direction(right, right_disparity, left_disparity, right_mask, left_mask, left, True)
        result['right'] = backward
        if not backward['success']:
            result['success'] = False
            return result
        disagreement = abs(length - backward['length']) / max(length, backward['length'])
        result['relative_length_disagreement'] = disagreement
        if disagreement > max_length_disagreement:
            result.update(success=False, error='Left/right curve lengths disagree')
            return result
        length = (length + backward['length']) / 2
    result['left_keypoints'] = centerline_keypoints(forward['source_points'])
    result['right_keypoints'] = centerline_keypoints(forward['matched_points'])
    result.update(curved_length=length, straight_length=forward['straight_length'],
                  curvature_ratio=length / forward['straight_length'])
    return result


def request_measurement(request, left_disparity, right_disparity=None):
    """Shared service/CLI adapter: explicit curves or binary mask paths + endpoints."""
    from viame.utilities import imageops

    curves, masks = {}, {}
    for side in ('left', 'right'):
        masks[side] = None
        if request.get(side + '_mask_path'):
            try:
                mask = imageops.read_image(request[side + '_mask_path'],
                                           grayscale=True)
            except OSError as exc:
                raise ValueError('Cannot read ' + side + ' mask') from exc
            masks[side] = mask > 0
        curves[side] = request.get(side + '_curve')
        if curves[side] is None and masks[side] is not None:
            curves[side] = mask_centerline(masks[side], request[side + '_endpoints'])
    if curves['left'] is None:
        raise ValueError('left_curve or left_mask_path with left_endpoints is required')
    return measure_curve(curves['left'], left_disparity, request['rectified_calibration'],
                         right_curve=curves['right'], right_disparity=right_disparity,
                         left_mask=masks['left'], right_mask=masks['right'],
                         **request.get('options', {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', help='JSON manifest; see curved_measurement documentation')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    scale = float(request.get('disparity_scale', 1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('disparity_scale must be positive (256 for VIAME uint16 disparities)')
    left = np.load(request['left_disparity_path'], allow_pickle=False) / scale
    right = None
    if request.get('right_disparity_path'):
        right = np.load(request['right_disparity_path'], allow_pickle=False) / scale
    result = request_measurement(request, left, right)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
