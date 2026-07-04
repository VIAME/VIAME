#!/usr/bin/env python3
"""detect_prior_coverage.py - Per-frame previously-observed image regions.

For every image of a (multi-camera) aerial survey, identifies the region that
was already observed in previous imagery and reports it as polygons in the
standard VIAME detection-CSV format. "Previously observed" covers three
provenances, output as separate classes so downstream consumers can filter:

  <class>_sequential    seen by the SAME camera in recent preceding frames
  <class>_cross_camera  seen by ANOTHER rig camera (PORT/CENTER/STAR overlap)
  <class>_revisit       seen further in the past: an earlier survey pass, a
                        loop closure, or (multi-folder runs) an earlier site
                        or day

Approach (method "hybrid", the default):

  1. Metadata: flight-log rows are linked to images by the per-day trigger
     counter embedded in SSL filenames (survey_metadata.py); 2025 UAS imagery
     uses embedded EXIF GPS instead. Gives per-frame GPS/altitude/attitude.
  2. Within-camera affine registration chains + rig-constant cross-camera
     transforms (cluster consensus) give precise pixel-level geometry for
     the recent-overlap classes. These reuse the proven machinery in
     viame.opencv.registration_utils (affine model, adaptive matching, GPS
     dead-reckoning fill for feature-poor open-water frames).
  3. A global ground-occupancy grid in local ENU metres tracks everything
     ever seen. Each new image is mapped into ENU via a per-frame transform
     built from its GPS fix, the GPS-track heading and the calibrated
     metres->pixels similarity - so revisits are found even when the
     registration chain cannot connect them (e.g. after a long open-water
     gap or on a different day). Revisit overlaps between two land frames
     are optionally CONFIRMED by direct feature registration.

Other methods: "metadata" needs no image registration at all (footprints
purely from GPS + altitude + focal length; fast, works over open water,
less precise) and "sfm-rig" uses COLMAP incremental SfM with a fixed
multi-camera rig configuration (experimental; requires pycolmap>=3.12).

Multiple site folders can be processed in one run against a shared coverage
grid, enabling cross-site / cross-day revisit detection:

  python detect_prior_coverage.py SITE_A [SITE_B ...] --flight-logs <dir>
  python detect_prior_coverage.py --all <root> --flight-logs <dir>

Outputs per site (in --output, default <site>_coverage):
  prior_coverage.csv       VIAME CSV; polygon rows per class as above
  revisits.csv             summary of detected revisit events
  coverage_map.png         ENU map of footprints coloured by pass/order
  prior_coverage_vis.png   thumbnail grid with coverage polygons overlaid
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# Tool scripts are installed side by side; allow running from the source tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import survey_metadata as smd

from viame.opencv import registration_utils as _sr
_sr.import_dependencies()

from viame.opencv.registration_utils import (
    compute_homography_pair, _compute_camera_chain,
    _poses_to_enu, _track_headings, _rot2,
    _geo_calibrate,
)

CAM_ORDER = {'CENTER': 0, 'PORT': 1, 'STAR': 2, None: 0}

VIAME_CSV_HEADER = (
    '# 1: Detection or Track-id,  2: Video or Image Identifier,  '
    '3: Unique Frame Identifier,  4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y),  '
    '8: Detection or Length Confidence,  9: Target Length,  '
    '10-11+: Repeated Species, Confidence Pairs or Attributes')


# ---------------------------------------------------------------------------
# Global ENU coverage grid
# ---------------------------------------------------------------------------

class CoverageGrid:
    """Sparse ground-occupancy grid in local ENU metres.

    Each cell stores the global order index of the FIRST observation that
    covered it (-1 = never seen). Tiles are allocated lazily so multi-km
    coastlines stay cheap.
    """

    TILE = 256

    def __init__(self, cell_m=1.0):
        self.cell = float(cell_m)
        self.tiles = {}

    def _tile(self, tx, ty, create=False):
        t = self.tiles.get((tx, ty))
        if t is None and create:
            t = np.full((self.TILE, self.TILE), -1, dtype=np.int32)
            self.tiles[(tx, ty)] = t
        return t

    def lookup(self, pts):
        """pts (N,2) ENU metres -> (N,) first-observer order index (-1 unseen)."""
        c = np.floor(np.asarray(pts) / self.cell).astype(np.int64)
        out = np.full(len(c), -1, dtype=np.int32)
        tx = c[:, 0] // self.TILE
        ty = c[:, 1] // self.TILE
        for key in set(zip(tx.tolist(), ty.tolist())):
            t = self.tiles.get(key)
            if t is None:
                continue
            m = (tx == key[0]) & (ty == key[1])
            lx = (c[m, 0] - key[0] * self.TILE)
            ly = (c[m, 1] - key[1] * self.TILE)
            out[m] = t[ly, lx]
        return out

    def stamp_polygon(self, quad_enu, order_idx):
        """Mark all cells inside the polygon as observed (keep first observer)."""
        import cv2
        q = np.asarray(quad_enu, dtype=np.float64) / self.cell
        lo = np.floor(q.min(axis=0)).astype(np.int64) - 1
        hi = np.ceil(q.max(axis=0)).astype(np.int64) + 1
        w, h = int(hi[0] - lo[0]), int(hi[1] - lo[1])
        if w <= 0 or h <= 0 or w * h > 64e6:
            return
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(q - lo).astype(np.int32)], 1)
        ys, xs = np.nonzero(mask)
        cx, cy = xs + lo[0], ys + lo[1]
        tx, ty = cx // self.TILE, cy // self.TILE
        for key in set(zip(tx.tolist(), ty.tolist())):
            m = (tx == key[0]) & (ty == key[1])
            t = self._tile(key[0], key[1], create=True)
            lx, ly = cx[m] - key[0] * self.TILE, cy[m] - key[1] * self.TILE
            sel = t[ly, lx] < 0
            t[ly[sel], lx[sel]] = order_idx


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def _apply_h(H, pts):
    """Apply 3x3 homography to (N,2) points."""
    p = np.asarray(pts, dtype=np.float64)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def _image_rect(w, h):
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    dtype=np.float64)


def _clip_poly_to_rect(poly, w, h):
    """Sutherland-Hodgman clip of polygon (N,2) to [0,w-1]x[0,h-1]."""
    def clip_edge(pts, inside, intersect):
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ia, ib = inside(a), inside(b)
            if ia:
                out.append(a)
                if not ib:
                    out.append(intersect(a, b))
            elif ib:
                out.append(intersect(a, b))
        return out

    def make(axis, lim, keep_low):
        def inside(p):
            return p[axis] >= lim if keep_low else p[axis] <= lim

        def intersect(a, b):
            t = (lim - a[axis]) / (b[axis] - a[axis])
            return a + t * (b - a)
        return inside, intersect

    pts = [np.asarray(p, dtype=np.float64) for p in poly]
    for axis, lim, keep_low in ((0, 0.0, True), (0, w - 1.0, False),
                                (1, 0.0, True), (1, h - 1.0, False)):
        if not pts:
            return None
        pts = clip_edge(pts, *make(axis, lim, keep_low))
    if len(pts) < 3:
        return None
    return np.array(pts)


def _sane_relative(R, max_aniso=1.5, scale_range=(0.6, 1.6)):
    """Sanity check for a relative image-to-image transform: near-nadir
    frames at constant altitude must map by a near-similarity with scale
    close to 1. Garbage water-frame registrations fail this (degenerate
    slivers / extreme skew)."""
    A = np.asarray(R)[:2, :2]
    if not np.all(np.isfinite(A)):
        return False
    s = np.linalg.svd(A, compute_uv=False)
    if s[1] <= 1e-6:
        return False
    scale = float(np.sqrt(s[0] * s[1]))
    return (s[0] / s[1] <= max_aniso
            and scale_range[0] <= scale <= scale_range[1])


def _poly_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _fmt_poly(poly):
    return '(poly) ' + ' '.join(f'{v:.1f}' for p in poly for v in p)


# ---------------------------------------------------------------------------
# Per-site processing
# ---------------------------------------------------------------------------

class Observation:
    """One image with everything needed for coverage reasoning."""

    __slots__ = ('order', 'site_id', 'site_tag', 'site_dir', 'cam', 'frame',
                 'rel', 'width', 'height', 'T_enu', 'chain_H', 'is_water',
                 'pass_no', 'day', 'has_gps', 'timestep')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _xcam_consensus(site_folder, cams, chains, frames_by_cam, water_info,
                    reg_args, trials=15, tol=300.0, verbose=True):
    """Rig-constant PORT->CENTER / STAR->CENTER transforms by mode-seeking
    cluster consensus over same-trigger image pairs (robust to >50% garbage
    estimates from water frames)."""
    out = {}
    if 'CENTER' not in cams:
        return out
    center_frames = frames_by_cam['CENTER']
    for cam in ('PORT', 'STAR'):
        if cam not in cams:
            continue
        common = sorted(set(frames_by_cam[cam]) & set(center_frames))

        def _quality(f):
            ic = center_frames.index(f)
            ix = frames_by_cam[cam].index(f)
            wc = water_info.get(cams['CENTER'][ic], {}).get('is_water', False)
            wx = water_info.get(cams[cam][ix], {}).get('is_water', False)
            return (0 if (not wc and not wx) else (1 if not wc or not wx else 2))

        common.sort(key=_quality)
        cand = common[:max(trials * 2, trials)]
        cand = cand[::max(1, len(cand) // trials)][:trials]
        ests = []
        for f in cand:
            ix = frames_by_cam[cam].index(f)
            ic = center_frames.index(f)
            H, _ = compute_homography_pair(
                os.path.join(site_folder, cams[cam][ix]),
                os.path.join(site_folder, cams['CENTER'][ic]),
                **reg_args)
            if H is not None:
                ests.append(H)
        if not ests:
            if verbose:
                print(f'    {cam}->CENTER: no direct matches (open water?)')
            continue
        # Mode-seeking: densest cluster of translations within `tol` px.
        t = np.array([[H[0, 2], H[1, 2]] for H in ests])
        best, support = None, 0
        for i in range(len(ests)):
            d = np.linalg.norm(t - t[i], axis=1)
            m = d <= tol
            if m.sum() > support:
                support, best = int(m.sum()), m
        H_avg = np.mean(np.stack([H for H, k in zip(ests, best) if k]), axis=0)
        H_avg /= H_avg[2, 2]
        out[cam] = H_avg
        if verbose:
            print(f'    {cam}->CENTER: consensus from {support}/{len(ests)} '
                  f'pair estimates')
    return out


def _fill_unchained_no_gps(chain, n, window=5):
    """No-metadata fallback for unchained (water) frames: carry the moving
    average of recent chained per-step translations forward/backward
    (interior gaps blend both directions). Orientation/scale are held from
    the nearest chained frame."""
    reg = sorted(chain.keys())
    if len(reg) < 2:
        return 0
    steps = {}
    for a, b in zip(reg[:-1], reg[1:]):
        if b - a >= 1:
            d = (np.array([chain[b][0, 2], chain[b][1, 2]])
                 - np.array([chain[a][0, 2], chain[a][1, 2]])) / (b - a)
            steps[a] = d

    def _avg_step(near, direction):
        keys = [k for k in steps if (k <= near if direction < 0 else k >= near)]
        keys.sort(key=lambda k: abs(k - near))
        sel = [steps[k] for k in keys[:window]]
        return np.mean(sel, axis=0) if sel else None

    filled = 0
    for k in range(n):
        if k in chain:
            continue
        prev = max((j for j in reg if j < k), default=None)
        nxt = min((j for j in reg if j > k), default=None)
        est = []
        if prev is not None:
            v = _avg_step(prev, -1)
            if v is not None:
                p = np.array([chain[prev][0, 2], chain[prev][1, 2]])
                est.append((abs(k - prev), p + v * (k - prev), prev))
        if nxt is not None:
            v = _avg_step(nxt, +1)
            if v is not None:
                p = np.array([chain[nxt][0, 2], chain[nxt][1, 2]])
                est.append((abs(nxt - k), p - v * (nxt - k), nxt))
        if not est:
            continue
        if len(est) == 2:
            (da, pa, ja), (db, pb, jb) = est
            wsum = da + db
            pos = pa * (db / wsum) + pb * (da / wsum)
            src = ja if da <= db else jb
        else:
            _, pos, src = est[0]
        H = chain[src].copy()
        H[0, 2], H[1, 2] = float(pos[0]), float(pos[1])
        chain[k] = H
        filled += 1
    return filled


def _expected_px_per_m(poses):
    """Physically-expected GSD scale (px/m) from metadata: a full-frame
    sensor at altitude A with 35mm-equivalent focal f images a ground width
    of A*36/f over `width` pixels."""
    alts = [p['alt_agl'] for p in poses.values() if p.get('alt_agl')]
    if not alts:
        return None
    alt = float(np.median(alts))
    if alt < 10:
        return None
    p0 = next(iter(poses.values()))
    f35 = p0.get('focal35_mm') or 85.0
    width = p0.get('width') or 5168
    return width / (alt * smd.SENSOR_W_MM / f35)


def _geo_anchor_with_cal(cam_chains, cams, poses_by_cam, pairwise_by_cam,
                         verbose=True):
    """Like registration_utils._geo_anchor_cameras but returns the per-camera
    calibration (M, enu, yaw) needed to build pixel->ENU transforms, and
    bounds the fitted scale by the metadata-expected GSD (few clean pairwise
    steps on water-heavy sites otherwise corrupt the scale by 50%+)."""
    from viame.opencv.registration_utils import _geo_fill
    cal = {}
    for cam in cams:
        if poses_by_cam.get(cam) is None:
            continue
        M, n, r, enu, yaw = _geo_calibrate(
            cam_chains.get(cam, {}), cams[cam],
            poses_by_cam[cam], pairwise_by_cam.get(cam))
        cal[cam] = {'M': M, 'n': n, 'res': r, 'enu': enu, 'yaw': yaw,
                    'expect': _expected_px_per_m(poses_by_cam[cam])}
    good = [np.sqrt(abs(np.linalg.det(c['M']))) for c in cal.values()
            if c['M'] is not None and c['n'] >= 8
            and c['res'] is not None and c['res'] < 150]
    shared = float(np.median(good)) if good else None
    for cam, c in cal.items():
        target = None       # rig-consensus scale first, metadata GSD second
        reliable = (c['M'] is not None and c['n'] >= 8
                    and c['res'] is not None and c['res'] < 150)
        if c['M'] is None:
            ref = shared or c['expect']
            if ref is not None and not np.all(np.isnan(c['yaw'])):
                # No usable pairwise steps at all (e.g. all-water camera):
                # synthesize M assuming the standard mounting (image up =
                # flight direction).
                c['M'] = np.array([[ref, 0.0], [0.0, -ref]])
                c['borrowed'] = True
            else:
                continue
        elif not reliable:
            target = shared or c['expect']
        else:
            # Even a "reliable" fit is distrusted when it disagrees with
            # physics by >30% - altitude and focal length are well known.
            own = np.sqrt(abs(np.linalg.det(c['M'])))
            if c['expect'] and abs(own / c['expect'] - 1.0) > 0.3:
                target = c['expect']
        if target is not None:
            own = np.sqrt(abs(np.linalg.det(c['M'])))
            if own > 1e-6:
                c['M'] = c['M'] * (target / own)
                c['borrowed'] = True
        if verbose:
            note = f"{cam}{'*' if c.get('borrowed') else ''}"
        else:
            note = ''
        _geo_fill(cam_chains.get(cam, {}), cams[cam], c['enu'], c['yaw'],
                  c['M'], label=note, n_steps=c['n'], residual=c['res'])
    return cal


def _pixel_to_enu_transform(enu_xy, yaw_deg, M, width, height, origin_off):
    """Per-frame affine pixel->global-ENU built from the GPS fix, GPS-track
    heading and the calibrated heading-frame-metres -> pixels map M.

    x_ground = x_plane + R(yaw) @ inv(M) @ (p - image_centre)
    """
    if M is None or enu_xy is None or np.any(np.isnan(enu_xy)):
        return None
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    yaw = 0.0 if yaw_deg is None or np.isnan(yaw_deg) else yaw_deg
    A = _rot2(yaw) @ Minv
    c = np.array([width / 2.0, height / 2.0])
    t = np.asarray(enu_xy) + np.asarray(origin_off) - A @ c
    T = np.eye(3)
    T[:2, :2] = A
    T[:2, 2] = t
    return T


def _metadata_transform(rec, heading, width, height, to_enu,
                        lateral_frac=0.0):
    """Pixel->ENU affine from metadata only (nadir assumption)."""
    if rec.get('lat') is None or to_enu is None:
        return None
    x, y = to_enu(rec['lat'], rec['lon'])   # already in the shared frame
    quad = smd.footprint_quad_enu(
        x, y, rec.get('alt_agl'),
        heading, rec.get('focal35_mm') or 85.0, lateral_frac)
    import cv2
    src = _image_rect(width, height).astype(np.float32)
    dst = np.array(quad, dtype=np.float32)
    T = cv2.getPerspectiveTransform(src, dst)
    return T


# ---------------------------------------------------------------------------
# Coverage engine
# ---------------------------------------------------------------------------

def compute_coverage(observations, grid, args, chains, xcam, frames_by_cam,
                     idx_by_cam, site_folder, water_info, obs_registry):
    """Walk observations in acquisition order; for each, report previously
    observed regions (precise tier for recent/same-trigger overlap, ENU grid
    tier for revisits), then stamp its own footprint into the grid.

    Returns (rows, revisit_events, frac_prior) where rows are
    (rel, class, polygon) tuples in order.
    """
    import cv2
    rows = []
    revisit_events = []
    frac_prior = {}
    stride = max(8, int(args.query_stride))
    pair_cache = {}

    # The registry spans ALL sites processed so far, so grid cells stamped by
    # an earlier site/day resolve to their true source observation.
    obs_by_order = obs_registry

    for o in observations:
        w, h = o.width, o.height
        rect = _image_rect(w, h)
        polys = []      # (class_suffix, poly, source_order or None)

        # ---- Tier 1: precise recent overlap via registration chains ----
        cam_chain = chains.get(o.cam)
        my_idx = idx_by_cam[o.cam].get(o.frame)
        H_cur = cam_chain.get(my_idx) if (cam_chain and my_idx is not None) \
            else None
        if H_cur is not None:
            try:
                H_cur_inv = np.linalg.inv(H_cur)
            except np.linalg.LinAlgError:
                H_cur_inv = None
        else:
            H_cur_inv = None

        def _rel_transform(prior):
            """3x3 mapping prior-image pixels -> current-image pixels."""
            if H_cur_inv is None:
                return None
            p_idx = idx_by_cam[prior.cam].get(prior.frame)
            p_chain = chains.get(prior.cam)
            if p_chain is None or p_idx not in p_chain:
                return None
            if prior.cam == o.cam:
                return H_cur_inv @ p_chain[p_idx]
            # Through CENTER anchor space via rig-constant cross-cam maps.
            Hx_cur = np.eye(3) if o.cam == 'CENTER' else xcam.get(o.cam)
            Hx_pri = np.eye(3) if prior.cam == 'CENTER' else xcam.get(prior.cam)
            if Hx_cur is None or Hx_pri is None:
                return None
            c_chain = chains.get('CENTER')
            ci = idx_by_cam['CENTER'].get(o.frame)
            cj = idx_by_cam['CENTER'].get(prior.frame)
            if c_chain is None or ci not in c_chain or cj not in c_chain:
                return None
            try:
                # prior px -> CENTER px (same trigger) -> CENTER anchor ->
                # CENTER px at current trigger -> current cam px.
                return (np.linalg.inv(Hx_cur) @ np.linalg.inv(c_chain[ci])
                        @ c_chain[cj] @ Hx_pri)
            except np.linalg.LinAlgError:
                return None

        recent = [p for p in observations
                  if p.order < o.order
                  and abs(o.timestep - p.timestep) <= args.window]
        for prior in recent:
            R = _rel_transform(prior)
            if R is None or not _sane_relative(R):
                continue
            proj = _apply_h(R, _image_rect(prior.width, prior.height))
            clipped = _clip_poly_to_rect(proj, w, h)
            if clipped is None or _poly_area(clipped) < args.min_area_px:
                continue
            suffix = ('sequential' if prior.cam == o.cam else 'cross_camera')
            polys.append((suffix, clipped, prior.order))

        # ---- Tier 2: ENU grid for everything tier 1 could not see ----
        # The grid categorizes every already-observed cell; cells whose first
        # observer was precisely handled by tier 1 above are skipped so the
        # imprecise (GPS-level) polygons never duplicate the precise ones.
        tier1_orders = {src for _sfx, _poly, src in polys}
        gx = None
        seen_any = None
        if o.T_enu is not None:
            xs = np.arange(stride // 2, w, stride, dtype=np.float64)
            ys = np.arange(stride // 2, h, stride, dtype=np.float64)
            gx, gy = np.meshgrid(xs, ys)
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            enu_pts = _apply_h(o.T_enu, pts)
            owner = grid.lookup(enu_pts).reshape(gy.shape)
            seen_any = owner >= 0
            cat_masks = {'sequential': np.zeros(owner.shape, np.uint8),
                         'cross_camera': np.zeros(owner.shape, np.uint8),
                         'revisit': np.zeros(owner.shape, np.uint8)}
            rev_owner_counts = {}
            for oo in np.unique(owner[seen_any]) if seen_any.any() else []:
                src = obs_by_order.get(int(oo))
                if src is None or int(oo) in tier1_orders:
                    continue
                same_visit = (src.site_id == o.site_id
                              and src.day == o.day
                              and src.pass_no == o.pass_no
                              and abs(o.timestep - src.timestep) <= args.window)
                if same_visit:
                    sfx = ('sequential' if src.cam == o.cam
                           else 'cross_camera')
                else:
                    sfx = 'revisit'
                    rev_owner_counts[int(oo)] = int((owner == oo).sum())
                cat_masks[sfx][owner == oo] = 1

            for sfx, mask in cat_masks.items():
                if not mask.any():
                    continue
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) < 3:
                        continue
                    cnt = cv2.approxPolyDP(cnt, 1.5, True).reshape(-1, 2)
                    if len(cnt) < 3:
                        continue
                    poly = cnt.astype(np.float64) * stride + stride // 2
                    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                    if _poly_area(poly) < args.min_area_px:
                        continue
                    polys.append((sfx, poly, None))

            if rev_owner_counts:
                n_rev = sum(rev_owner_counts.values())
                src = obs_by_order.get(
                    max(rev_owner_counts, key=rev_owner_counts.get))
                # Optional confirmation by direct registration (land-land).
                confirmed = None
                if (args.verify_revisits and src is not None
                        and not o.is_water and not src.is_water):
                    key = (src.site_id, src.rel, o.rel)
                    if key not in pair_cache:
                        H, _ = compute_homography_pair(
                            os.path.join(src.site_dir, src.rel),
                            os.path.join(site_folder, o.rel),
                            scale=args.match_scale, use_affine=True,
                            match_ratio=args.match_ratio,
                            min_inliers=args.min_inliers)
                        pair_cache[key] = H
                    confirmed = pair_cache[key] is not None
                revisit_events.append({
                    'image': o.rel, 'camera': o.cam or '',
                    'frame': o.frame,
                    'source_image': src.rel if src else '',
                    'source_site': src.site_tag if src else '',
                    'source_pass': src.pass_no if src else '',
                    'source_day': src.day if src else '',
                    'overlap_frac': n_rev / owner.size,
                    'confirmed': ('' if confirmed is None
                                  else ('yes' if confirmed else 'no')),
                })

        # ---- Fraction of image already seen (info only) ----
        if gx is not None:
            seen = seen_any.copy()
            for _sfx, poly, _src in polys:
                m = np.zeros(gx.shape, dtype=np.uint8)
                cv2.fillPoly(m, [np.round(
                    (poly - stride // 2) / stride).astype(np.int32)], 1)
                seen |= m.astype(bool)
            frac_prior[o.rel] = float(seen.mean())
        else:
            covered = 0.0
            for _sfx, poly, _src in polys:
                covered = max(covered, _poly_area(poly))
            frac_prior[o.rel] = covered / (w * h)

        for suffix, poly, _src in polys:
            rows.append((o.rel, suffix, poly))

        # ---- Stamp own footprint ----
        if o.T_enu is not None:
            quad = _apply_h(o.T_enu, rect)
            grid.stamp_polygon(quad, o.order)

    return rows, revisit_events, frac_prior


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_viame_csv(path, rows, coverage_class):
    order = []
    seen = set()
    for rel, _sfx, _poly in rows:
        if rel not in seen:
            seen.add(rel)
            order.append(rel)
    frame_ids = {rel: i + 1 for i, rel in enumerate(order)}
    with open(path, 'w') as f:
        f.write(VIAME_CSV_HEADER + '\n')
        for tid, (rel, suffix, poly) in enumerate(rows):
            x0, y0 = poly.min(axis=0)
            x1, y1 = poly.max(axis=0)
            cls = f'{coverage_class}_{suffix}'
            f.write(f'{tid},{rel},{frame_ids[rel]},'
                    f'{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f},1.0,-1,'
                    f'{cls},1.0,{_fmt_poly(poly)}\n')


def write_revisits_csv(path, events):
    cols = ['image', 'camera', 'frame', 'source_image', 'source_site',
            'source_pass', 'source_day', 'overlap_frac', 'confirmed']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            e = dict(e)
            e['overlap_frac'] = f"{e['overlap_frac']:.3f}"
            w.writerow(e)


def render_coverage_map(path, observations, site_tag):
    """ENU map of image footprints coloured by pass (line) and order (fill)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap('viridis')
    pass_colors = {1: 'black', 2: 'crimson', 3: 'darkorange'}
    n = max(1, len(observations) - 1)
    drew = 0
    for o in observations:
        if o.T_enu is None:
            continue
        quad = _apply_h(o.T_enu, _image_rect(o.width, o.height))
        ax.add_patch(MplPoly(quad, closed=True,
                             facecolor=cmap(o.order / n), alpha=0.12,
                             edgecolor=pass_colors.get(o.pass_no, 'purple'),
                             linewidth=0.4))
        drew += 1
    if not drew:
        plt.close(fig)
        return
    ax.autoscale_view()
    ax.relim()
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title(f'{site_tag}: image footprints '
                 f'(fill = acquisition order, edge colour = pass)')
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def render_thumbnail_grid(path, site_folder, observations, rows, water_info,
                          max_rows=24, thumb_w=420):
    """Thumbnail grid (rows = triggers, cols = cameras) with coverage
    polygons overlaid: sequential=orange, cross_camera=cyan, revisit=magenta."""
    import cv2
    by_image = {}
    for rel, suffix, poly in rows:
        by_image.setdefault(rel, []).append((suffix, poly))
    cams = sorted({o.cam for o in observations}, key=lambda c: CAM_ORDER[c])
    triggers = sorted({o.timestep for o in observations})
    if len(triggers) > max_rows:
        step = len(triggers) / max_rows
        triggers = [triggers[int(i * step)] for i in range(max_rows)]
    obs_map = {(o.cam, o.timestep): o for o in observations}
    colors = {'sequential': (0, 165, 255), 'cross_camera': (255, 255, 0),
              'revisit': (255, 0, 255)}
    tiles = []
    th = None
    for t in triggers:
        row_tiles = []
        for cam in cams:
            o = obs_map.get((cam, t))
            if o is None:
                row_tiles.append(None)
                continue
            img = cv2.imread(os.path.join(site_folder, o.rel))
            if img is None:
                row_tiles.append(None)
                continue
            s = thumb_w / img.shape[1]
            th = int(img.shape[0] * s)
            thumb = cv2.resize(img, (thumb_w, th))
            overlay = thumb.copy()
            for suffix, poly in by_image.get(o.rel, []):
                p = np.round(poly * s).astype(np.int32)
                cv2.fillPoly(overlay, [p], colors[suffix])
            thumb = cv2.addWeighted(overlay, 0.35, thumb, 0.65, 0)
            for suffix, poly in by_image.get(o.rel, []):
                p = np.round(poly * s).astype(np.int32)
                cv2.polylines(thumb, [p], True, colors[suffix], 2)
            label = f'{cam or "CAM"} #{o.frame}'
            if o.is_water:
                label += ' [water]'
            cv2.putText(thumb, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(thumb, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            row_tiles.append(thumb)
        tiles.append(row_tiles)
    if th is None:
        return
    blank = np.full((th, thumb_w, 3), 32, dtype=np.uint8)
    grid_img = np.vstack([
        np.hstack([t if t is not None else blank for t in row])
        for row in tiles])
    cv2.imwrite(path, grid_img)


# ---------------------------------------------------------------------------
# Site pipeline
# ---------------------------------------------------------------------------

def process_site(site_folder, site_id, grid, order_start, args, to_enu,
                 origin_ref, obs_registry):
    """Process one site folder against the shared coverage grid. Returns the
    observations (with coverage rows already written)."""
    t0 = time.time()
    site_tag = os.path.basename(os.path.normpath(site_folder))
    print(f'\n=== {site_tag} ===')

    records, cams = smd.build_image_records(
        site_folder, flight_logs=args.flight_logs, read_exif=True)
    frames_by_cam = {}
    idx_by_cam = {}
    for cam, rels in cams.items():
        fr = [smd.parse_image_filename(r)['frame'] for r in rels]
        fr = [f if f is not None else i for i, f in enumerate(fr)]
        frames_by_cam[cam] = fr
        idx_by_cam[cam] = {f: i for i, f in enumerate(fr)}

    all_rels = [r for cam in cams for r in cams[cam]]
    day = smd.folder_date(site_folder) or ''

    # ENU origin: first GPS fix of the whole run defines the shared frame.
    first_fix = next((records[r] for r in all_rels
                      if records.get(r, {}).get('lat') is not None), None)
    if first_fix is not None:
        if origin_ref.get('lat') is None:
            origin_ref['lat'], origin_ref['lon'] = (first_fix['lat'],
                                                    first_fix['lon'])
            to_enu = origin_ref['to_enu'] = smd.make_enu(
                origin_ref['lat'], origin_ref['lon'])
        else:
            to_enu = origin_ref['to_enu']

    # ---- Registration (hybrid mode) ----
    chains, xcam, cal = {}, {}, {}
    water_info = {}
    if args.method == 'hybrid':
        try:
            import reconstruct_3d as _r3d
            _r3d.import_dependencies()
            water_info = _r3d.classify_images_fast(site_folder, all_rels)
        except Exception as e:
            print(f'    Water classifier unavailable ({e}); SIFT heuristic')
            water_info = {}
        reg_kwargs = dict(
            water_info=water_info, match_ratio=args.match_ratio,
            min_inliers=args.min_inliers, scale=args.match_scale,
            use_affine=True, consistency_filter=True)
        print('  Computing within-camera registration chains...')
        pairwise = {}
        for cam, rels in cams.items():
            ch, pw = _compute_camera_chain(site_folder, rels,
                                           label=str(cam), **reg_kwargs)
            chains[cam], pairwise[cam] = ch, pw
            print(f'    {cam}: {len(ch)}/{len(rels)} frames chained')
        if len(cams) > 1:
            print('  Computing rig cross-camera transforms...')
            xcam = _xcam_consensus(
                site_folder, cams, chains, frames_by_cam, water_info,
                dict(scale=args.match_scale, use_affine=True,
                     match_ratio=args.match_ratio,
                     min_inliers=args.min_inliers),
                trials=args.cross_cam_trials, tol=args.xcam_cluster_tol)
        poses_by_cam = {cam: {r: records[r] for r in rels
                              if records.get(r, {}).get('lat') is not None}
                        for cam, rels in cams.items()}
        have_gps = any(poses_by_cam[cam] for cam in cams)
        if have_gps:
            print('  Geo-anchoring chains (GPS dead-reckoning fill)...')
            cal = _geo_anchor_with_cal(chains, cams, poses_by_cam, pairwise)
        else:
            print('  No GPS metadata: moving-average fill for water frames')
            for cam, rels in cams.items():
                n_fill = _fill_unchained_no_gps(chains[cam], len(rels))
                if n_fill:
                    print(f'    {cam}: filled {n_fill} frames via moving '
                          f'average of chained motion')

    # ---- Build observations with pixel->ENU transforms ----
    # Per-camera precomputation: local ENU (relative to the camera's first
    # GPS fix), GPS-track headings, and the offset of that local frame in
    # the run-wide shared ENU frame.
    cam_geo = {}
    for cam, rels in cams.items():
        poses = {r: records[r] for r in rels
                 if records.get(r, {}).get('lat') is not None}
        if not poses or to_enu is None:
            continue
        enu_local, _yaw_logged = _poses_to_enu(poses, rels)
        heads = _track_headings(enu_local)
        ref_rec = next(records[r] for r in rels
                       if records.get(r, {}).get('lat') is not None)
        off = to_enu(ref_rec['lat'], ref_rec['lon'])
        cam_geo[cam] = {'enu': enu_local, 'heads': heads, 'off': off}

    observations = []
    order = order_start
    triggers = sorted({f for fr in frames_by_cam.values() for f in fr})
    center_T = {}   # trigger -> CENTER pixel->ENU transform
    for t in triggers:
        for cam in sorted(cams, key=lambda c: CAM_ORDER[c]):
            i = idx_by_cam[cam].get(t)
            if i is None:
                continue
            rel = cams[cam][i]
            rec = records.get(rel, {})
            w = rec.get('width') or 5168
            h = rec.get('height') or 3448
            geo = cam_geo.get(cam)
            T = None
            if args.method == 'hybrid' and cam in ('CENTER', None) \
                    and cal.get(cam, {}).get('M') is not None \
                    and geo is not None:
                # The GPS fix is (near enough) the CENTER nadir point, so the
                # calibrated aircraft-centred transform applies to CENTER only.
                c = cal[cam]
                enu_xy = c['enu'][i] if i < len(c['enu']) else None
                yaw = c['yaw'][i] if i < len(c['yaw']) else float('nan')
                T = _pixel_to_enu_transform(enu_xy, yaw, c['M'], w, h,
                                            geo['off'])
                if T is not None:
                    center_T[t] = T
            elif args.method == 'hybrid' and cam in xcam \
                    and t in center_T:
                # PORT/STAR image centres sit ~100 m across-track from the
                # aircraft; the measured rig transform places them exactly:
                # cam px -> CENTER px (same trigger) -> ENU.
                T = center_T[t] @ xcam[cam]
            if T is None and geo is not None \
                    and not np.isnan(geo['enu'][i, 0]):
                # Metadata-only footprint (also the hybrid fallback for
                # frames/cameras the calibration could not cover).
                heading = geo['heads'][i]
                if np.isnan(heading) and rec.get('yaw') is not None:
                    heading = rec['yaw']
                lat_frac = {'PORT': -args.xcam_offset_frac, 'CENTER': 0.0,
                            'STAR': args.xcam_offset_frac}.get(cam, 0.0)
                T = _metadata_transform(
                    rec, 0.0 if np.isnan(heading) else heading, w, h,
                    to_enu, lat_frac)
            observations.append(Observation(
                order=order, site_id=site_id, site_tag=site_tag,
                site_dir=site_folder, cam=cam,
                frame=rec.get('frame') if rec.get('frame') is not None else t,
                rel=rel, width=w, height=h, T_enu=T,
                chain_H=None, timestep=t,
                is_water=water_info.get(rel, {}).get('is_water', False),
                pass_no=rec.get('pass') or 1, day=day,
                has_gps=rec.get('lat') is not None))
            order += 1

    n_geo = sum(1 for o in observations if o.T_enu is not None)
    print(f'  {len(observations)} images, {n_geo} geo-referenced')
    for o in observations:
        obs_registry[o.order] = o

    # ---- Coverage ----
    print('  Computing prior coverage...')
    rows, revisit_events, frac = compute_coverage(
        observations, grid, args, chains, xcam, frames_by_cam, idx_by_cam,
        site_folder, water_info, obs_registry)

    # ---- Outputs ----
    out_dir = args.output or (os.path.normpath(site_folder) + '_coverage')
    if args.output and len(args.sites) > 1:
        out_dir = os.path.join(args.output, site_tag)
    os.makedirs(out_dir, exist_ok=True)
    write_viame_csv(os.path.join(out_dir, 'prior_coverage.csv'), rows,
                    args.coverage_class)
    write_revisits_csv(os.path.join(out_dir, 'revisits.csv'), revisit_events)
    render_coverage_map(os.path.join(out_dir, 'coverage_map.png'),
                        observations, site_tag)
    if not args.no_thumbnails:
        render_thumbnail_grid(
            os.path.join(out_dir, 'prior_coverage_vis.png'),
            site_folder, observations, rows, water_info)

    by_cam = {}
    for o in observations:
        by_cam.setdefault(o.cam, []).append(frac.get(o.rel, 0.0))
    for cam in sorted(by_cam, key=lambda c: CAM_ORDER[c]):
        v = by_cam[cam]
        n_any = sum(1 for x in v if x > 0.01)
        print(f'    {cam or "MONO"}: {n_any}/{len(v)} frames with prior '
              f'coverage, '
              f'mean seen fraction {np.mean(v):.2f}')
    n_rev = sum(1 for e in revisit_events if e['overlap_frac'] > 0.02)
    print(f'    revisit overlaps: {n_rev} frames'
          f' ({sum(1 for e in revisit_events if e["confirmed"] == "yes")}'
          f' registration-confirmed)')
    print(f'  Outputs -> {out_dir}  ({time.time() - t0:.0f}s)')
    return observations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Detect previously-observed regions in survey imagery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('sites', nargs='*', help='Site folder(s), oldest first')
    ap.add_argument('--all', metavar='ROOT',
                    help='Process every site folder under ROOT (sorted by '
                         'date then name)')
    ap.add_argument('--flight-logs', default=None,
                    help='Flight-log CSV or directory of daily FMCLOG CSVs')
    ap.add_argument('--output', default=None,
                    help='Output directory (default <site>_coverage)')
    ap.add_argument('--method', choices=['hybrid', 'metadata', 'sfm-rig'],
                    default='hybrid')
    ap.add_argument('--coverage-class', default='prior_coverage',
                    help='Class-name prefix for CSV rows')
    ap.add_argument('--window', type=int, default=8,
                    help='Trigger window treated as "recent" overlap; beyond '
                         'this a same-ground observation counts as a revisit')
    ap.add_argument('--grid-cell', type=float, default=1.0,
                    help='Coverage grid cell size (metres)')
    ap.add_argument('--query-stride', type=int, default=32,
                    help='Pixel stride when sampling the grid per image')
    ap.add_argument('--min-area-px', type=float, default=40000,
                    help='Ignore coverage polygons smaller than this (px^2)')
    ap.add_argument('--xcam-offset-frac', type=float, default=0.9,
                    help='Metadata-only lateral footprint offset of PORT/'
                         'STAR as a fraction of footprint width')
    ap.add_argument('--verify-revisits', action='store_true', default=True)
    ap.add_argument('--no-verify-revisits', dest='verify_revisits',
                    action='store_false')
    ap.add_argument('--no-thumbnails', action='store_true')
    # Registration options (defaults follow the validated experiment config).
    ap.add_argument('--match-ratio', type=float, default=0.80)
    ap.add_argument('--match-scale', type=float, default=0.5)
    ap.add_argument('--min-inliers', type=int, default=10)
    ap.add_argument('--cross-cam-trials', type=int, default=15)
    ap.add_argument('--xcam-cluster-tol', type=float, default=300.0)
    args = ap.parse_args()

    sites = list(args.sites)
    if args.all:
        for d in sorted(os.listdir(args.all)):
            p = os.path.join(args.all, d)
            if not os.path.isdir(p):
                continue
            if any(v for v in smd.list_site_images(p).values()):
                sites.append(p)
    if not sites:
        ap.error('no site folders given (positional or --all)')
    args.sites = sites

    if args.method == 'sfm-rig':
        print('sfm-rig method: delegating to prior_coverage_sfm')
        import prior_coverage_sfm
        return prior_coverage_sfm.run(args)

    grid = CoverageGrid(cell_m=args.grid_cell)
    origin_ref = {'lat': None, 'lon': None, 'to_enu': None}
    obs_registry = {}
    order = 0
    for site_id, site in enumerate(sites):
        obs = process_site(site, site_id, grid, order, args,
                           origin_ref.get('to_enu'), origin_ref,
                           obs_registry)
        order += len(obs)
    print(f'\nDone: {order} images, '
          f'{len(grid.tiles)} grid tiles ({args.grid_cell} m cells)')


if __name__ == '__main__':
    main()
