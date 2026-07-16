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
  prior_coverage.csv       VIAME CSV; one polygon row per PRIOR FRAME whose
                           ground this image re-observes, class as above, with
                           the source frame in a trailing "(note) src=CAM#NNNN"
  revisits.csv             summary of detected revisit events
  coverage_map.png         ENU map of footprints coloured by pass/order
  prior_coverage_vis.png   thumbnail grid (rows = a contiguous run of triggers,
                           columns = STAR|CENTER|PORT) with each prior frame's
                           region outlined and labelled separately
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# The registration core, metadata reader and low-level registration machinery
# all live in the viame.opencv plugin, so this tool and the in-pipeline
# registration node (viame.colmap.colmap_registration) share one
# implementation. This tool keeps only the coverage grid, CSV/visualization
# outputs and the CLI.
from viame.core import survey_metadata as smd
from viame.opencv import registration_utils as _sr
_sr.import_dependencies()
from viame.opencv.registration_utils import compute_homography_pair
from viame.opencv.prior_coverage_opencv import (
    CAM_ORDER, Observation, SiteRegistration, _register_site,
    compute_frame_homographies, _metadata_transform,
    _apply_h, _image_rect, _clip_poly_to_rect, _sane_relative, _poly_area,
    _fmt_poly,
)


# Column order for the thumbnail-grid visualization: physical rig layout
# as seen from behind the aircraft (STAR | CENTER | PORT).
VIS_ORDER = {'STAR': 0, 'CENTER': 1, 'PORT': 2, None: 1}

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


def parse_frame_range(spec):
    """'240-270' -> (240, 270); None -> None."""
    if not spec:
        return None
    a, b = spec.split('-')
    return int(a), int(b)


# ---------------------------------------------------------------------------
# Coverage engine
# ---------------------------------------------------------------------------

def compute_coverage(observations, grid, args, chains, xcam, frames_by_cam,
                     idx_by_cam, site_folder, water_info, obs_registry):
    """Walk observations in acquisition order; for each, report previously
    observed regions (precise tier for recent/same-trigger overlap, ENU grid
    tier for revisits), then stamp its own footprint into the grid.

    Returns (rows, revisit_events, frac_prior) where rows are
    (rel, class, polygon, source_order) tuples in order; source_order is the
    global order index of the earlier image that observed the region.
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
            rev_owner_counts = {}
            # One contour set PER SOURCE FRAME (not per class): the region a
            # given earlier image contributed keeps its own boundary, so
            # overlapping prior frames stay distinguishable downstream instead
            # of merging into a single blob per class.
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
                mask = (owner == oo).astype(np.uint8)
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
                    polys.append((sfx, poly, int(oo)))

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

        for suffix, poly, src in polys:
            rows.append((o.rel, suffix, poly, src))

        # ---- Stamp own footprint ----
        if o.T_enu is not None:
            quad = _apply_h(o.T_enu, rect)
            grid.stamp_polygon(quad, o.order)

    return rows, revisit_events, frac_prior


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _source_tag(obs):
    """Compact, space-free identifier of a source image for CSV notes."""
    if obs is None:
        return None
    return f'{obs.cam or "CAM"}#{obs.frame}'


def write_viame_csv(path, rows, coverage_class, obs_registry=None):
    order = []
    seen = set()
    for rel, _sfx, _poly, _src in rows:
        if rel not in seen:
            seen.add(rel)
            order.append(rel)
    frame_ids = {rel: i + 1 for i, rel in enumerate(order)}
    with open(path, 'w') as f:
        f.write(VIAME_CSV_HEADER + '\n')
        for tid, (rel, suffix, poly, src) in enumerate(rows):
            x0, y0 = poly.min(axis=0)
            x1, y1 = poly.max(axis=0)
            cls = f'{coverage_class}_{suffix}'
            # Each row is ONE source frame's region, so name it: consumers (and
            # the thumbnail grid) can tell overlapping prior frames apart.
            tag = _source_tag((obs_registry or {}).get(src))
            note = f',(note) src={tag}' if tag else ''
            f.write(f'{tid},{rel},{frame_ids[rel]},'
                    f'{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f},1.0,-1,'
                    f'{cls},1.0,{_fmt_poly(poly)}{note}\n')


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


CLASS_COLOR = {'sequential': (0, 165, 255),     # orange
               'cross_camera': (255, 255, 0),   # cyan
               'revisit': (255, 0, 255)}        # magenta
CLASS_TAG = {'sequential': 'S', 'cross_camera': 'X', 'revisit': 'R'}
CLASS_DRAW_ORDER = ['sequential', 'cross_camera', 'revisit']


def _shade(color, i, n):
    """Per-polygon stroke shade: oldest prior frame dark, newest bright."""
    f = 1.0 if n <= 1 else 0.45 + 0.55 * (i / (n - 1.0))
    return tuple(int(round(c * f)) for c in color)


def _put_text(img, text, org, scale=0.55, color=(255, 255, 255)):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


def _vis_window(triggers, obs_map, by_image, max_rows):
    """Pick a CONTIGUOUS run of triggers (never subsample): the window with the
    most revisit regions, so the loop closures are the part shown in full."""
    if max_rows <= 0 or len(triggers) <= max_rows:
        return triggers
    cams = {o.cam for o in obs_map.values()}
    score = []
    for t in triggers:
        s = 0
        for cam in cams:
            o = obs_map.get((cam, t))
            if o is not None:
                s += sum(1 for sfx, _p, _s in by_image.get(o.rel, [])
                         if sfx == 'revisit')
        score.append(s)
    best_i = max(range(len(triggers) - max_rows + 1),
                 key=lambda i: sum(score[i:i + max_rows]))
    return triggers[best_i:best_i + max_rows]


def render_thumbnail_grid(path, site_folder, observations, rows, water_info,
                          obs_registry=None, max_rows=40, thumb_w=420,
                          frames=None, title=None):
    """Thumbnail grid (rows = triggers, cols = STAR|CENTER|PORT) with the
    previously-observed regions overlaid: sequential=orange, cross_camera=cyan,
    revisit=magenta.

    Every region keeps its OWN boundary - each row of `rows` is one prior frame
    warped into this image, so it is stroked separately, tagged S1..Sn/X1../R1..
    and shaded dark (older) to bright (newer). Regions are only lightly filled,
    so overlapping prior frames stay individually visible rather than merging
    into one flat blob of colour.

    Rows are a contiguous run of triggers (`frames` = explicit (first, last), or
    the `max_rows` window with the most revisits; max_rows <= 0 = whole site).
    Each tile also shows the water/land classifier verdict (the class label,
    tinted cyan for water and green for land).
    """
    import cv2
    by_image = {}
    for rel, suffix, poly, src in rows:
        by_image.setdefault(rel, []).append((suffix, poly, src))
    src_obs = dict(obs_registry or {})
    # Classifier method used (uniform across the site) for the header banner.
    water_method = next((v.get('method') for v in water_info.values()
                         if v and v.get('method')), None)
    cams = sorted({o.cam for o in observations}, key=lambda c: VIS_ORDER[c])
    obs_map = {(o.cam, o.timestep): o for o in observations}
    triggers = sorted({o.timestep for o in observations})
    if frames:
        triggers = [t for t in triggers if frames[0] <= t <= frames[1]]
    else:
        triggers = _vis_window(triggers, obs_map, by_image, max_rows)
    if not triggers:
        return

    tiles = []
    th = None
    for t in triggers:
        row_tiles = []
        for cam in cams:
            o = obs_map.get((cam, t))
            img = (cv2.imread(os.path.join(site_folder, o.rel))
                   if o is not None else None)
            if img is None:
                row_tiles.append(None)
                continue
            s = thumb_w / img.shape[1]
            th = int(img.shape[0] * s)
            thumb = cv2.resize(img, (thumb_w, th))
            grouped = {c: [(p, sr) for sfx, p, sr in by_image.get(o.rel, [])
                           if sfx == c] for c in CLASS_DRAW_ORDER}

            # Light fill, one polygon at a time: overlaps accumulate (and get
            # darker) instead of flattening into a single uniform region.
            for cls in CLASS_DRAW_ORDER:
                for i, (poly, _sr) in enumerate(grouped[cls]):
                    p = np.round(poly * s).astype(np.int32)
                    lay = thumb.copy()
                    cv2.fillPoly(lay, [p], _shade(CLASS_COLOR[cls], i,
                                                  len(grouped[cls])))
                    thumb = cv2.addWeighted(lay, 0.10, thumb, 0.90, 0)
            # Boundaries on top: one stroke + tag per prior frame.
            for cls in CLASS_DRAW_ORDER:
                n = len(grouped[cls])
                for i, (poly, sr) in enumerate(grouped[cls]):
                    p = np.round(poly * s).astype(np.int32)
                    col = _shade(CLASS_COLOR[cls], i, n)
                    cv2.polylines(thumb, [p], True, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.polylines(thumb, [p], True, col, 2, cv2.LINE_AA)
                    tag = f'{CLASS_TAG[cls]}{i + 1}'
                    stag = _source_tag(src_obs.get(sr))
                    if stag:
                        tag += f' {stag}'
                    # Anchor at the region centroid: with many overlapping
                    # regions, corner-anchored labels pile up on each other.
                    c = p.mean(axis=0)
                    org = (int(np.clip(c[0] - 55, 2, thumb_w - 120)),
                           int(np.clip(c[1], 80, th - 6)))
                    _put_text(thumb, tag, org, 0.45, col)

            _put_text(thumb, f'{cam or "CAM"} #{o.frame}', (8, 24), 0.62)
            counts = ' '.join(f'{CLASS_TAG[c]}x{len(grouped[c])}'
                              for c in CLASS_DRAW_ORDER if grouped[c])
            if counts:
                _put_text(thumb, counts, (8, 46), 0.5, (200, 200, 200))
            # Water/land classifier verdict: class label tinted cyan for
            # water, green for land (BGR).
            cinfo = water_info.get(o.rel, {})
            clabel = cinfo.get('label')
            if clabel:
                ccol = (255, 255, 0) if cinfo.get('is_water') else (0, 220, 0)
                _put_text(thumb, clabel, (8, 68), 0.5, ccol)
            cv2.rectangle(thumb, (0, 0), (thumb_w - 1, th - 1), (60, 60, 60), 1)
            row_tiles.append(thumb)
        tiles.append(row_tiles)
    if th is None:
        return
    blank = np.full((th, thumb_w, 3), 32, dtype=np.uint8)
    grid_img = np.vstack([
        np.hstack([t if t is not None else blank for t in row])
        for row in tiles])

    # Header banner: column order, frame range, and the overlay legend.
    banner = np.full((76, grid_img.shape[1], 3), 24, dtype=np.uint8)
    wm = {'svm': 'SVM background classifier',
          'sift': 'SIFT keypoint heuristic'}.get(water_method,
                                                 water_method or 'n/a')
    site_tag = title or os.path.basename(os.path.normpath(site_folder))
    _put_text(banner, f'{site_tag}   columns: {" | ".join(str(c) for c in cams)}'
                      f'   frames {triggers[0]}-{triggers[-1]} '
                      f'({len(triggers)} consecutive triggers, none skipped)',
              (10, 26), 0.6)
    _put_text(banner, 'previously-observed regions, one outline per prior frame'
                      ' (shade: dark = older, bright = newer)   |   water class'
                      f' (cyan=water green=land) via {wm}',
              (10, 50), 0.5, (200, 200, 200))
    x = 10
    for cls, txt in (('sequential', 'S = sequential (same camera)'),
                     ('cross_camera', 'X = cross-camera (rig)'),
                     ('revisit', 'R = revisit / loop closure')):
        cv2.rectangle(banner, (x, 60), (x + 18, 70), CLASS_COLOR[cls], -1)
        _put_text(banner, txt, (x + 24, 70), 0.5, CLASS_COLOR[cls])
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x += 24 + tw + 40
    grid_img = np.vstack([banner, grid_img])
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

    reg = _register_site(site_folder, site_id, order_start, args, to_enu,
                         origin_ref)
    observations = reg.observations
    records, cams = reg.records, reg.cams
    frames_by_cam, idx_by_cam = reg.frames_by_cam, reg.idx_by_cam
    chains, xcam, water_info = reg.chains, reg.xcam, reg.water_info

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
    if not args.revisits_only:
        write_viame_csv(os.path.join(out_dir, 'prior_coverage.csv'), rows,
                        args.coverage_class, obs_registry)
    write_revisits_csv(os.path.join(out_dir, 'revisits.csv'), revisit_events)
    render_coverage_map(os.path.join(out_dir, 'coverage_map.png'),
                        observations, site_tag)
    if not args.no_thumbnails and not args.revisits_only:
        render_thumbnail_grid(
            os.path.join(out_dir, 'prior_coverage_vis.png'),
            site_folder, observations, rows, water_info,
            obs_registry=obs_registry, max_rows=args.vis_rows,
            thumb_w=args.vis_thumb_width,
            frames=parse_frame_range(args.vis_frames))

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
    ap.add_argument('--water-method', choices=['auto', 'svm', 'sift'],
                    default='auto',
                    help="Water/land classifier (hybrid method only). 'svm' "
                         "= VIAME sea-lion background classifier (more "
                         "accurate; errors out if its models are missing); "
                         "'sift' = keypoint-count heuristic (no models, but "
                         "textured water reads as land); 'auto' = SVM when "
                         "available else SIFT (default)")
    ap.add_argument('--sfm-matching', choices=['auto', 'sequential',
                                               'exhaustive'], default='auto',
                    help="Feature-matching strategy for --method sfm-rig. "
                         "'auto' = sequential + GPS-spatial pairing (falls "
                         "back to exhaustive only without GPS on small sites); "
                         "'exhaustive' = all-pairs (finds loop closures with "
                         "no GPS, but O(n^2) and riskier over water); "
                         "'sequential' = neighbours only")
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
    ap.add_argument('--vis-rows', type=int, default=40,
                    help='Triggers in the thumbnail grid. Rows are always a '
                         'CONTIGUOUS run (the window with the most revisits); '
                         'no frames are skipped. 0 = the whole site')
    ap.add_argument('--vis-frames', default=None,
                    help='Explicit contiguous trigger range for the thumbnail '
                         'grid, e.g. 240-270 (overrides --vis-rows)')
    ap.add_argument('--vis-thumb-width', type=int, default=420,
                    help='Thumbnail width (px) in the grid')
    ap.add_argument('--revisits-only', action='store_true',
                    help='Only detect/report revisit events (revisits.csv '
                         '+ coverage map); skip per-frame coverage CSV and '
                         'thumbnails. Supersedes detect_site_revisits.py.')
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
        print('sfm-rig method: delegating to viame.colmap.prior_coverage_sfm')
        # The SFM plugin imports this tool by name for the coverage grid and
        # output writers, so make sure the tool's directory is importable.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from viame.colmap import prior_coverage_sfm
        except ImportError:
            # Source-tree layout (plugin not installed as a package).
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'plugins', 'colmap'))
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
