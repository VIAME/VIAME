#!/usr/bin/env python3
"""prior_coverage_opencv.py - shared registration core for previously-observed
region detection.

Registers a (multi-camera) aerial survey and produces, for every image, a
pixel->reference transform placing it in a common frame (local ENU metres when
GPS/flight-log metadata is available, else pseudo-ENU from the registration
chains). Both the detect_prior_coverage.py tool and the in-pipeline VIAME
registration node (viame.opencv.colmap_registration) build on this, so the
node needs nothing from the tool.

The heavy lifting (affine chains, adaptive matching, GPS dead-reckoning fill,
water/land classification) lives in viame.opencv.registration_utils; this
module adds the rig cross-camera consensus, the metadata/GPS geo-anchoring, and
the per-frame pixel->ENU assembly.
"""

import argparse
import os

import numpy as np

from viame.core import survey_metadata as smd
from viame.opencv import registration_utils as _sr
# registration_utils keeps numpy/cv2 as lazily-populated module globals; make
# sure they are bound before any of its functions run (the tool does the same).
_sr.import_dependencies()
from viame.opencv.registration_utils import (
    compute_homography_pair, _compute_camera_chain,
    _poses_to_enu, _track_headings, _rot2, _geo_calibrate,
    reconcile_enu_with_chain,
)

# Physical rig layout as seen from behind the aircraft; acquisition order used
# when walking triggers so CENTER anchors the cross-camera composition.
CAM_ORDER = {'CENTER': 0, 'PORT': 1, 'STAR': 2, None: 0}


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
                         verbose=True, reconcile=True):
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

    # Rig-shared CHIRALITY: the cameras are one rigid body so the ground->pixel
    # handedness is identical; a per-camera fit over ambiguous terrain can lock
    # onto the mirrored solution. Vote the consensus handedness (quality-weighted
    # by steps/residual) and refit any dissenter with it (safety net; matches
    # registration_utils._geo_anchor_cameras).
    def _chir(M):
        return 1 if np.linalg.det(M) >= 0 else -1
    votes = {}
    for c in cal.values():
        if c['M'] is None or c['n'] < 3:
            continue
        votes[_chir(c['M'])] = votes.get(_chir(c['M']), 0.0) + c['n'] / (1.0 + (c['res'] or 1e3))
    if len(votes) > 1:
        consensus = max(votes, key=votes.get)
        for cam, c in cal.items():
            if c['M'] is None or _chir(c['M']) == consensus:
                continue
            M2, n2, r2, enu2, yaw2 = _geo_calibrate(
                cam_chains.get(cam, {}), cams[cam], poses_by_cam[cam],
                pairwise_by_cam.get(cam), force_chir=consensus)
            if M2 is not None:
                c.update({'M': M2, 'n': n2, 'res': r2, 'enu': enu2, 'yaw': yaw2})
                if verbose:
                    print(f"    {cam}: chirality was mirrored vs the rig; "
                          f"refit to consensus handedness")
    for cam, c in cal.items():
        target = None       # rig-consensus scale first, metadata GSD second
        reliable = (c['M'] is not None and c['n'] >= 8
                    and c['res'] is not None and c['res'] < 150)
        if c['M'] is None:
            ref = shared or c['expect']
            if ref is not None and not np.all(np.isnan(c['yaw'])):
                # No usable pairwise steps at all (e.g. all-water camera):
                # synthesize M from the known mounting. CENTER/PORT image-up =
                # flight direction; STAR is mounted ~180 deg rotated (measured
                # on the 2024 rig: sequential pixel dy is +1300 px for
                # CENTER/PORT vs -1380 px for STAR), so its M is negated.
                sgn = -1.0 if str(cam).upper() == 'STAR' else 1.0
                c['M'] = sgn * np.array([[ref, 0.0], [0.0, -ref]])
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
        # Correct per-frame GPS positions that disagree with the image chain
        # (sub-second trigger / GPS-sample aliasing) before dead-reckoning the
        # unregistered frames off them. Falls back to raw GPS where the chain is
        # unusable, so it is safe on every camera.
        if reconcile:
            c['enu'] = reconcile_enu_with_chain(
                cam_chains.get(cam, {}), c['enu'], c['yaw'], c['M'],
                label=(note if verbose else ''))
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
    # body->ENU = _rot2(-yaw), the inverse of the ENU->body _rot2(+yaw) used
    # when fitting M in registration_utils._geo_calibrate/_geo_fill. The prior
    # _rot2(+yaw) was self-consistent only for single-heading / out-and-back
    # fitted M; with a SYNTHESIZED M (camera with <3 usable pairwise steps, e.g.
    # VALDEZ ARM) it mis-rotated every footprint by -2*heading -- the bogus
    # 'crabbing' coverage map.
    A = _rot2(-yaw) @ Minv
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


class SiteRegistration:
    """Everything produced by registering one site: the per-image
    Observations (each carrying its pixel->reference transform in T_enu) plus
    the registration state the coverage engine needs downstream."""

    __slots__ = ('observations', 'records', 'cams', 'frames_by_cam',
                 'idx_by_cam', 'chains', 'xcam', 'cal', 'water_info', 'to_enu')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _register_site(site_folder, site_id, order_start, args, to_enu,
                   origin_ref, images=None):
    """Register one site: build image records, run the hybrid registration
    (within-camera affine chains + rig cross-camera consensus + GPS geo-
    anchoring, or pseudo-georeferencing without GPS), and turn every image
    into an Observation whose T_enu maps its pixels into the shared reference
    frame (local ENU metres, or pseudo-ENU when there is no GPS). Returns a
    SiteRegistration. This is the shared core used both by the coverage tool
    (process_site) and by the in-pipeline registration node.

    `images` (optional) restricts registration to an explicit subset of the
    folder's images instead of scanning the whole folder; the chains still need
    a reasonably contiguous per-camera run to register well."""
    site_tag = os.path.basename(os.path.normpath(site_folder))

    records, cams = smd.build_image_records(
        site_folder, flight_logs=args.flight_logs, read_exif=True,
        image_list=images)
    # A frozen/duplicated GPS track (valid-looking fix flags, stale position) is
    # worse than no GPS: dead-reckoning stacks every frame on one spot, so all
    # coverage reads as "already seen" and revisits become meaningless. Drop the
    # metadata and let the registration chains pseudo-georeference the site.
    _bad_gps, _why = smd.gps_degenerate(records, cams)
    if _bad_gps:
        print(f'  ** Ignoring flight-log GPS: {_why}')
        print('     falling back to registration-only geo-referencing')
        for _rel, _rec in records.items():
            if _rec:
                _rec['lat'] = _rec['lon'] = None
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
        # A 'svm' request that cannot be honored raises; let it propagate so
        # the run fails loudly rather than silently degrading. 'auto'/'sift'
        # never raise for availability.
        print(f'  Classifying water/land ({args.water_method})...')
        water_info = _sr.classify_images_fast(
            site_folder, all_rels, method=args.water_method)
        used = next((v.get('method') for v in water_info.values() if v), None)
        n_water = sum(1 for v in water_info.values() if v.get('is_water'))
        print(f'    {used or args.water_method} classifier: {n_water}/'
              f'{len(all_rels)} water frames')
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
            cal = _geo_anchor_with_cal(
                chains, cams, poses_by_cam, pairwise,
                reconcile=getattr(args, 'gps_chain_reconcile', True))
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

    # No metadata anywhere: fall back to PSEUDO-georeferencing so within-
    # site coverage (incl. grid-based loop closures) still works. The
    # CENTER registration chain defines the site frame; PORT/STAR compose
    # through the measured rig transform; a nominal GSD converts pixels to
    # "metres" and a large per-site offset keeps unrelated sites from
    # colliding in the shared grid (cross-site revisits need real GPS).
    NOMINAL_PX_PER_M = 48.0
    pseudo = (args.method == 'hybrid' and to_enu is None and chains)
    if pseudo:
        print('  No GPS metadata: pseudo-georeferencing from registration '
              'chains (within-site coverage only)')
        S_pseudo = np.diag([1.0 / NOMINAL_PX_PER_M,
                            1.0 / NOMINAL_PX_PER_M, 1.0])
        S_pseudo[0, 2] = S_pseudo[1, 2] = site_id * 1e5

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
            if T is None and pseudo:
                ref_cam = 'CENTER' if 'CENTER' in cams else \
                    sorted(cams, key=lambda c: CAM_ORDER[c])[0]
                ref_chain = chains.get(ref_cam, {})
                ci = idx_by_cam.get(ref_cam, {}).get(t)
                if cam == ref_cam and i in ref_chain:
                    T = S_pseudo @ ref_chain[i]
                elif cam in xcam and ci is not None and ci in ref_chain:
                    T = S_pseudo @ ref_chain[ci] @ xcam[cam]
            if T is None and geo is not None \
                    and not np.isnan(geo['enu'][i, 0]):
                # Metadata-only footprint (also the hybrid fallback for
                # frames/cameras the calibration could not cover).
                heading = geo['heads'][i]
                if np.isnan(heading) and rec.get('yaw') is not None:
                    heading = rec['yaw']
                # Sign verified against imagery (2024 survey): the PORT-folder
                # camera LOOKS starboard (+across) and STAR looks port -- the
                # rig is cross-aimed. See survey_metadata.CAM_LATERAL_FRAC.
                lat_frac = {'PORT': args.xcam_offset_frac, 'CENTER': 0.0,
                            'STAR': -args.xcam_offset_frac}.get(cam, 0.0)
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

    return SiteRegistration(
        observations=observations, records=records, cams=cams,
        frames_by_cam=frames_by_cam, idx_by_cam=idx_by_cam, chains=chains,
        xcam=xcam, cal=cal, water_info=water_info, to_enu=to_enu)


def compute_frame_homographies(site_folder, flight_logs=None, method='hybrid',
                               water_method='auto', reg_overrides=None,
                               images=None):
    """Register one site and return per-image pixel->reference homographies,
    for callers (e.g. the VIAME registration node) that want the geometry
    without the coverage CSV/visualization.

    `images` (optional) restricts registration to an explicit subset of the
    folder's images (paths absolute or relative to site_folder) rather than
    scanning the whole folder.

    Returns a dict rel -> info, where info has:
      'H'         3x3 float64 pixel->reference transform, or None if the frame
                  could not be placed. The reference is local ENU metres when
                  GPS/flight-log metadata is available (so transforms from
                  different rig cameras at one trigger share a frame and their
                  relative mapping is exact), else pseudo-ENU from the
                  registration chains (still cross-camera consistent within the
                  site). Units cancel for cross-camera work either way.
      'cam'       camera tag ('STAR'/'CENTER'/'PORT' or None)
      'frame'     per-day trigger/frame counter
      'timestep'  shared trigger index across cameras
      'width', 'height', 'is_water'

    method='metadata' skips image registration (footprints straight from GPS +
    altitude; fast). Without any metadata, method='hybrid' still returns
    pseudo-ENU transforms from the registration chains.
    """
    args = argparse.Namespace(
        method=method, water_method=water_method, flight_logs=flight_logs,
        match_ratio=0.80, match_scale=0.5, min_inliers=10,
        cross_cam_trials=15, xcam_cluster_tol=300.0, xcam_offset_frac=0.9,
        gps_chain_reconcile=True)
    for k, v in (reg_overrides or {}).items():
        setattr(args, k, v)
    reg = _register_site(site_folder, 0, 0, args, None,
                         {'lat': None, 'lon': None, 'to_enu': None},
                         images=images)
    out = {}
    for o in reg.observations:
        out[o.rel] = {
            'H': None if o.T_enu is None else np.asarray(o.T_enu, np.float64),
            'cam': o.cam, 'frame': o.frame, 'timestep': o.timestep,
            'width': o.width, 'height': o.height, 'is_water': o.is_water}
    return out
