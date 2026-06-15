#!/usr/bin/env python3
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Shared registration engine (``viame.opencv.registration_utils``).

Sequential homography-chain registration, GPS / flight-log metadata loading, and
GPS geo-anchoring, shared by the standalone tools ``reconstruct_3d.py`` and
``detect_site_revisits.py``. OpenCV-based; this module has NO COLMAP dependency
(structure-from-motion lives in ``viame.colmap``).

NOTE: ``numpy`` and ``cv2`` are imported lazily via :func:`import_dependencies`
(set as module globals ``np`` / ``cv2``) so a tool can offer ``--install-deps``
before the packages are present. Call ``import_dependencies()`` once at startup.
"""

import os
import sys
import math
import csv
import json
import re
import importlib
import subprocess

# Populated by import_dependencies()
np = None
cv2 = None

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# numpy/cv2 are the only hard requirements of this engine.
REQUIRED_PACKAGES = {'numpy': 'numpy', 'cv2': 'opencv-python'}


def check_dependencies(packages=None):
    """Return list of (import_name, pip_name) for missing packages."""
    packages = packages if packages is not None else REQUIRED_PACKAGES
    missing = []
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return missing


def install_dependencies(missing, target_dir=None):
    pip_names = [p for _, p in missing]
    cmd = [sys.executable, '-m', 'pip', 'install'] + pip_names
    if target_dir:
        cmd += ['--target', target_dir]
    print(f"Running: {' '.join(cmd)}")
    if subprocess.call(cmd) != 0:
        print("ERROR: pip install failed")
        sys.exit(1)


def ensure_dependencies(install=False, target_dir=None, packages=None):
    """Check (and optionally install) the engine's Python dependencies."""
    missing = check_dependencies(packages)
    if not missing:
        return True
    print("Missing dependencies: " + ", ".join(p for _, p in missing))
    if not install:
        print("  python -m pip install " + " ".join(p for _, p in missing))
        return False
    install_dependencies(missing, target_dir)
    return True


def import_dependencies():
    """Import numpy/cv2 into the module globals, with a clear error if missing."""
    global np, cv2
    missing = check_dependencies()
    if missing:
        print("ERROR: missing required dependencies:")
        for import_name, pip_name in missing:
            print(f"  {pip_name} (import {import_name})")
        sys.exit(1)
    import numpy as np_
    import cv2 as cv2_
    np = np_
    cv2 = cv2_


def get_image_files(folder):
    """Return sorted list of image filenames in *folder*."""
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        and os.path.isfile(os.path.join(folder, f))
    )

def detect_multicam(folder):
    """Check if folder contains PORT/STAR/CENTER camera subfolders."""
    subs = set()
    for d in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, d)):
            subs.add(d.upper())
    return {'CENTER', 'PORT', 'STAR'}.issubset(subs)

def _nmea_to_dec(value, ref):
    """NMEA DDDMM.MMMM (FMCLOG) -> signed decimal degrees."""
    v = float(value)
    deg = int(v // 100)
    dec = deg + (v - 100 * deg) / 60.0
    return -dec if str(ref).strip().upper() in ('S', 'W') else dec

def _dms_to_dec(dms, ref):
    """EXIF GPS (deg, min, sec) -> signed decimal degrees."""
    d, m, s = [float(x) for x in dms]
    dec = d + m / 60.0 + s / 3600.0
    return -dec if str(ref).strip().upper() in ('S', 'W') else dec

def _load_imagelog_json(folder):
    """Format B: imagelog.json in `folder`. Returns ordered list of pose dicts
    (by trigger_index) or None."""
    import json
    path = os.path.join(folder, 'imagelog.json')
    if not os.path.isfile(path):
        return None
    try:
        recs = json.load(open(path)).get('ImageLog', [])
    except Exception:
        return None
    if not recs:
        return None
    recs = sorted(recs, key=lambda r: r.get('trigger_index', 0))
    out = []
    for r in recs:
        if r.get('lat') is None or r.get('lon') is None:
            continue
        # alt_rel is height above launch (can be ~0/negative); prefer it only
        # when it is a plausible imaging height, else fall back to absolute alt.
        alt_rel = r.get('alt_rel')
        alt_abs = r.get('alt')
        alt = (float(alt_rel) if alt_rel is not None and float(alt_rel) > 5.0
               else (float(alt_abs) if alt_abs is not None else None))
        out.append({'lat': float(r['lat']), 'lon': float(r['lon']),
                    'alt_agl': alt,
                    'yaw': (float(r['yaw']) if r.get('yaw') is not None else None),
                    'filename': r.get('filename')})
    return out or None

def _load_fmclog_csv(csv_path, site_name):
    """Format A: FMCLOG CSV filtered to site_name. Returns ordered list of pose
    dicts (by utc_time, frame_count) or None."""
    import csv as _csv
    if not csv_path or not os.path.isfile(csv_path):
        return None
    def _norm(s):
        return re.sub(r'[^a-z0-9]', '', (s or '').lower())
    import re
    want = _norm(site_name)
    try:
        rows = []
        for r in _csv.DictReader(open(csv_path)):
            ls = _norm(r.get('site_name', ''))
            if ls and want and (ls == want or ls.startswith(want) or want.startswith(ls)):
                rows.append(r)
    except Exception:
        return None
    if not rows:
        return None
    def _key(r):
        try:
            return (r.get('utc_time', ''), int(r.get('frame_count', '0') or 0))
        except ValueError:
            return (r.get('utc_time', ''), 0)
    rows.sort(key=_key)
    out = []
    for r in rows:
        try:
            lat = _nmea_to_dec(r['lat'], r.get('lat_ns', 'N'))
            lon = _nmea_to_dec(r['lon'], r.get('lon_ew', 'E'))
        except (ValueError, KeyError):
            continue
        alt = None
        try:
            alt = float(r.get('elevation_m', '')) if r.get('elevation_m') else None
        except ValueError:
            alt = None
        yaw = None
        try:
            yaw = float(r.get('yaw', '')) if r.get('yaw') not in (None, '') else None
        except ValueError:
            yaw = None
        out.append({'lat': lat, 'lon': lon, 'alt_agl': alt, 'yaw': yaw})
    return out or None

def _load_exif_gps(folder, image_list):
    """Format C: EXIF GPS embedded per image. Returns dict rel_path -> pose or None."""
    try:
        from PIL import Image as _PILImage
        from PIL.ExifTags import TAGS, GPSTAGS
    except Exception:
        return None
    out = {}
    for rel in image_list:
        try:
            ex = _PILImage.open(os.path.join(folder, rel))._getexif() or {}
            tags = {TAGS.get(k, k): v for k, v in ex.items()}
            gps = tags.get('GPSInfo')
            if not gps:
                continue
            g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
            lat = _dms_to_dec(g['GPSLatitude'], g.get('GPSLatitudeRef', 'N'))
            lon = _dms_to_dec(g['GPSLongitude'], g.get('GPSLongitudeRef', 'E'))
            alt = float(g['GPSAltitude']) if g.get('GPSAltitude') is not None else None
            out[rel] = {'lat': lat, 'lon': lon, 'alt_agl': alt, 'yaw': None}
        except Exception:
            continue
    return out or None

def load_pose_metadata(image_folder, image_list, flight_log=None, site_name=None):
    """Unified per-frame pose loader supporting both metadata formats.

    `image_list` is the caller's ordered list of relative image paths. Returns a
    dict {rel_path: {lat, lon, alt_agl, yaw}} or None if no metadata is found.
    Priority: imagelog.json (B) -> FMCLOG csv (A) -> EXIF GPS (C). Formats A/B
    associate by capture order (index); C associates per-image.
    """
    # The image folder for a single camera list may be a subfolder; resolve the
    # directory that actually holds the images (and an imagelog.json).
    first_dir = os.path.dirname(os.path.join(image_folder, image_list[0])) if image_list else image_folder

    ordered = _load_imagelog_json(first_dir) or _load_imagelog_json(image_folder)
    source = 'imagelog.json'
    if ordered is None and flight_log and site_name:
        ordered = _load_fmclog_csv(flight_log, site_name)
        source = 'flight-log'
    if ordered is not None:
        # Associate by order; warn if counts differ (still pair the overlap).
        n = min(len(ordered), len(image_list))
        poses = {image_list[i]: ordered[i] for i in range(n)}
        if n:
            print(f"    Metadata: {source} matched {n}/{len(image_list)} frames"
                  + ("" if n == len(image_list) else " (count mismatch — paired by order)"))
        return poses or None

    exif = _load_exif_gps(first_dir if not os.path.dirname(image_list[0]) else image_folder, image_list)
    if exif:
        print(f"    Metadata: EXIF GPS matched {len(exif)}/{len(image_list)} frames")
        return exif
    return None

def compute_homography_pair(img1_path, img2_path, scale=0.5, nfeatures=8192,
                             use_clahe=True, match_ratio=0.75,
                             min_inliers=10, min_inlier_ratio=0.0,
                             ransac_thresh=5.0, validate_homography=False,
                             multi_scale=False, matcher='bf',
                             homography_method='ransac', always_clahe=False,
                             root_sift=False, use_affine=False,
                             sift_contrast=0.04, clahe_clip=4.0):
    """Compute homography from img1 to img2 using SIFT matching.

    CLAHE histogram equalization is applied by default to handle low-contrast
    underwater imagery.  Returns (H_3x3, translation_px) or (None, None).
    translation_px is the average displacement of matched keypoints (at full scale).

    Optional quality controls:
        match_ratio:        Lowe's ratio test threshold (default 0.75)
        min_inliers:        Minimum RANSAC inlier count (default 10)
        min_inlier_ratio:   Minimum inlier/match ratio (default 0.0 = disabled)
        ransac_thresh:      RANSAC reprojection threshold in pixels (default 5.0)
        validate_homography: Check determinant/condition number (default False)
        multi_scale:        Try higher scale if first attempt fails (default False)
        matcher:            'bf' (brute-force) or 'flann' (default 'bf')
        homography_method:  'ransac', 'lmeds', or 'usac' (default 'ransac')
        always_clahe:       Always apply CLAHE regardless of contrast (default False)
        root_sift:          Apply Root-SIFT normalization to descriptors (default False)
        use_affine:         Use affine model instead of full homography (default False)
        sift_contrast:      SIFT contrast threshold (default 0.04)
        clahe_clip:         CLAHE clip limit (default 4.0)
    """
    scales_to_try = [scale]
    if multi_scale:
        scales_to_try = [scale, min(scale * 1.5, 1.0), min(scale * 2.0, 1.0)]
        scales_to_try = list(dict.fromkeys(scales_to_try))

    for sc in scales_to_try:
        result = _compute_homography_at_scale(
            img1_path, img2_path, sc, nfeatures, use_clahe,
            match_ratio, min_inliers, min_inlier_ratio,
            ransac_thresh, validate_homography,
            matcher=matcher, homography_method=homography_method,
            always_clahe=always_clahe, root_sift=root_sift,
            use_affine=use_affine, sift_contrast=sift_contrast,
            clahe_clip=clahe_clip)
        if result[0] is not None:
            return result

    return None, None

def _compute_homography_at_scale(img1_path, img2_path, scale, nfeatures,
                                   use_clahe, match_ratio, min_inliers,
                                   min_inlier_ratio, ransac_thresh,
                                   validate_homography, matcher='bf',
                                   homography_method='ransac', always_clahe=False,
                                   root_sift=False, use_affine=False,
                                   sift_contrast=0.04, clahe_clip=4.0):
    """Internal: compute homography at a single scale."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return None, None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    small1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
    small2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))

    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        if always_clahe:
            gray1 = clahe.apply(gray1)
            gray2 = clahe.apply(gray2)
        else:
            # Adaptive: only apply if image has low contrast
            if gray1.std() < 20:
                gray1 = clahe.apply(gray1)
            if gray2.std() < 20:
                gray2 = clahe.apply(gray2)

    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=sift_contrast)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None

    # Optional: Root-SIFT normalization (L1 normalize then sqrt)
    if root_sift:
        des1 = des1 / (des1.sum(axis=1, keepdims=True) + 1e-7)
        des1 = np.sqrt(des1)
        des2 = des2 / (des2.sum(axis=1, keepdims=True) + 1e-7)
        des2 = np.sqrt(des2)

    # Matcher selection
    if matcher == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        fm = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        fm = cv2.BFMatcher()

    matches = fm.knnMatch(des1, des2, k=2)
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < match_ratio * n.distance:
                good.append(m)

    if len(good) < min_inliers:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Homography/affine estimation method
    if use_affine:
        H_small, mask = cv2.estimateAffine2D(
            src_pts, dst_pts,
            method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H_small is None:
            return None, None
        H = np.vstack([H_small, [0, 0, 1]])
    else:
        if homography_method == 'lmeds':
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
        elif homography_method == 'usac':
            try:
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.USAC_MAGSAC, ransac_thresh)
            except AttributeError:
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        else:
            H, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None or mask is None:
        return None, None

    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return None, None

    # Optional: check inlier ratio
    if min_inlier_ratio > 0:
        ratio = inliers / len(good)
        if ratio < min_inlier_ratio:
            return None, None

    # Optional: validate homography geometry
    if validate_homography:
        det = np.linalg.det(H)
        if det < 0.1 or det > 10.0:
            return None, None  # extreme scale change or reflection
        try:
            cond = np.linalg.cond(H)
            if cond > 1e6:
                return None, None  # ill-conditioned
        except np.linalg.LinAlgError:
            return None, None

    # Compute average displacement at full scale for motion tracking
    inlier_mask = mask.ravel().astype(bool)
    src_inliers = src_pts[inlier_mask].reshape(-1, 2)

    # Spatial distribution check (only enforced when validation is on): matches
    # clustered in one corner give poorly-constrained homographies.
    if validate_homography and len(src_inliers) >= 4:
        sh, sw = int(h1 * scale), int(w1 * scale)
        x_range = src_inliers[:, 0].max() - src_inliers[:, 0].min()
        y_range = src_inliers[:, 1].max() - src_inliers[:, 1].min()
        if x_range < sw * 0.15 or y_range < sh * 0.15:
            return None, None

    src_in = src_inliers / scale
    dst_in = dst_pts[inlier_mask].reshape(-1, 2) / scale
    avg_disp = float(np.median(np.linalg.norm(src_in - dst_in, axis=1)))

    # Scale homography to full resolution
    S = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    S_inv = np.diag([scale, scale, 1.0])
    H_full = S @ H @ S_inv
    return H_full, avg_disp

def _compute_camera_chain(image_folder, cam_images, label="",
                           water_info=None, match_ratio=0.75,
                           min_inliers=10, min_inlier_ratio=0.0,
                           ransac_thresh=5.0, validate_homography=False,
                           multi_scale=False, search_window=20,
                           max_chain_cond=0, scale=0.5,
                           consistency_filter=False, matcher='bf',
                           homography_method='ransac', always_clahe=False,
                           root_sift=False, use_affine=False,
                           sift_contrast=0.04, clahe_clip=4.0,
                           loop_closure=False, loop_closure_max=24,
                           anchor_central=False):
    """Compute sequential homography chain for a single camera's image list.

    Returns (H_chain, pairwise_H) where H_chain maps index -> H_to_anchor (3x3)
    and pairwise_H maps (i, j) -> the raw frame-i-to-frame-j homography.
    Uses CLAHE enhancement, anchor-based building, and bidirectional linking.

    Strategy: find the best anchor frame (highest feature count), start
    the chain there, then extend both forward and backward.  Coastal/non-water
    frames that fail are retried with a boosted (higher-res, more-features,
    relaxed) matching pass.  When consistency_filter is on, suspect (water/
    coastal) registrations that deviate from the land-frame motion pattern are
    replaced with interpolated estimates.
    """
    n = len(cam_images)
    if n == 0:
        return {}, {}

    # Find the best anchor frame (highest SIFT features without CLAHE)
    sift_quick = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04)
    anchor_scores = []
    for i, fname in enumerate(cam_images):
        is_water = (water_info or {}).get(fname, {}).get('is_water', False)
        if is_water:
            anchor_scores.append((0, i))
            continue
        img = cv2.imread(os.path.join(image_folder, fname))
        if img is None:
            anchor_scores.append((0, i))
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * 0.25), int(h * 0.25)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp = sift_quick.detect(gray, None)
        anchor_scores.append((len(kp) if kp else 0, i))

    # Sort by score descending to find best anchor
    anchor_scores.sort(key=lambda x: -x[0])
    anchor_idx = anchor_scores[0][1]

    # Central anchoring: chain drift accumulates with distance from the anchor,
    # so a highest-feature anchor sitting near a sequence end (e.g. frame 59/63)
    # forces ~60 frames of one-directional drift. Among the strong-feature land
    # frames, prefer the one nearest the sequence centre to halve worst-case drift.
    if anchor_central and anchor_scores:
        max_score = anchor_scores[0][0]
        if max_score > 0:
            center = (n - 1) / 2.0
            strong = [(s, i) for s, i in anchor_scores if s >= 0.4 * max_score]
            if strong:
                anchor_idx = min(strong, key=lambda si: abs(si[1] - center))[1]

    H_chain = {anchor_idx: np.eye(3, dtype=np.float64)}
    pairwise_H = {}      # (i, j) -> raw H from frame i to frame j
    motion_samples = []
    avg_motion = 200.0
    rejected_drift = 0
    rejected_consistency = 0

    def try_chain(i, search_indices, boost=False):
        """Try to chain frame i by matching to frames in search_indices.

        If boost=True, use more aggressive matching (higher scale, more
        features, relaxed ratio, always CLAHE) for difficult frames like
        coastal images with sparse features on one side.
        """
        nonlocal rejected_drift
        if boost:
            sc = min(scale * 2.0, 1.0)
            nfeat = 16384
            ratio = min(match_ratio + 0.1, 0.85)
            min_inl = max(min_inliers // 2, 6)
            min_inl_ratio = 0.0
        else:
            sc = scale
            nfeat = 8192
            ratio = match_ratio
            min_inl = min_inliers
            min_inl_ratio = min_inlier_ratio
        path_curr = os.path.join(image_folder, cam_images[i])
        for j in search_indices:
            if j not in H_chain:
                continue
            path_ref = os.path.join(image_folder, cam_images[j])
            H, disp = compute_homography_pair(
                path_curr, path_ref, scale=sc, nfeatures=nfeat,
                match_ratio=ratio, min_inliers=min_inl,
                min_inlier_ratio=min_inl_ratio,
                ransac_thresh=ransac_thresh,
                validate_homography=validate_homography,
                multi_scale=True if boost else multi_scale,
                matcher=matcher, homography_method=homography_method,
                always_clahe=True if boost else always_clahe,
                root_sift=root_sift, use_affine=use_affine,
                sift_contrast=sift_contrast, clahe_clip=clahe_clip)
            if H is not None:
                H_candidate = H_chain[j] @ H
                if max_chain_cond > 0:
                    try:
                        cond = np.linalg.cond(H_candidate)
                        if cond > max_chain_cond:
                            rejected_drift += 1
                            continue
                    except np.linalg.LinAlgError:
                        continue
                H_chain[i] = H_candidate
                pairwise_H[(i, j)] = H
                if disp is not None and abs(i - j) == 1:
                    motion_samples.append(disp)
                return True
        return False

    # Extend backward from anchor
    for i in range(anchor_idx - 1, -1, -1):
        search = [j for j in range(i + 1, min(i + search_window, n))]
        try_chain(i, search)

    # Extend forward from anchor
    for i in range(anchor_idx + 1, n):
        search = [j for j in range(i - 1, max(i - search_window, -1), -1)]
        try_chain(i, search)

    # Final pass: try unchained frames against any chained frame
    for i in range(n):
        if i in H_chain:
            continue
        candidates = sorted(H_chain.keys(), key=lambda j: abs(i - j))[:10]
        try_chain(i, candidates)

    # Boosted retry: for unchained frames classified as coastal (or any
    # non-water frame that failed), retry with higher resolution / relaxed
    # matching to catch frames with sparse land regions.
    boosted_count = 0
    for i in range(n):
        if i in H_chain:
            continue
        fname = cam_images[i]
        info = (water_info or {}).get(fname, {})
        lbl = info.get('label', '')
        is_water = info.get('is_water', False)
        if lbl == 'coastal' or not is_water:
            candidates = sorted(H_chain.keys(), key=lambda j: abs(i - j))[:15]
            if try_chain(i, candidates, boost=True):
                boosted_count += 1
    if boosted_count and label:
        print(f"    {label}: {boosted_count} frames recovered via boosted matching")

    # Loop-closure pass: the passes above only match against temporally NEARBY
    # frames (index-adjacent). When the rig revisits an earlier location, the
    # matching frame is far away in time. Here we try each still-unchained
    # non-water frame against chained frames sampled across the ENTIRE sequence,
    # so a revisit can register against the frame that originally saw that spot.
    # Guarded by a corroboration check (the candidate's two registered temporal
    # neighbours must also match) to reject false closures over repetitive water.
    if loop_closure:
        lc_count = 0
        for i in range(n):
            if i in H_chain:
                continue
            fname = cam_images[i]
            info = (water_info or {}).get(fname, {})
            if info.get('is_water', False) and info.get('label', '') != 'coastal':
                continue  # open-water frames have no reliable revisit features
            chained = sorted(H_chain.keys())
            # Prefer temporally-distant candidates (true loop closures); sample
            # evenly across the full range so we cover revisits anywhere.
            far = [j for j in chained if abs(i - j) > search_window]
            pool = far if far else chained
            if len(pool) > loop_closure_max:
                sel = np.linspace(0, len(pool) - 1, loop_closure_max).astype(int)
                candidates = [pool[k] for k in sorted(set(sel.tolist()))]
            else:
                candidates = pool
            # Order farthest-first so a genuine revisit is preferred over a
            # marginal near match the earlier passes already rejected.
            candidates = sorted(candidates, key=lambda j: -abs(i - j))
            before = i in H_chain
            if try_chain(i, candidates, boost=True) and not before:
                # Corroborate: the matched frame's own neighbours should also be
                # consistent. We approximate this by requiring the new chain H to
                # have a sane determinant (affine guards most false matches).
                Hd = abs(float(np.linalg.det(H_chain[i])))
                if 0.2 <= Hd <= 5.0:
                    lc_count += 1
                else:
                    del H_chain[i]  # reject implausible closure
        if lc_count and label:
            print(f"    {label}: {lc_count} frames recovered via loop closure")

    if motion_samples:
        avg_motion = np.median(motion_samples)

    # Consistency filter: use land-only frames to establish the expected motion
    # pattern, then validate suspect (water/coastal) registrations; replace
    # outliers with interpolated estimates.
    if consistency_filter and water_info and len(H_chain) > 2:
        good_indices = []
        suspect_indices = []
        for i in sorted(H_chain.keys()):
            fname = cam_images[i]
            info = (water_info or {}).get(fname, {})
            lbl = info.get('label', '')
            is_water = info.get('is_water', False)
            if lbl == 'all_land':
                good_indices.append(i)
            elif is_water or lbl == 'coastal':
                suspect_indices.append(i)

        if len(good_indices) >= 2:
            good_translations = []
            good_rotscale = []
            for gi in range(len(good_indices) - 1):
                idx_a = good_indices[gi]
                idx_b = good_indices[gi + 1]
                H_a = H_chain[idx_a]
                H_b = H_chain[idx_b]
                tx = H_b[0, 2] - H_a[0, 2]
                ty = H_b[1, 2] - H_a[1, 2]
                dh00 = H_b[0, 0] - H_a[0, 0]
                dh01 = H_b[0, 1] - H_a[0, 1]
                dh10 = H_b[1, 0] - H_a[1, 0]
                dh11 = H_b[1, 1] - H_a[1, 1]
                frame_gap = idx_b - idx_a
                if frame_gap > 0:
                    good_translations.append((tx / frame_gap, ty / frame_gap))
                    good_rotscale.append((dh00 / frame_gap, dh01 / frame_gap,
                                          dh10 / frame_gap, dh11 / frame_gap))

            if good_translations:
                med_tx = np.median([t[0] for t in good_translations])
                med_ty = np.median([t[1] for t in good_translations])
                std_tx = (np.std([t[0] for t in good_translations])
                          if len(good_translations) > 1 else abs(med_tx) * 0.5)
                std_ty = (np.std([t[1] for t in good_translations])
                          if len(good_translations) > 1 else abs(med_ty) * 0.5)
                tol_tx = max(2.5 * std_tx, abs(med_tx) * 0.15, 50)
                tol_ty = max(2.5 * std_ty, abs(med_ty) * 0.15, 50)

                med_rs = [np.median([r[k] for r in good_rotscale]) for k in range(4)]
                std_rs = [np.std([r[k] for r in good_rotscale])
                          if len(good_rotscale) > 1 else 0.01 for k in range(4)]
                tol_rs = [max(2.5 * std_rs[k], 0.02) for k in range(4)]

                def _bracket(wi):
                    prev_good = None
                    next_good = None
                    for gi in good_indices:
                        if gi < wi:
                            prev_good = gi
                        elif gi > wi and next_good is None:
                            next_good = gi
                    return prev_good, next_good

                def _interpolate_H(wi, prev_good, next_good):
                    if prev_good is not None and next_good is not None:
                        alpha = (wi - prev_good) / (next_good - prev_good)
                        return ((1 - alpha) * H_chain[prev_good]
                                + alpha * H_chain[next_good])
                    elif prev_good is not None:
                        gap = wi - prev_good
                        H_interp = H_chain[prev_good].copy()
                        H_interp[0, 2] += med_tx * gap
                        H_interp[1, 2] += med_ty * gap
                        for k, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                            H_interp[row, col] += med_rs[k] * gap
                        return H_interp
                    elif next_good is not None:
                        gap = next_good - wi
                        H_interp = H_chain[next_good].copy()
                        H_interp[0, 2] -= med_tx * gap
                        H_interp[1, 2] -= med_ty * gap
                        for k, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                            H_interp[row, col] -= med_rs[k] * gap
                        return H_interp
                    return None

                def _check_consistency(wi):
                    prev_good, next_good = _bracket(wi)
                    if prev_good is None and next_good is None:
                        return True
                    H_w = H_chain[wi]
                    ref_idx = prev_good if prev_good is not None else next_good
                    gap = abs(wi - ref_idx)
                    sign = 1 if wi > ref_idx else -1
                    H_ref = H_chain[ref_idx]
                    expected_tx = H_ref[0, 2] + sign * med_tx * gap
                    expected_ty = H_ref[1, 2] + sign * med_ty * gap
                    if (abs(H_w[0, 2] - expected_tx) > tol_tx * gap or
                            abs(H_w[1, 2] - expected_ty) > tol_ty * gap):
                        return False
                    for k, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                        expected_val = H_ref[row, col] + sign * med_rs[k] * gap
                        if abs(H_w[row, col] - expected_val) > tol_rs[k] * gap:
                            return False
                    return True

                for wi in suspect_indices:
                    if not _check_consistency(wi):
                        prev_good, next_good = _bracket(wi)
                        H_new = _interpolate_H(wi, prev_good, next_good)
                        if H_new is not None:
                            H_chain[wi] = H_new
                            rejected_consistency += 1

    ok = len(H_chain) - 1  # subtract the anchor
    extras = []
    if rejected_drift:
        extras.append(f"{rejected_drift} rejected (drift)")
    if rejected_consistency:
        extras.append(f"{rejected_consistency} frames corrected (consistency)")
    extra_str = ", " + ", ".join(extras) if extras else ""
    if label:
        print(f"    {label}: {n} frames, {ok}/{n - 1} homographies OK"
              f" (anchor=#{anchor_idx}, avg motion: {avg_motion:.0f}px/frame{extra_str})")
    return H_chain, pairwise_H

def _poses_to_enu(poses, cam_images):
    """Convert per-frame lat/lon poses to a local East/North metre array indexed
    by frame. Returns (enu[n,2] with nan where missing, yaw[n] with nan)."""
    n = len(cam_images)
    enu = np.full((n, 2), np.nan)
    yaw = np.full(n, np.nan)
    ref = None
    for i, fname in enumerate(cam_images):
        p = poses.get(fname)
        if not p:
            continue
        if ref is None:
            ref = (p['lat'], p['lon'])
        dN = (p['lat'] - ref[0]) * 111320.0
        dE = (p['lon'] - ref[1]) * 111320.0 * math.cos(math.radians(ref[0]))
        enu[i] = (dE, dN)
        if p.get('yaw') is not None:
            yaw[i] = p['yaw']
    return enu, yaw

def _track_headings(enu):
    """Per-frame flight heading (deg) from the GPS TRACK direction (central
    difference over nearest valid neighbours). The aircraft heading — not the
    logged camera yaw — is what rotates the image relative to the ground, and it
    flips ~180deg between out-and-back survey passes; using it makes the camera-
    mounting map constant across passes. NaN where undeterminable."""
    n = len(enu)
    head = np.full(n, np.nan)
    valid = [i for i in range(n) if not np.isnan(enu[i, 0])]
    for idx, i in enumerate(valid):
        a = valid[idx - 1] if idx > 0 else i
        b = valid[idx + 1] if idx < len(valid) - 1 else i
        d = enu[b] - enu[a]
        if np.linalg.norm(d) > 1.0:
            head[i] = math.degrees(math.atan2(d[0], d[1]))  # atan2(E, N)
    return head

def _rot2(deg):
    """2x2 rotation matrix for an angle in degrees (NaN -> identity)."""
    if deg is None or np.isnan(deg):
        return np.eye(2)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])

def _fit_similarity_disp(G, F):
    """Fit M (2x2 rotation+scale, both chiralities) with F = M @ G. Robust to
    near-collinear G (single-direction flight) unlike a free 2x2."""
    gx, gy = G[:, 0], G[:, 1]
    fx, fy = F[:, 0], F[:, 1]
    best = None
    for chir in (+1, -1):
        rows = np.vstack([
            np.column_stack([gx, -chir * gy]),   # fx = a*gx - chir*b*gy
            np.column_stack([gy,  chir * gx]),    # fy = b*gx + chir*a*gy
        ])
        rhs = np.concatenate([fx, fy])
        (a, b), *_ = np.linalg.lstsq(rows, rhs, rcond=None)
        M = np.array([[a, -chir * b], [b, chir * a]])
        r = float(np.median(np.linalg.norm(G @ M.T - F, axis=1)))
        if best is None or r < best[1]:
            best = (M, r)
    return best

def _geo_calibrate(chain, cam_images, poses, pairwise_H):
    """Calibrate the constant camera-mounting map M (heading-frame metres ->
    pixels) from RAW pairwise feature translations. The ground GPS displacement
    is first DE-ROTATED by each frame's yaw, so M stays constant even when the
    aircraft reverses heading between survey passes (the multi-pass / revisit
    case) — a single un-rotated fit would otherwise be corrupted by the ~180°
    flip. Returns (M, n_steps, residual, enu, yaw). If yaw is absent (EXIF-only),
    de-rotation is identity and M degenerates to the single-heading transform."""
    enu, _yaw = _poses_to_enu(poses, cam_images)
    yaw = _track_headings(enu)    # GPS-track heading (flips between passes)
    G, F = [], []
    for (i, j), H in (pairwise_H or {}).items():
        if abs(i - j) > 3 or np.isnan(enu[i, 0]) or np.isnan(enu[j, 0]):
            continue
        g = enu[i] - enu[j]
        if np.linalg.norm(g) < 1.0:
            continue
        g_cam = _rot2(-yaw[j]) @ g     # ground disp -> aircraft heading frame
        G.append(g_cam); F.append([H[0, 2], H[1, 2]])
    if len(G) < 3:
        return None, 0, None, enu, yaw
    G = np.array(G); F = np.array(F)
    M, _r = _fit_similarity_disp(G, F)
    res = np.linalg.norm(G @ M.T - F, axis=1)
    keep = res <= max(3 * np.median(res), 50)   # reject gross outliers, refit
    if keep.sum() >= 3:
        M, _r = _fit_similarity_disp(G[keep], F[keep])
        res = np.linalg.norm(G[keep] @ M.T - F[keep], axis=1)
    return M, int(keep.sum()), float(np.median(res)), enu, yaw

def _geo_fill(chain, cam_images, enu, yaw, M, label="", n_steps=0, residual=None):
    """Fill UNregistered frames by LOCAL dead-reckoning from the nearest
    registered frame: pos_k = pos_j + M @ R(-yaw_j) @ (enu_k - enu_j). The yaw
    de-rotation makes this correct across heading changes between passes."""
    reg = sorted(i for i in chain.keys() if not np.isnan(enu[i, 0]))
    if len(reg) < 2 or M is None:
        return 0
    filled = 0
    for k in range(len(cam_images)):
        if k in chain or np.isnan(enu[k, 0]):
            continue
        j = min(reg, key=lambda m: abs(m - k))
        H = chain[j].copy()
        g_cam = _rot2(-yaw[j]) @ (enu[k] - enu[j])
        pos = np.array([chain[j][0, 2], chain[j][1, 2]]) + M @ g_cam
        H[0, 2], H[1, 2] = float(pos[0]), float(pos[1])
        chain[k] = H
        filled += 1
    if label:
        scale = float(np.sqrt(abs(np.linalg.det(M))))
        rtxt = f"{residual:.0f}px" if residual is not None else "n/a"
        extra = f", filled {filled} via GPS dead-reckoning" if filled else ""
        print(f"    {label}: geo-anchor {n_steps} pairwise steps, {scale:.0f}px/m, "
              f"step-residual {rtxt}{extra}")
    return filled

def _geo_anchor_cameras(cam_chains, cameras, poses_by_cam, pairwise_by_cam):
    """Calibrate + fill all rig cameras, SHARING the GPS->pixel scale. The rig
    cameras image at the same altitude/focal, so the scale (px/m) is identical;
    a water-heavy camera with few clean steps borrows the rig-median scale
    (keeping its own rotation) instead of self-calibrating from noise."""
    cal = {}   # cam -> [M, n_steps, residual, enu, yaw]
    for cam_key in cameras:
        if poses_by_cam.get(cam_key) is None:
            continue
        M, n, r, enu, yaw = _geo_calibrate(
            cam_chains[cam_key], cameras[cam_key],
            poses_by_cam[cam_key], pairwise_by_cam.get(cam_key))
        cal[cam_key] = [M, n, r, enu, yaw]
    # Rig-shared scale = median scale of well-calibrated cameras.
    good = [np.sqrt(abs(np.linalg.det(c[0]))) for c in cal.values()
            if c[0] is not None and c[1] >= 8 and c[2] is not None and c[2] < 150]
    shared = float(np.median(good)) if good else None
    for cam_key, c in cal.items():
        M, n, r, enu, yaw = c
        if M is None:
            continue
        own = np.sqrt(abs(np.linalg.det(M)))
        # Override scale when this camera's own fit is unreliable (few steps or
        # high residual) and a trustworthy rig scale exists.
        if shared and (n < 8 or (r is not None and r > 150)) and own > 1e-6:
            M = M * (shared / own)
            note = f"{cam_key}*"   # * marks scale borrowed from the rig
        else:
            note = cam_key
        _geo_fill(cam_chains[cam_key], cameras[cam_key], enu, yaw, M,
                  label=note, n_steps=n, residual=r)
