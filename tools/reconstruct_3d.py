#!/usr/bin/env python3
"""
reconstruct_3d.py - Convert UAS drone imagery folders into 3D mesh models.

Pipeline:
  1. SIFT feature extraction (pycolmap)
  2. Feature matching (exhaustive <=50 images, sequential otherwise)
  3. Incremental Structure from Motion -> sparse point cloud + cameras
  3b. (Optional) Prior-coverage polygon output (--coverage-class)
  4. Densification (one of two methods):                                [skip with --no-dense]
     a. --dense-method sift (default): OpenCV SIFT matching + triangulation (CPU)
     b. --dense-method mvs: COLMAP PatchMatch stereo + fusion (requires CUDA)
  5. Poisson surface reconstruction (Open3D) -> triangle mesh          [skip with --no-dense]
  6. Export PLY point cloud + PLY/OBJ mesh

Usage:
  python reconstruct_3d.py --install-deps          # check/install dependencies
  python reconstruct_3d.py <image_folder> [--output <output_dir>] [--scale <0.25>]
  python reconstruct_3d.py <image_folder> --dense-method mvs   # use COLMAP MVS (GPU)
  python reconstruct_3d.py --all                    # process all subfolders
"""

import os
import sys
import json
import math
import argparse
import time
import subprocess
import importlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

# Map of import name -> pip package name
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'pycolmap': 'pycolmap',
    'open3d': 'open3d',
}


def check_dependencies():
    """Check which required packages are missing. Returns list of (import_name, pip_name)."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return missing


def install_dependencies(missing, target_dir=None):
    """Install missing packages via pip in a subprocess.

    Args:
        missing: list of (import_name, pip_name) tuples
        target_dir: if set, pass --target to pip to install into that directory
    """
    pip_names = [pip_name for _, pip_name in missing]
    cmd = [sys.executable, '-m', 'pip', 'install'] + pip_names
    if target_dir:
        cmd += ['--target', target_dir]
    print(f"Running: {' '.join(cmd)}")
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"ERROR: pip install failed (exit code {ret})")
        sys.exit(1)
    print("Dependencies installed successfully.")


def ensure_dependencies(install=False, target_dir=None):
    """Check deps and optionally install. Returns True if all present."""
    missing = check_dependencies()
    if not missing:
        print("All dependencies are installed.")
        return True

    print("Missing dependencies:")
    for import_name, pip_name in missing:
        print(f"  {pip_name} (import {import_name})")

    if not install:
        print(f"\nRun with --install-deps to install them, or manually:")
        pip_names = ' '.join(p for _, p in missing)
        print(f"  python -m pip install {pip_names}")
        return False

    # Prompt user for confirmation
    print(f"\nThe following packages will be installed: "
          f"{', '.join(p for _, p in missing)}")
    if target_dir:
        print(f"Target directory: {target_dir}")
    response = input("Proceed? [y/N] ").strip().lower()
    if response not in ('y', 'yes'):
        print("Aborted.")
        sys.exit(0)

    install_dependencies(missing, target_dir)
    return True


def _get_viame_site_packages():
    """Try to find the VIAME install's site-packages directory."""
    # Check for VIAME install env variable
    viame_dir = os.environ.get('VIAME_INSTALL')
    if viame_dir:
        sp = os.path.join(viame_dir, 'lib', 'python' +
                          f'{sys.version_info.major}.{sys.version_info.minor}',
                          'site-packages')
        if os.path.isdir(sp):
            return sp
    # Fall back to user site-packages (pip default without --target)
    return None


def import_dependencies():
    """Import required packages, with a clear error message if missing."""
    global np, cv2, pycolmap, o3d
    try:
        import numpy as np_
        import cv2 as cv2_
        import pycolmap as pycolmap_
        import open3d as o3d_
        np = np_
        cv2 = cv2_
        pycolmap = pycolmap_
        o3d = o3d_
    except ImportError:
        missing = check_dependencies()
        if missing:
            print("ERROR: Missing required dependencies:")
            for import_name, pip_name in missing:
                print(f"  {pip_name} (import {import_name})")
            print(f"\nRun: python {sys.argv[0]} --install-deps")
            sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def get_image_files(folder):
    """Return sorted list of image filenames in *folder*."""
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        and os.path.isfile(os.path.join(folder, f))
    )


# ---------------------------------------------------------------------------
# Multi-camera support
# ---------------------------------------------------------------------------

def detect_multicam(folder):
    """Check if folder contains PORT/STAR/CENTER camera subfolders."""
    subs = set()
    for d in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, d)):
            subs.add(d.upper())
    return {'CENTER', 'PORT', 'STAR'}.issubset(subs)


def get_multicam_image_files(folder):
    """Return list of relative image paths from PORT/STAR/CENTER subfolders,
    ordered by sequence number then camera (CENTER, PORT, STAR)."""
    import re
    images = []
    cam_dirs = {}
    for d in os.listdir(folder):
        full = os.path.join(folder, d)
        if os.path.isdir(full) and d.upper() in ('CENTER', 'PORT', 'STAR'):
            cam_dirs[d.upper()] = d

    for cam_key in ('CENTER', 'PORT', 'STAR'):
        if cam_key not in cam_dirs:
            continue
        cam_dir = cam_dirs[cam_key]
        cam_path = os.path.join(folder, cam_dir)
        for f in sorted(os.listdir(cam_path)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                if os.path.isfile(os.path.join(cam_path, f)):
                    images.append(os.path.join(cam_dir, f))

    # Sort by sequence number then camera
    def sort_key(name):
        match = re.search(r'(\d+)\.\w+$', name)
        seq = int(match.group(1)) if match else 0
        parts = name.split(os.sep)
        cam = parts[0].upper() if len(parts) > 1 else ''
        cam_order = {'CENTER': 0, 'PORT': 1, 'STAR': 2}.get(cam, 9)
        return (seq, cam_order)

    images.sort(key=sort_key)
    return images


def timeit(msg):
    """Simple context-manager timer."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  {msg}")
            print(f"{'='*60}", flush=True)
            return self
        def __exit__(self, *_):
            dt = time.time() - self.t0
            print(f"  -> done in {dt:.1f}s", flush=True)
    return _Timer()


# ---------------------------------------------------------------------------
# Stage 1-3: Structure from Motion via pycolmap
# ---------------------------------------------------------------------------

def run_sfm(image_folder, output_dir, image_names, multicam=False):
    """Run incremental SfM.  Returns a pycolmap.Reconstruction or None.

    If multicam=True, image_names should be relative paths including subfolder
    (e.g. 'CENTER/img.jpg') and PER_FOLDER camera mode is used.
    """
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    n = len(image_names)

    # Use SINGLE camera mode even for multicam - the cameras in a rig are
    # typically the same model, and sharing intrinsics helps SfM register
    # more images across cameras.
    cam_mode = pycolmap.CameraMode.SINGLE

    # --- feature extraction ---
    with timeit(f"Extracting SIFT features from {n} images"
                f" ({'multicam PER_FOLDER' if multicam else 'single camera'})"):
        ext_opts = pycolmap.FeatureExtractionOptions()
        ext_opts.max_image_size = 3200
        sift = pycolmap.SiftExtractionOptions()
        sift.max_num_features = 8192
        ext_opts.sift = sift

        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_folder,
            image_names=image_names,
            camera_mode=cam_mode,
            camera_model="OPENCV",
            extraction_options=ext_opts,
        )

    # --- matching ---
    # For multicam, force exhaustive matching so cross-camera pairs are found
    with timeit("Matching features"):
        if n <= 50 or (multicam and n <= 200):
            print(f"  (exhaustive matching, {n} images)")
            pycolmap.match_exhaustive(database_path=db_path)
        else:
            print("  (sequential matching, overlap=15)")
            seq = pycolmap.SequentialPairingOptions()
            seq.overlap = 15
            pycolmap.match_sequential(database_path=db_path, pairing_options=seq)

    # --- incremental mapping ---
    with timeit("Incremental SfM"):
        mapper_opts = pycolmap.IncrementalPipelineOptions()
        mapper_opts.min_num_matches = 15
        maps = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=image_folder,
            output_path=sparse_dir,
            options=mapper_opts,
        )

    if not maps:
        print("ERROR: SfM produced no reconstruction.")
        return None

    best_idx = max(maps.keys(), key=lambda k: maps[k].num_points3D())
    rec = maps[best_idx]
    print(f"  Reconstructions: {len(maps)}")
    print(f"  Best map: {rec.num_reg_images()}/{n} images registered, "
          f"{rec.num_points3D()} sparse 3D points")

    rec.write(sparse_dir)
    return rec


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def get_camera_matrix(cam):
    """Extract 3x3 intrinsic matrix K from a pycolmap Camera."""
    params = cam.params
    model = cam.model.name
    if model == "OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx = fy = params[0]
        cx, cy = params[1] if len(params) > 1 else 0, params[2] if len(params) > 2 else 0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def get_dist_coeffs(cam):
    """Extract distortion coefficients from a pycolmap Camera."""
    params = cam.params
    model = cam.model.name
    if model == "OPENCV":
        return np.array([params[4], params[5], params[6], params[7]], dtype=np.float64)
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        k1 = params[3] if len(params) > 3 else 0
        k2 = params[4] if len(params) > 4 else 0
        return np.array([k1, k2, 0, 0], dtype=np.float64)
    return np.zeros(4, dtype=np.float64)


def image_pose(image):
    """Return (R, t) world-to-camera from a pycolmap Image.
    R is 3x3, t is (3,). Camera centre = -R^T @ t."""
    cfw = image.cam_from_world()
    R = np.array(cfw.rotation.matrix())
    t = np.array(cfw.translation)
    return R, t


# ---------------------------------------------------------------------------
# Prior Coverage
# ---------------------------------------------------------------------------

def generate_prior_coverage(rec, output_csv, class_name):
    """Write a VIAME-CSV file with one polygon per frame marking the prior-coverage region.

    For each registered image (filename-sorted), the 3D points observed by all
    *previous* frames are projected into the current camera.  The convex hull of
    those projections is stored as a flattened polygon on a DetectedObject.

    Args:
        rec:         pycolmap.Reconstruction after SfM.
        output_csv:  Path for the output CSV file.
        class_name:  Class label to attach to every detection (e.g. 'suppressed').
    """
    from kwiver.vital.modules import load_known_modules
    from kwiver.vital.algo import DetectedObjectSetOutput
    from kwiver.vital.types import (
        DetectedObject, DetectedObjectSet, DetectedObjectType, BoundingBoxD,
    )

    load_known_modules()

    # Build filename -> pycolmap image map
    images = rec.images
    cameras = rec.cameras
    name_to_img = {img.name: img for img in images.values()}

    # Iterate registered images in filename-sorted order
    sorted_names = sorted(name_to_img.keys())

    # Cache point IDs observed by each image
    def observed_pids(img):
        pids = set()
        for p2d in img.points2D:
            if p2d.has_point3D():
                pids.add(p2d.point3D_id)
        return pids

    writer = DetectedObjectSetOutput.create("viame_csv")
    writer.open(output_csv)

    prior_pids = set()

    for idx, fname in enumerate(sorted_names):
        img = name_to_img[fname]

        if idx == 0:
            # First frame: seed prior set, write empty detection set
            prior_pids = observed_pids(img)
            writer.write_set(DetectedObjectSet(), fname)
            continue

        # Project prior 3D points into this camera
        cam = cameras[img.camera_id]
        K = get_camera_matrix(cam)
        dist = get_dist_coeffs(cam)
        R, t = image_pose(img)
        w = cam.width
        h = cam.height

        # Gather world coordinates for prior point IDs that still exist
        pts_3d = []
        for pid in prior_pids:
            if pid in rec.points3D:
                pts_3d.append(rec.points3D[pid].xyz)

        if len(pts_3d) < 3:
            writer.write_set(DetectedObjectSet(), fname)
            prior_pids |= observed_pids(img)
            continue

        pts_3d = np.array(pts_3d, dtype=np.float64)

        # Filter to points in front of this camera (positive Z in camera frame)
        pts_cam = (R @ pts_3d.T + t.reshape(3, 1)).T
        in_front = pts_cam[:, 2] > 0
        pts_3d = pts_3d[in_front]

        if len(pts_3d) < 3:
            writer.write_set(DetectedObjectSet(), fname)
            prior_pids |= observed_pids(img)
            continue

        # Project with distortion via cv2.projectPoints
        rvec, _ = cv2.Rodrigues(R)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist)
        pts_2d = pts_2d.reshape(-1, 2)

        # Filter to within image bounds
        margin = 0
        inside = (
            (pts_2d[:, 0] >= -margin) & (pts_2d[:, 0] < w + margin) &
            (pts_2d[:, 1] >= -margin) & (pts_2d[:, 1] < h + margin)
        )
        pts_2d = pts_2d[inside]

        if len(pts_2d) < 3:
            writer.write_set(DetectedObjectSet(), fname)
            prior_pids |= observed_pids(img)
            continue

        # Convex hull
        hull = cv2.convexHull(pts_2d.astype(np.float32))
        hull_pts = hull.reshape(-1, 2)

        # Clamp hull vertices to image bounds
        hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0, w - 1)
        hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0, h - 1)

        # Bounding box from hull
        x1 = float(hull_pts[:, 0].min())
        y1 = float(hull_pts[:, 1].min())
        x2 = float(hull_pts[:, 0].max())
        y2 = float(hull_pts[:, 1].max())

        det = DetectedObject(
            BoundingBoxD(x1, y1, x2, y2),
            1.0,
            DetectedObjectType(class_name, 1.0),
        )

        # Flatten hull as [x1,y1, x2,y2, ...]
        flat_poly = []
        for px, py in hull_pts:
            flat_poly.append(float(px))
            flat_poly.append(float(py))
        det.set_flattened_polygon(flat_poly)

        det_set = DetectedObjectSet()
        det_set.add(det)
        writer.write_set(det_set, fname)

        # Accumulate this frame's point IDs
        prior_pids |= observed_pids(img)

    writer.complete()
    print(f"  Prior-coverage CSV ({len(sorted_names)} frames) -> {output_csv}")


# ---------------------------------------------------------------------------
# Flight-log / image metadata loading (GPS geo-anchoring)
# ---------------------------------------------------------------------------
#
# Two metadata formats are supported, both associated with images by CAPTURE
# ORDER (filenames are often renamed after capture, so order is the robust key):
#
#   A) FMCLOG CSV (e.g. 2024 SSL-FMCLOG_*.csv): one row per camera trigger with
#      NMEA lat/lon, yaw, AGL elevation_m, site_name + frame_count. Pass via
#      --flight-log <csv>; rows are filtered to the folder's site_name.
#   B) imagelog.json (e.g. 2025 UAS): {"ImageLog": [ {lat, lon, alt_rel, yaw,
#      pitch, roll, trigger_index, ...}, ... ]} co-located with the images.
#      Auto-detected in the image folder. Decimal lat/lon.
#   C) EXIF GPS embedded in the images (fallback when neither file is present;
#      gives lat/lon/alt but no heading).
#
# All are normalised to a per-frame pose dict: {lat, lon (decimal deg),
# alt_agl (m, may be None), yaw (deg, may be None)}.


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


def classify_images_fast(image_folder, image_list, scale=0.5, threshold=500):
    """Classify images using EfficientNet V2 S + SVM background classifiers.

    Uses the VIAME sea lion background classifier pipeline:
    EfficientNet V2 S -> SVM classifiers for 5 categories:
    all_land, coastal, cloudy, open_water, seaweed_water.

    Falls back to SIFT keypoint heuristic if models unavailable.

    Returns dict of filename -> {'is_water': bool, 'label': str,
                                  'scores': dict, 'keypoints': int}.
    """
    # Try to load the real classifier
    viame_install = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', '..', 'build', 'install')
    model_path = os.path.join(viame_install, 'configs', 'pipelines', 'models',
                               'pytorch_efficientnet_v2_s.pth')
    svm_dir = os.path.join(viame_install, 'configs', 'pipelines', 'models',
                            'sea_lion_v3_bg_classifiers')
    site_packages = os.path.join(viame_install, 'lib', 'python3.10', 'site-packages')

    if os.path.exists(model_path) and os.path.exists(svm_dir):
        try:
            return _classify_with_bg_classifier(
                image_folder, image_list, model_path, svm_dir, site_packages)
        except Exception as e:
            print(f"    Warning: background classifier failed ({e}), "
                  f"falling back to SIFT heuristic")

    # Fallback: SIFT keypoint heuristic
    return _classify_sift_heuristic(image_folder, image_list, scale, threshold)


def _classify_with_bg_classifier(image_folder, image_list, model_path, svm_dir,
                                   site_packages):
    """Classify images using EfficientNet + SVM background classifiers."""
    import torch
    from torchvision import models, transforms
    from PIL import Image as pilImage
    from ctypes import c_double

    # Add site-packages for libsvm
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    from svm import libsvm, toPyModel, gen_svm_nodearray

    # Load EfficientNet V2 S
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_v2_s()
    weights = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # Hook to get avgpool features
    features_out = {}
    def hook_fn(module, inp, out):
        features_out['feats'] = out.detach()
    model.avgpool.register_forward_hook(hook_fn)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load SVM models using low-level API (avoids bytes/str issues)
    svm_models = {}
    for f in sorted(os.listdir(svm_dir)):
        if f.endswith('.svm'):
            label = f.replace('.svm', '')
            fpath = os.path.join(svm_dir, f).encode()
            m = libsvm.svm_load_model(fpath)
            if m:
                svm_models[label] = toPyModel(m)

    svm_labels = sorted(svm_models.keys())
    print(f"    Loaded {len(svm_labels)} SVM classifiers: {', '.join(svm_labels)}")

    water_labels = {'open_water', 'seaweed_water', 'cloudy'}

    results = {}
    with torch.no_grad():
        for i, fname in enumerate(image_list):
            img_path = os.path.join(image_folder, fname)
            try:
                pil_img = pilImage.open(img_path).convert('RGB')
            except Exception:
                results[fname] = {'is_water': True, 'label': 'unknown',
                                   'scores': {}, 'keypoints': 0}
                continue

            # Extract features (full frame)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            model(tensor)
            feat = features_out['feats'].cpu().squeeze().flatten().tolist()

            # Build sparse feature dict for libsvm
            feat_dict = {idx: val for idx, val in enumerate(feat)}
            xi, max_idx = gen_svm_nodearray(feat_dict)

            # Run each SVM with probability prediction
            scores = {}
            for label in svm_labels:
                svm_model = svm_models[label]
                prob_est = (c_double * 2)()
                libsvm.svm_predict_probability(svm_model, xi, prob_est)
                # Get probability for the positive class (+1)
                model_labels = svm_model.get_labels()
                pos_idx = 0 if model_labels[0] == 1 else 1
                scores[label] = float(prob_est[pos_idx])

            # Best label = highest score
            best_label = max(scores, key=scores.get) if scores else 'unknown'
            is_water = best_label in water_labels

            results[fname] = {
                'is_water': is_water,
                'label': best_label,
                'scores': scores,
                'keypoints': -1,
            }

            if (i + 1) % 20 == 0 or (i + 1) == len(image_list):
                print(f"    Classified {i + 1}/{len(image_list)} images...")

    return results


def _classify_sift_heuristic(image_folder, image_list, scale=0.5, threshold=500):
    """Fallback: classify using SIFT keypoint count."""
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04)
    results = {}
    for fname in image_list:
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            results[fname] = {'is_water': True, 'label': 'unknown',
                               'scores': {}, 'keypoints': 0}
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp = sift.detect(gray, None)
        n_kp = len(kp) if kp else 0
        is_water = n_kp < threshold
        results[fname] = {
            'is_water': is_water,
            'label': 'open_water' if is_water else 'all_land',
            'scores': {},
            'keypoints': n_kp,
        }
    return results


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


def _fit_similarity(src, dst):
    """Least-squares similarity (rotation+uniform scale+reflection+translation)
    mapping src[N,2] -> dst[N,2]. Returns a function p->mapped and the RMS residual."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, _, Vt = np.linalg.svd(S.T @ D)
    R = (Vt.T @ U.T)
    denom = float(np.sum(S * S))
    scale = float(np.sum(D * (S @ R.T)) / denom) if denom > 1e-9 else 1.0
    def fn(p):
        return (np.asarray(p, float) - mu_s) @ R.T * scale + mu_d
    rms = float(np.sqrt(np.mean(np.sum((fn(src) - dst) ** 2, 1)))) if len(src) else 0.0
    return fn, rms


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


def _rot2(deg):
    """2x2 rotation matrix for an angle in degrees (NaN -> identity)."""
    if deg is None or np.isnan(deg):
        return np.eye(2)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


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


def run_planar_coverage(image_folder, output_dir, coverage_file="prior_coverage.csv",
                         class_name="suppressed", visualize=False, multicam=False,
                         match_ratio=0.75, min_inliers=10, min_inlier_ratio=0.10,
                         ransac_thresh=5.0, validate_homography=False,
                         multi_scale=False, search_window=20, max_chain_cond=0,
                         scale=0.5, cross_cam_trials=5, classifier='auto',
                         interpolate=True, consistency_filter=False,
                         adaptive_scale=True, matcher='bf',
                         homography_method='ransac', always_clahe=False,
                         root_sift=False, use_affine=False,
                         sift_contrast=0.04, clahe_clip=4.0,
                         loop_closure=False, loop_closure_max=24,
                         xcam_robust=False, xcam_cluster_tol=300.0,
                         xcam_low_drift=False, anchor_central=False,
                         flight_log=None, geo_anchor=False):
    """Compute coverage/suppression regions using homographies (planar scene).

    For multicam: computes homographies within each camera separately (temporal
    chain), plus cross-camera homographies, then combines all coverage.

    For single camera: just chains sequential homographies.
    """
    import re

    if multicam:
        # Group images by camera
        cam_dirs = {}
        for d in os.listdir(image_folder):
            full = os.path.join(image_folder, d)
            if os.path.isdir(full) and d.upper() in ('CENTER', 'PORT', 'STAR'):
                cam_dirs[d.upper()] = d

        cameras = {}
        for cam_key in ('CENTER', 'PORT', 'STAR'):
            if cam_key not in cam_dirs:
                continue
            cam_dir = cam_dirs[cam_key]
            cam_path = os.path.join(image_folder, cam_dir)
            imgs = sorted(f for f in os.listdir(cam_path)
                          if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                          and os.path.isfile(os.path.join(cam_path, f)))
            cameras[cam_key] = [os.path.join(cam_dir, f) for f in imgs]

        n_per_cam = {k: len(v) for k, v in cameras.items()}
        n_total = sum(n_per_cam.values())
        print(f"\n  Computing planar coverage for {n_total} images "
              f"({', '.join(f'{k}={v}' for k, v in n_per_cam.items())})")

        # Pre-classify images as water/benthic
        cam_order = ('CENTER', 'PORT', 'STAR')
        print("  Classifying images (water vs benthic)...")
        all_imgs_flat = []
        for cam_imgs in cameras.values():
            all_imgs_flat.extend(cam_imgs)
        water_info = classify_images_fast(image_folder, all_imgs_flat)
        for cam_key in cam_order:
            if cam_key not in cameras:
                continue
            cam_imgs = cameras[cam_key]
            n_water = sum(1 for f in cam_imgs if water_info.get(f, {}).get('is_water', False))
            print(f"    {cam_key}: {n_water}/{len(cam_imgs)} water frames")

        # Adaptive per-camera scale: cameras dominated by water (>50%) get a
        # higher matching scale and relaxed inlier ratio, since water frames
        # have sparse, low-contrast features. Land-heavy cameras keep the base
        # scale which works best for them.
        cam_scale = {}
        cam_inlier_ratio = {}
        for cam_key in cam_order:
            if cam_key not in cameras:
                continue
            cam_imgs = cameras[cam_key]
            n_water = sum(1 for f in cam_imgs
                          if water_info.get(f, {}).get('is_water', False))
            frac = n_water / max(len(cam_imgs), 1)
            if adaptive_scale and frac > 0.5:
                cam_scale[cam_key] = min(scale * 1.5, 1.0)
                cam_inlier_ratio[cam_key] = min_inlier_ratio * 0.5
            else:
                cam_scale[cam_key] = scale
                cam_inlier_ratio[cam_key] = min_inlier_ratio

        # Step 1: Compute within-camera sequential homography chains
        print("  Computing within-camera homography chains...")
        cam_chains = {}
        cam_pairwise = {}
        for cam_key, cam_imgs in cameras.items():
            chain, pw_H = _compute_camera_chain(
                image_folder, cam_imgs, label=cam_key, water_info=water_info,
                match_ratio=match_ratio, min_inliers=min_inliers,
                min_inlier_ratio=cam_inlier_ratio.get(cam_key, min_inlier_ratio),
                ransac_thresh=ransac_thresh,
                validate_homography=validate_homography,
                multi_scale=multi_scale, search_window=search_window,
                max_chain_cond=max_chain_cond,
                scale=cam_scale.get(cam_key, scale),
                consistency_filter=consistency_filter,
                matcher=matcher, homography_method=homography_method,
                always_clahe=always_clahe, root_sift=root_sift,
                use_affine=use_affine, sift_contrast=sift_contrast,
                clahe_clip=clahe_clip, loop_closure=loop_closure,
                loop_closure_max=loop_closure_max, anchor_central=anchor_central)
            cam_chains[cam_key] = chain
            cam_pairwise[cam_key] = pw_H

        # Step 1b: GPS geo-anchoring (optional). Use flight-log / imagelog / EXIF
        # poses to fill unregistered (water) frames with drift-free positions and
        # report feature-chain drift vs GPS.
        if geo_anchor:
            site_name = re.sub(r'^\d{8}_', '', os.path.basename(image_folder.rstrip('/'))).replace('_', ' ')
            poses_by_cam = {}
            for cam_key, cam_imgs in cameras.items():
                poses_by_cam[cam_key] = load_pose_metadata(
                    image_folder, cam_imgs, flight_log=flight_log, site_name=site_name)
            if any(poses_by_cam.values()):
                _geo_anchor_cameras(cam_chains, cameras, poses_by_cam, cam_pairwise)
            else:
                print("    (geo-anchor: no metadata found for this folder)")

        # Step 2: Compute cross-camera homographies (PORT->CENTER, STAR->CENTER).
        # For a fixed camera rig the cross-camera H is nearly constant, so we
        # compute it from MANY frame pairs and robust-average with MAD outlier
        # rejection. The spread of these per-frame estimates is itself a strong
        # quality signal (low spread = good loop closure).
        print("  Computing cross-camera homographies...")
        cross_cam_H = {}  # cam_key -> H mapping cam to CENTER
        cross_cam_H['CENTER'] = np.eye(3, dtype=np.float64)
        cross_cam_spread = {}  # cam_key -> (mad_tx, mad_ty, n_inliers, n_raw)

        center_imgs = cameras.get('CENTER', [])
        center_chain = cam_chains.get('CENTER', {})

        def _chain_anchor(chain):
            """The anchor is the frame whose chain H is the identity."""
            for k, H in chain.items():
                if np.allclose(H, np.eye(3), atol=1e-9):
                    return k
            return None

        center_anchor = _chain_anchor(center_chain)

        for cam_key in ('PORT', 'STAR'):
            if cam_key not in cameras:
                continue
            cam_imgs = cameras[cam_key]
            cam_chain = cam_chains[cam_key]
            cam_anchor = _chain_anchor(cam_chain)

            # Collect candidate frame indices where BOTH cameras registered.
            candidate_pairs = []
            for i in range(min(len(cam_imgs), len(center_imgs))):
                if i in cam_chain and i in center_chain:
                    cam_info = (water_info or {}).get(cam_imgs[i], {})
                    ctr_info = (water_info or {}).get(center_imgs[i], {})
                    is_land = (cam_info.get('label', '') == 'all_land' and
                               ctr_info.get('label', '') == 'all_land')
                    is_water_cam = cam_info.get('is_water', False)
                    is_water_ctr = ctr_info.get('is_water', False)
                    if is_water_cam and is_water_ctr:
                        continue  # skip water-water (unreliable)
                    score = 2 if is_land else (1 if not is_water_cam and not is_water_ctr else 0)
                    # Accumulated chain drift at frame i ~ distance from both anchors.
                    # Low-drift frames give cross-cam estimates that actually agree.
                    drift = 0
                    if cam_anchor is not None and center_anchor is not None:
                        drift = abs(i - cam_anchor) + abs(i - center_anchor)
                    candidate_pairs.append((score, drift, i))

            if xcam_low_drift:
                # Prefer land frames, then LOWEST drift (closest to both anchors).
                candidate_pairs.sort(key=lambda x: (-x[0], x[1]))
            else:
                candidate_pairs.sort(key=lambda x: -x[0])
            candidate_pairs = [(s, i) for s, _d, i in candidate_pairs]
            max_trials = max(cross_cam_trials * 3, 15)
            candidate_pairs = candidate_pairs[:max_trials]

            raw_H_list = []
            for _, try_idx in candidate_pairs:
                center_try = try_idx
                H_raw, _ = compute_homography_pair(
                    os.path.join(image_folder, cam_imgs[try_idx]),
                    os.path.join(image_folder, center_imgs[center_try]),
                    scale=min(scale * 1.5, 1.0), nfeatures=12000,
                    match_ratio=match_ratio, min_inliers=min_inliers,
                    ransac_thresh=ransac_thresh,
                    matcher=matcher, homography_method=homography_method,
                    always_clahe=always_clahe, root_sift=root_sift,
                    use_affine=use_affine, sift_contrast=sift_contrast,
                    clahe_clip=clahe_clip)
                if H_raw is not None:
                    cam_chain_i = cam_chain.get(try_idx)
                    center_chain_i = center_chain.get(center_try)
                    if cam_chain_i is not None and center_chain_i is not None:
                        try:
                            cam_chain_i_inv = np.linalg.inv(cam_chain_i)
                            H_norm = center_chain_i @ H_raw @ cam_chain_i_inv
                            raw_H_list.append((try_idx, H_norm))
                        except np.linalg.LinAlgError:
                            pass

            if raw_H_list:
                tx_vals = np.array([H[0, 2] for _, H in raw_H_list])
                ty_vals = np.array([H[1, 2] for _, H in raw_H_list])

                if xcam_robust and len(raw_H_list) >= 2:
                    # Mode-seeking consensus: chain drift over water makes MOST
                    # normalized cross-cam estimates disagree, so a median (and a
                    # MAD tolerance scaled by that median) accepts garbage. Instead
                    # pick the estimate whose translation has the MOST neighbours
                    # within a FIXED tolerance, and average that dominant cluster.
                    # Robust to >50% outliers.
                    pts = np.stack([tx_vals, ty_vals], axis=1)
                    fixed_tol = max(xcam_cluster_tol, 0.0) or 300.0
                    best_members = None
                    best_cnt = -1
                    for a in range(len(pts)):
                        d = np.linalg.norm(pts - pts[a], axis=1)
                        members = np.where(d <= fixed_tol)[0]
                        if len(members) > best_cnt:
                            best_cnt = len(members)
                            best_members = members
                    inlier_H = [raw_H_list[m][1] for m in best_members]
                    cl_tx = tx_vals[best_members]
                    cl_ty = ty_vals[best_members]
                    mad_tx = float(np.median(np.abs(cl_tx - np.median(cl_tx))) * 1.4826)
                    mad_ty = float(np.median(np.abs(cl_ty - np.median(cl_ty))) * 1.4826)
                    if inlier_H:
                        H_stack = np.stack(inlier_H)
                        cross_cam_H[cam_key] = np.median(H_stack, axis=0)
                        cross_cam_spread[cam_key] = (
                            mad_tx, mad_ty, len(inlier_H), len(raw_H_list))
                        print(f"    {cam_key}->CENTER: OK [robust cluster] "
                              f"({len(inlier_H)}/{len(raw_H_list)} in dominant "
                              f"cluster, MAD tx={mad_tx:.0f}px ty={mad_ty:.0f}px)")
                    else:
                        cross_cam_H[cam_key] = raw_H_list[0][1]
                        cross_cam_spread[cam_key] = (mad_tx, mad_ty, 1, len(raw_H_list))
                        print(f"    {cam_key}->CENTER: OK (1 frame, no cluster)")
                    continue

                med_tx = np.median(tx_vals)
                med_ty = np.median(ty_vals)
                mad_tx = np.median(np.abs(tx_vals - med_tx)) * 1.4826
                mad_ty = np.median(np.abs(ty_vals - med_ty)) * 1.4826
                tol_tx = max(mad_tx * 3.0, abs(med_tx) * 0.1, 50)
                tol_ty = max(mad_ty * 3.0, abs(med_ty) * 0.1, 50)

                inlier_H = []
                for idx, H in raw_H_list:
                    if (abs(H[0, 2] - med_tx) <= tol_tx and
                            abs(H[1, 2] - med_ty) <= tol_ty):
                        inlier_H.append(H)

                if inlier_H:
                    H_stack = np.stack(inlier_H)
                    cross_cam_H[cam_key] = np.median(H_stack, axis=0)
                    cross_cam_spread[cam_key] = (
                        float(mad_tx), float(mad_ty), len(inlier_H), len(raw_H_list))
                    print(f"    {cam_key}->CENTER: OK ({len(inlier_H)}/{len(raw_H_list)} "
                          f"inliers from {len(candidate_pairs)} trials, "
                          f"MAD tx={mad_tx:.0f}px ty={mad_ty:.0f}px)")
                else:
                    cross_cam_H[cam_key] = raw_H_list[0][1]
                    cross_cam_spread[cam_key] = (
                        float(mad_tx), float(mad_ty), 1, len(raw_H_list))
                    print(f"    {cam_key}->CENTER: OK (1 frame, no consensus)")
            else:
                print(f"    {cam_key}->CENTER: FAILED (will use {cam_key} only)")

        # Step 3: Build unified image list with H_to_global for each image
        # Global reference = CENTER camera, frame 0
        # Order: by timestep, CENTER first (so PORT/STAR map to CENTER)
        all_images = []  # list of (filename, H_to_global, timestep, cam_key, cam_order)

        for cam_key, cam_imgs in cameras.items():
            chain = cam_chains[cam_key]
            H_cam_to_center = cross_cam_H.get(cam_key)

            for i, fname in enumerate(cam_imgs):
                if i not in chain:
                    continue
                H_to_cam0 = chain[i]
                if H_cam_to_center is not None:
                    H_to_global = H_cam_to_center @ H_to_cam0
                else:
                    H_to_global = H_to_cam0

                # Extract sequence number for ordering
                match = re.search(r'(\d+)\.\w+$', fname)
                seq = int(match.group(1)) if match else i

                # CENTER first so it gets priority as "first seen"
                cam_order_val = {'CENTER': 0, 'PORT': 1, 'STAR': 2}.get(cam_key, 9)
                all_images.append((fname, H_to_global, seq, cam_key, cam_order_val))

        # Sort by timestep, then camera order (CENTER first)
        all_images.sort(key=lambda x: (x[2], x[4]))

    else:
        # Single camera mode
        image_names = get_image_files(image_folder)
        n_total = len(image_names)
        if n_total < 2:
            print(f"ERROR: Need at least 2 images, found {n_total}")
            return False
        print(f"\n  Computing planar coverage for {n_total} images...")
        print("  Computing frame-to-frame homographies...")
        chain, _pw = _compute_camera_chain(
            image_folder, image_names, label="frames",
            match_ratio=match_ratio, min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio, ransac_thresh=ransac_thresh,
            validate_homography=validate_homography, multi_scale=multi_scale,
            search_window=search_window, max_chain_cond=max_chain_cond,
            scale=scale, consistency_filter=consistency_filter,
            matcher=matcher, homography_method=homography_method,
            always_clahe=always_clahe, root_sift=root_sift,
            use_affine=use_affine, sift_contrast=sift_contrast,
            clahe_clip=clahe_clip, loop_closure=loop_closure,
            loop_closure_max=loop_closure_max, anchor_central=anchor_central)

        # GPS geo-anchoring (optional) — single camera (e.g. 2025 UAS imagelog).
        if geo_anchor:
            site_name = re.sub(r'^\d{8}_', '', os.path.basename(image_folder.rstrip('/'))).replace('_', ' ')
            poses = load_pose_metadata(image_folder, image_names,
                                       flight_log=flight_log, site_name=site_name)
            if poses:
                M, n, r, enu, yaw = _geo_calibrate(chain, image_names, poses, _pw)
                _geo_fill(chain, image_names, enu, yaw, M, label="frames",
                          n_steps=n, residual=r)
            else:
                print("    (geo-anchor: no metadata found for this folder)")

        all_images = []
        for i, fname in enumerate(image_names):
            if i in chain:
                all_images.append((fname, chain[i], i, "SINGLE", 0))

    if len(all_images) < 2:
        print("ERROR: Too few images with valid homographies")
        return False

    print(f"  {len(all_images)} images with valid homographies")

    # Step 4: Generate coverage CSV
    # For each image, project previous images' footprints (separate temporal vs cross-cam)
    output_csv = os.path.join(output_dir, coverage_file)
    det_id = 0
    entries = []

    # Separate lookups for visualization
    temporal_poly_lookup = {}   # purple: suppression from previous timesteps
    crosscam_poly_lookup = {}   # blue: suppression from same timestep, different camera

    # Cache image dimensions
    dim_cache = {}

    def _get_dims(fname):
        if fname not in dim_cache:
            img_tmp = cv2.imread(os.path.join(image_folder, fname))
            if img_tmp is None:
                return None
            dim_cache[fname] = (img_tmp.shape[0], img_tmp.shape[1])
        return dim_cache[fname]

    def _project_hull(sources, H_i_inv, wi, hi):
        """Project source frames' footprints into frame i's space, return hull."""
        all_pts = []
        for fname_j, H_j in sources:
            dims = _get_dims(fname_j)
            if dims is None:
                continue
            hj, wj = dims
            corners_j = np.float32(
                [[0, 0], [wj, 0], [wj, hj], [0, hj]]).reshape(-1, 1, 2)
            H_j_to_i = H_i_inv @ H_j
            try:
                projected = cv2.perspectiveTransform(corners_j, H_j_to_i)
                pts = projected.reshape(-1, 2)
                if np.all(np.abs(pts) < max(wi, hi) * 3):
                    all_pts.append(pts)
            except Exception:
                continue
        if not all_pts:
            return None
        combined = np.vstack(all_pts)
        combined[:, 0] = np.clip(combined[:, 0], 0, wi - 1)
        combined[:, 1] = np.clip(combined[:, 1], 0, hi - 1)
        if len(combined) < 3:
            return None
        try:
            hull = cv2.convexHull(combined.astype(np.float32))
            hull_pts = hull.reshape(-1, 2)
            return hull_pts if len(hull_pts) >= 3 else None
        except Exception:
            return None

    # Pre-index: for each timestep, collect all images at that timestep so
    # cross-camera suppression can be computed bidirectionally.
    timestep_images = {}  # seq -> list of (fname, H, cam_key)
    for fname_t, H_t, seq_t, cam_t, _ in all_images:
        timestep_images.setdefault(seq_t, []).append((fname_t, H_t, cam_t))

    with open(output_csv, 'w') as f:
        f.write("# 1: Detection or Track-id,  2: Video or Image Identifier,  "
                "3: Unique Frame Identifier,  4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y),  "
                "8: Detection or Length Confidence,  9: Target Length,  "
                "10-11+: Repeated Species, Confidence Pairs or Attributes\n")

        for idx in range(1, len(all_images)):
            fname_i, H_i, seq_i, cam_i, _ = all_images[idx]

            dims = _get_dims(fname_i)
            if dims is None:
                continue
            hi, wi = dims

            try:
                H_i_inv = np.linalg.inv(H_i)
            except np.linalg.LinAlgError:
                continue

            # Temporal sources: all images from PREVIOUS timesteps
            temporal_sources = []
            for j in range(idx):
                fname_j, H_j, seq_j, cam_j, _ = all_images[j]
                if seq_j < seq_i:
                    temporal_sources.append((fname_j, H_j))

            # Cross-camera sources: ALL images at the SAME timestep from a
            # DIFFERENT camera (bidirectional — look both forward and backward).
            crosscam_sources = []
            for fname_j, H_j, cam_j in timestep_images.get(seq_i, []):
                if cam_j != cam_i:
                    crosscam_sources.append((fname_j, H_j))

            # Compute separate hulls
            temporal_hull = _project_hull(temporal_sources, H_i_inv, wi, hi)
            crosscam_hull = _project_hull(crosscam_sources, H_i_inv, wi, hi)

            # Combined hull for the CSV (all suppression)
            all_sources = temporal_sources + crosscam_sources
            combined_hull = _project_hull(all_sources, H_i_inv, wi, hi)

            if temporal_hull is not None:
                temporal_poly_lookup[fname_i] = [(px, py) for px, py in temporal_hull]
            if crosscam_hull is not None:
                crosscam_poly_lookup[fname_i] = [(px, py) for px, py in crosscam_hull]

            if combined_hull is not None:
                hull_pts = combined_hull
                x1 = float(hull_pts[:, 0].min())
                y1 = float(hull_pts[:, 1].min())
                x2 = float(hull_pts[:, 0].max())
                y2 = float(hull_pts[:, 1].max())

                poly_str = " ".join(f"{px:.1f} {py:.1f}" for px, py in hull_pts)
                f.write(f"{det_id},{fname_i},{idx},"
                        f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},"
                        f"1.0,-1,{class_name},1.0,(poly) {poly_str}\n")

                entries.append({
                    'filename': fname_i,
                    'bbox': (x1, y1, x2, y2),
                    'polygon': [(px, py) for px, py in hull_pts],
                })
                det_id += 1

    print(f"  Prior-coverage CSV ({len(all_images)} frames, "
          f"{det_id} detections) -> {output_csv}")

    # Step 5: Motion interpolation for unchained frames
    # For frames that failed registration, estimate suppression from average motion
    # between the previous and next successful registrations in the same camera
    interp_poly_lookup = {}  # red: estimated suppression via motion interpolation
    if multicam and interpolate:
        for cam_key, cam_imgs in cameras.items():
            chain = cam_chains[cam_key]
            chained_indices = sorted(chain.keys())
            if len(chained_indices) < 2:
                continue

            for i, fname in enumerate(cam_imgs):
                if i in chain:
                    continue  # already registered

                # Find bracketing chained frames
                prev_idx = None
                next_idx = None
                for ci in chained_indices:
                    if ci < i:
                        prev_idx = ci
                    elif ci > i and next_idx is None:
                        next_idx = ci

                if prev_idx is None or next_idx is None:
                    continue

                # Interpolate H by blending prev and next
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                H_prev = chain[prev_idx]
                H_next = chain[next_idx]
                # Linear interpolation of the homography matrices
                H_interp = (1 - alpha) * H_prev + alpha * H_next

                H_cam_to_center = cross_cam_H.get(cam_key)
                if H_cam_to_center is not None:
                    H_global = H_cam_to_center @ H_interp
                else:
                    H_global = H_interp

                # Get this frame's dimensions
                dims = _get_dims(fname)
                if dims is None:
                    continue
                hi, wi = dims

                try:
                    H_inv = np.linalg.inv(H_global)
                except np.linalg.LinAlgError:
                    continue

                # Project all earlier chained images into this frame
                all_sources = []
                for item in all_images:
                    fname_j, H_j, seq_j, cam_j, _ = item
                    match_i = re.search(r'(\d+)\.\w+$', fname)
                    seq_i_val = int(match_i.group(1)) if match_i else i
                    if seq_j < seq_i_val or (seq_j == seq_i_val and cam_j != cam_key):
                        all_sources.append((fname_j, H_j))

                hull = _project_hull(all_sources, H_inv, wi, hi)
                if hull is not None:
                    interp_poly_lookup[fname] = [(px, py) for px, py in hull]

        if interp_poly_lookup:
            print(f"  Motion interpolation: {len(interp_poly_lookup)} unchained frames estimated")

    if visualize:
        vis_path = os.path.join(output_dir,
                                 os.path.splitext(coverage_file)[0] + "_vis.png")

        if multicam:
            _visualize_multicam_grid(
                cameras, temporal_poly_lookup, crosscam_poly_lookup,
                interp_poly_lookup, image_base_dir=image_folder,
                output_path=vis_path, all_images=all_images,
                water_info=water_info)
        elif entries:
            poly_lookup = {}
            for e in entries:
                poly_lookup[e['filename']] = e['polygon']
            _visualize_entries(entries, image_folder, vis_path)

    # Geometry-based quality evaluation (NOT a raw "valid count" — that metric
    # is misleading because false matches over water pass inlier thresholds yet
    # produce geometrically wrong registrations). We instead measure whether the
    # surviving registrations are self-consistent: determinant sanity, condition
    # number, temporal smoothness of motion, and cross-camera rigidity.
    if multicam:
        try:
            _evaluate_registration_quality(
                cam_chains, cross_cam_spread, cameras, water_info, output_dir)
        except Exception as e:
            print(f"  (quality eval skipped: {e})")

    return True


def _evaluate_registration_quality(cam_chains, cross_cam_spread, cameras,
                                    water_info, output_dir):
    """Score registration QUALITY from geometric self-consistency.

    A registration that merely passed RANSAC inlier thresholds can still be
    geometrically wrong (common over water, where repetitive texture yields
    confident-but-false matches). This function ignores raw counts and instead
    asks: are the surviving homographies internally consistent?

    Per-camera signals (all in [0,1], higher = better):
        det_sanity   : fraction of chain Hs with determinant in [0.5, 2.0]
        well_cond    : fraction of chain Hs with condition number < 1e4
        smoothness   : 1/(1+normalized motion jerk) — penalizes per-frame
                       translation that jumps around (sign of bad registration)
    Cross-camera signal:
        rigidity     : 1/(1+normalized MAD of the cam->CENTER translation) —
                       a fixed rig should give a near-constant cross-cam H, so
                       high spread means bad loop closure.

    Writes quality.json and prints a summary. The composite is deliberately
    conservative: it rewards FEWER but TRUSTWORTHY registrations over many
    garbage ones.
    """
    import json

    report = {'cameras': {}, 'cross_camera': {}}
    cam_geom_scores = []

    for cam_key, chain in cam_chains.items():
        idxs = sorted(k for k in chain.keys())
        n_reg = len(idxs)
        n_total = len(cameras.get(cam_key, []))

        dets = []
        conds = []
        for i in idxs:
            H = chain[i]
            try:
                d = abs(float(np.linalg.det(H)))
                dets.append(d)
                conds.append(float(np.linalg.cond(H)))
            except np.linalg.LinAlgError:
                conds.append(1e12)
        det_sanity = (sum(1 for d in dets if 0.5 <= d <= 2.0) / len(dets)
                      if dets else 0.0)
        well_cond = (sum(1 for c in conds if c < 1e4) / len(conds)
                     if conds else 0.0)

        # Temporal smoothness: per-frame translation steps between consecutive
        # registered frames, normalized; jerk = variability of those steps.
        steps = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            gap = b - a
            if gap <= 0:
                continue
            tx = (chain[b][0, 2] - chain[a][0, 2]) / gap
            ty = (chain[b][1, 2] - chain[a][1, 2]) / gap
            steps.append((tx, ty))
        if len(steps) >= 2:
            steps_arr = np.array(steps)
            med_mag = np.median(np.linalg.norm(steps_arr, axis=1)) + 1e-6
            diffs = np.linalg.norm(np.diff(steps_arr, axis=0), axis=1)
            norm_jerk = float(np.median(diffs) / med_mag)
            smoothness = 1.0 / (1.0 + norm_jerk)
        else:
            norm_jerk = None
            smoothness = 0.0

        geom = float(np.mean([det_sanity, well_cond, smoothness]))
        cam_geom_scores.append(geom)
        report['cameras'][cam_key] = {
            'n_registered': n_reg,
            'n_total': n_total,
            'det_sanity': round(det_sanity, 3),
            'well_conditioned': round(well_cond, 3),
            'norm_motion_jerk': round(norm_jerk, 3) if norm_jerk is not None else None,
            'smoothness': round(smoothness, 3),
            'geom_quality': round(geom, 3),
        }

    rigidity_scores = []
    for cam_key, spread in (cross_cam_spread or {}).items():
        mad_tx, mad_ty, n_inl, n_raw = spread
        mad_mag = (mad_tx ** 2 + mad_ty ** 2) ** 0.5
        # Normalize by a typical frame size scale (~5000px); rigidity in [0,1].
        rigidity = 1.0 / (1.0 + mad_mag / 500.0)
        inlier_frac = n_inl / max(n_raw, 1)
        rigidity_scores.append(rigidity)
        report['cross_camera'][cam_key] = {
            'mad_tx_px': round(mad_tx, 1),
            'mad_ty_px': round(mad_ty, 1),
            'inlier_pairs': n_inl,
            'raw_pairs': n_raw,
            'inlier_frac': round(inlier_frac, 3),
            'rigidity': round(rigidity, 3),
        }

    mean_geom = float(np.mean(cam_geom_scores)) if cam_geom_scores else 0.0
    mean_rigidity = float(np.mean(rigidity_scores)) if rigidity_scores else 0.0
    # Composite: geometry of chains (70%) + cross-camera rigidity (30%).
    composite = round(100.0 * (0.7 * mean_geom + 0.3 * mean_rigidity), 1)
    report['summary'] = {
        'mean_geom_quality': round(mean_geom, 3),
        'mean_rigidity': round(mean_rigidity, 3),
        'composite_quality': composite,
    }

    out_path = os.path.join(output_dir, 'quality.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("  --- Registration QUALITY (geometric self-consistency) ---")
    for cam_key, c in report['cameras'].items():
        print(f"    {cam_key}: reg={c['n_registered']}/{c['n_total']} "
              f"det_sane={c['det_sanity']} cond_ok={c['well_conditioned']} "
              f"smooth={c['smoothness']} -> geom={c['geom_quality']}")
    for cam_key, c in report['cross_camera'].items():
        print(f"    {cam_key}->CENTER rigidity={c['rigidity']} "
              f"(MAD {c['mad_tx_px']},{c['mad_ty_px']}px, "
              f"{c['inlier_pairs']}/{c['raw_pairs']} pairs)")
    print(f"    COMPOSITE QUALITY = {composite}/100 "
          f"(geom {mean_geom:.2f}, rigidity {mean_rigidity:.2f})")
    return report


def _visualize_multicam_grid(cameras, temporal_poly, crosscam_poly,
                              interp_poly, image_base_dir, output_path,
                              all_images, water_info=None):
    """Create a grid visualization: columns = STAR|CENTER|PORT, rows = timesteps.

    Color coding:
        Purple: suppression from previous timesteps (temporal)
        Blue:   suppression from same-timestep neighboring cameras (cross-cam)
        Red:    estimated suppression via motion interpolation (unchained frames)

    Classification label shown on EVERY image (water/benthic badge).

    Args:
        cameras:        dict of cam_key -> list of relative image paths
        temporal_poly:  dict fname -> polygon for temporal suppression
        crosscam_poly:  dict fname -> polygon for cross-camera suppression
        interp_poly:    dict fname -> polygon for interpolated suppression
        image_base_dir: base directory for images
        output_path:    output figure path
        all_images:     list of (fname, H, seq, cam_key, cam_order) tuples
        water_info:     dict of filename -> {'is_water': bool, 'keypoints': int}
    """
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    # Column order: STAR (left), CENTER (center), PORT (right)
    cam_order = ['STAR', 'CENTER', 'PORT']
    cam_colors = {'CENTER': '#2196F3', 'PORT': '#4CAF50', 'STAR': '#FF9800'}

    # Build timestep -> {cam -> filename} mapping from ALL source images
    timesteps = {}
    for cam_key in cam_order:
        if cam_key not in cameras:
            continue
        for fname in cameras[cam_key]:
            match = re.search(r'(\d+)\.\w+$', fname)
            seq = int(match.group(1)) if match else 0
            timesteps.setdefault(seq, {})[cam_key] = fname

    sorted_seqs = sorted(timesteps.keys())
    nrows = len(sorted_seqs)
    ncols = len(cam_order)

    if nrows == 0:
        return

    cell_w, cell_h = 4.5, 3.0
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(cell_w * ncols, cell_h * nrows),
                              facecolor='white')
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    chained_files = {item[0] for item in all_images}

    for row_idx, seq in enumerate(sorted_seqs):
        for col_idx, cam_key in enumerate(cam_order):
            ax = axes[row_idx, col_idx]
            fname = timesteps[seq].get(cam_key)

            if fname is None:
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=10, color='#999999', transform=ax.transAxes)
                ax.axis('off')
                continue

            img_path = os.path.join(image_base_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, 'Unreadable', ha='center', va='center',
                        fontsize=8, color='#999999', transform=ax.transAxes)
                ax.axis('off')
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            display_w = 900
            scale = display_w / w if w > display_w else 1.0
            if scale < 1.0:
                img_rgb = cv2.resize(img_rgb, (display_w, int(h * scale)))

            ax.imshow(img_rgb)

            # Classification badge on EVERY image (from background classifier)
            info = (water_info or {}).get(fname, {})
            cls_label = info.get('label', 'unknown')
            is_water = info.get('is_water', False)
            # Color by label category
            label_colors = {
                'all_land': '#2E7D32',       # green
                'coastal': '#FF8F00',         # amber
                'cloudy': '#78909C',          # blue-grey
                'open_water': '#1565C0',      # blue
                'seaweed_water': '#00838F',   # teal
                'unknown': '#999999',         # grey
            }
            cls_color = label_colors.get(cls_label, '#999999')
            # Show confidence if available
            scores = info.get('scores', {})
            conf = scores.get(cls_label, 0)
            disp_label = cls_label.replace('_', ' ')
            if conf > 0:
                disp_label += f" {conf:.0%}"
            ax.text(0.98, 0.98, disp_label, ha='right', va='top',
                    fontsize=6, color='white', fontweight='bold',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor=cls_color, alpha=0.8,
                              edgecolor='none'))

            # Overlay suppression polygons by source type
            total_supp_area = 0

            # 1. Temporal suppression (purple)
            t_poly_raw = temporal_poly.get(fname)
            if t_poly_raw:
                t_pts = [(px * scale, py * scale) for px, py in t_poly_raw]
                poly = MplPolygon(t_pts, closed=True, alpha=0.30,
                                  facecolor='#7B1FA2', edgecolor='#4A148C',
                                  linewidth=1.5)
                ax.add_patch(poly)
                total_supp_area += cv2.contourArea(
                    np.array(t_poly_raw, dtype=np.float32).reshape(-1, 1, 2))

            # 2. Cross-camera suppression (blue)
            c_poly_raw = crosscam_poly.get(fname)
            if c_poly_raw:
                c_pts = [(px * scale, py * scale) for px, py in c_poly_raw]
                poly = MplPolygon(c_pts, closed=True, alpha=0.30,
                                  facecolor='#1565C0', edgecolor='#0D47A1',
                                  linewidth=1.5)
                ax.add_patch(poly)
                total_supp_area += cv2.contourArea(
                    np.array(c_poly_raw, dtype=np.float32).reshape(-1, 1, 2))

            # 3. Interpolated suppression (red) - for unchained frames
            i_poly_raw = interp_poly.get(fname)
            if i_poly_raw:
                i_pts = [(px * scale, py * scale) for px, py in i_poly_raw]
                poly = MplPolygon(i_pts, closed=True, alpha=0.25,
                                  facecolor='#E53935', edgecolor='#B71C1C',
                                  linewidth=1.0, linestyle='--')
                ax.add_patch(poly)
                total_supp_area += cv2.contourArea(
                    np.array(i_poly_raw, dtype=np.float32).reshape(-1, 1, 2))

            # Coverage percentage label
            if total_supp_area > 0:
                pct = min(total_supp_area / (h * w) * 100, 100)
                ax.text(3 * scale, 18 * scale, f"{pct:.0f}%",
                        color='white', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='#333333', alpha=0.85,
                                  edgecolor='none'))
            elif fname not in chained_files and not i_poly_raw:
                # Unchained and no interpolation
                ax.text(0.5, 0.5, 'no registration', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold',
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#666666', alpha=0.7))

            ax.axis('off')

            if col_idx == 0:
                ax.set_ylabel(f"#{seq}", fontsize=8, rotation=0,
                              labelpad=30, va='center', fontweight='bold')

    # Column headers with stats
    for col_idx, cam_key in enumerate(cam_order):
        color = cam_colors.get(cam_key, '#333333')
        n_temporal = n_crosscam = n_interp = n_new = n_water = n_noreg = 0
        for seq in sorted_seqs:
            fname = timesteps[seq].get(cam_key)
            if fname is None:
                continue
            has_any = (temporal_poly.get(fname) or crosscam_poly.get(fname)
                       or interp_poly.get(fname))
            if has_any:
                if temporal_poly.get(fname):
                    n_temporal += 1
                if crosscam_poly.get(fname):
                    n_crosscam += 1
                if interp_poly.get(fname):
                    n_interp += 1
            elif fname not in chained_files:
                if (water_info or {}).get(fname, {}).get('is_water', False):
                    n_water += 1
                else:
                    n_noreg += 1
            else:
                n_new += 1
        parts = [f"{cam_key}"]
        stat_items = []
        if n_temporal:
            stat_items.append(f"{n_temporal} temp")
        if n_crosscam:
            stat_items.append(f"{n_crosscam} xcam")
        if n_interp:
            stat_items.append(f"{n_interp} interp")
        if n_new:
            stat_items.append(f"{n_new} new")
        if n_water:
            stat_items.append(f"{n_water} water")
        if stat_items:
            parts.append(f"({' / '.join(stat_items)})")
        axes[0, col_idx].set_title(' '.join(parts), fontsize=9,
                                    fontweight='bold', color=color, pad=8)

    folder_name = os.path.basename(image_base_dir.rstrip('/'))
    total_frames = sum(1 for seq in sorted_seqs for cam in cam_order
                       if timesteps[seq].get(cam))
    n_any_supp = len(set(list(temporal_poly.keys()) +
                          list(crosscam_poly.keys()) +
                          list(interp_poly.keys())))
    fig.suptitle(f"Prior-Coverage (Suppression) Regions — {folder_name}\n"
                 f"{n_any_supp} of {total_frames} frames have prior coverage",
                 fontsize=13, fontweight='bold', y=1.01)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7B1FA2',
               markersize=12, alpha=0.5, label='Temporal (past frames)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1565C0',
               markersize=12, alpha=0.5, label='Cross-camera (same timestep)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E53935',
               markersize=12, alpha=0.5, label='Interpolated (est. motion)'),
    ]
    # Add classifier label legend
    label_colors = {
        'all_land': '#2E7D32', 'coastal': '#FF8F00', 'cloudy': '#78909C',
        'open_water': '#1565C0', 'seaweed_water': '#00838F',
    }
    for lbl, clr in label_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=clr,
                   markersize=10, alpha=0.6, label=lbl.replace('_', ' ')))
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=7, frameon=True, fancybox=True,
               edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0.03, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Visualization saved -> {output_path}")


def _visualize_entries(entries, image_base_dir, output_path,
                        max_frames=12, figsize_per_img=(6, 4)):
    """Create a report-ready figure showing suppression regions on sample images.

    Shows evenly spaced frames with red overlay for "already seen" regions,
    percentage labels, camera labels, and a clean layout.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    import matplotlib.patheffects as pe

    if not entries:
        return

    n = len(entries)
    if n <= max_frames:
        selected = list(range(n))
    else:
        selected = [int(i * n / max_frames) for i in range(max_frames)]

    ncols = min(4, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_img[0] * ncols,
                                       figsize_per_img[1] * nrows),
                              facecolor='white')
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    cam_colors = {
        'CENTER': '#2196F3', 'PORT': '#4CAF50', 'STAR': '#FF9800',
    }

    for plot_idx, entry_idx in enumerate(selected):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]
        entry = entries[entry_idx]
        fname = entry['filename']

        img_path = os.path.join(image_base_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            ax.set_title(f"Missing: {fname}", fontsize=8)
            ax.axis('off')
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        display_w = 1200
        scale = display_w / w if w > display_w else 1.0
        if scale < 1.0:
            img_rgb = cv2.resize(img_rgb, (display_w, int(h * scale)))

        ax.imshow(img_rgb)

        # Detect camera from path
        cam = "UNKNOWN"
        for c in ('CENTER', 'PORT', 'STAR'):
            if c in fname.upper().split(os.sep)[0] if os.sep in fname else "":
                cam = c
                break

        if entry['polygon']:
            poly_pts = [(px * scale, py * scale) for px, py in entry['polygon']]
            # Red suppression overlay
            poly = MplPolygon(poly_pts, closed=True, alpha=0.30,
                              facecolor='#E53935', edgecolor='#B71C1C',
                              linewidth=1.5, linestyle='-')
            ax.add_patch(poly)

            # Compute % coverage
            hull_area = cv2.contourArea(
                np.array(entry['polygon'], dtype=np.float32).reshape(-1, 1, 2))
            img_area = h * w
            pct = min(hull_area / img_area * 100, 100)

            # Coverage label with outline for readability
            txt = ax.text(5 * scale, 25 * scale, f"{pct:.0f}% suppressed",
                          color='white', fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='#E53935', alpha=0.85,
                                    edgecolor='#B71C1C', linewidth=0.5))

        # Camera + frame label
        short_name = os.path.basename(fname)
        cam_label = fname.split(os.sep)[0] if os.sep in fname else ""
        title = f"{cam_label}/{short_name}" if cam_label else short_name
        cam_color = cam_colors.get(cam, '#666666')
        ax.set_title(title, fontsize=7, color=cam_color, fontweight='bold')
        ax.axis('off')

    for plot_idx in range(len(selected), nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row, col].axis('off')

    # Title
    folder_name = os.path.basename(image_base_dir.rstrip('/'))
    fig.suptitle(f"Prior-Coverage (Suppression) Regions - {folder_name}\n"
                 f"Red overlay = area already seen in previous frames/cameras",
                 fontsize=13, fontweight='bold', y=1.02)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E53935',
               markersize=12, alpha=0.5, label='Suppressed (already seen)'),
    ]
    for cam, color in cam_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=8, label=f'{cam} camera'))
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=9, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Visualization saved -> {output_path}")


def generate_prior_coverage_standalone(rec, output_csv, class_name="suppressed"):
    """Write a CSV file with suppression polygons - no kwiver dependency.

    Output format (VIAME-CSV compatible):
      det_id, filename, frame_num, x1, y1, x2, y2, confidence, length, class conf, (poly) ...

    For each registered image (filename-sorted), projects 3D points observed
    by all previous frames into the current camera and writes the convex hull.
    """
    images = rec.images
    cameras = rec.cameras
    name_to_img = {img.name: img for img in images.values()}
    sorted_names = sorted(name_to_img.keys())

    def observed_pids(img):
        pids = set()
        for p2d in img.points2D:
            if p2d.has_point3D():
                pids.add(p2d.point3D_id)
        return pids

    prior_pids = set()
    det_id = 0

    with open(output_csv, 'w') as f:
        f.write("# 1: Detection or Track-id,  2: Video or Image Identifier,  "
                "3: Unique Frame Identifier,  4-7: Img-bbox(TL_x, TL_y, BR_x, BR_y),  "
                "8: Detection or Length Confidence,  9: Target Length,  "
                "10-11+: Repeated Species, Confidence Pairs or Attributes\n")

        for idx, fname in enumerate(sorted_names):
            img = name_to_img[fname]

            if idx == 0:
                prior_pids = observed_pids(img)
                continue

            cam = cameras[img.camera_id]
            K = get_camera_matrix(cam)
            dist = get_dist_coeffs(cam)
            R, t = image_pose(img)
            w = cam.width
            h = cam.height

            pts_3d = []
            for pid in prior_pids:
                if pid in rec.points3D:
                    pts_3d.append(rec.points3D[pid].xyz)

            if len(pts_3d) < 3:
                prior_pids |= observed_pids(img)
                continue

            pts_3d = np.array(pts_3d, dtype=np.float64)
            pts_cam = (R @ pts_3d.T + t.reshape(3, 1)).T
            in_front = pts_cam[:, 2] > 0
            pts_3d = pts_3d[in_front]

            if len(pts_3d) < 3:
                prior_pids |= observed_pids(img)
                continue

            rvec, _ = cv2.Rodrigues(R)
            pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist)
            pts_2d = pts_2d.reshape(-1, 2)

            inside = (
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
            )
            pts_2d = pts_2d[inside]

            if len(pts_2d) < 3:
                prior_pids |= observed_pids(img)
                continue

            hull = cv2.convexHull(pts_2d.astype(np.float32))
            hull_pts = hull.reshape(-1, 2)
            hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0, w - 1)
            hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0, h - 1)

            x1 = float(hull_pts[:, 0].min())
            y1 = float(hull_pts[:, 1].min())
            x2 = float(hull_pts[:, 0].max())
            y2 = float(hull_pts[:, 1].max())

            poly_str = " ".join(f"{px:.1f} {py:.1f}" for px, py in hull_pts)
            f.write(f"{det_id},{fname},{idx},"
                    f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},"
                    f"1.0,-1,{class_name},1.0,(poly) {poly_str}\n")
            det_id += 1
            prior_pids |= observed_pids(img)

    print(f"  Prior-coverage CSV ({len(sorted_names)} frames, {det_id} detections) -> {output_csv}")


def visualize_coverage(coverage_csv, image_base_dir, output_path,
                       max_frames=12, figsize_per_img=(6, 4)):
    """Create a figure showing suppression regions overlaid on sample images.

    Args:
        coverage_csv:   Path to the prior-coverage CSV file.
        image_base_dir: Base directory where images are located.
        output_path:    Output path for the figure (e.g. .png, .pdf).
        max_frames:     Max number of frames to show in the figure.
        figsize_per_img: (width, height) in inches per subplot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import re

    # Parse the CSV
    entries = []
    with open(coverage_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 10:
                continue

            fname = parts[1].strip()
            x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])

            # Extract polygon
            poly_pts = []
            rest = ','.join(parts[9:])
            poly_match = re.search(r'\(poly\)\s*(.*)', rest)
            if poly_match:
                coords = poly_match.group(1).strip().split()
                for i in range(0, len(coords) - 1, 2):
                    poly_pts.append((float(coords[i]), float(coords[i+1])))

            entries.append({
                'filename': fname,
                'bbox': (x1, y1, x2, y2),
                'polygon': poly_pts,
            })

    if not entries:
        print("  No coverage entries found in CSV.")
        return

    # Select evenly spaced frames
    n = len(entries)
    if n <= max_frames:
        selected = list(range(n))
    else:
        selected = [int(i * n / max_frames) for i in range(max_frames)]

    # Layout
    ncols = min(4, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_img[0] * ncols,
                                       figsize_per_img[1] * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for plot_idx, entry_idx in enumerate(selected):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]

        entry = entries[entry_idx]
        fname = entry['filename']

        img_path = os.path.join(image_base_dir, fname)
        if not os.path.exists(img_path):
            ax.set_title(f"Missing: {fname}", fontsize=8)
            ax.axis('off')
            continue

        img = cv2.imread(img_path)
        if img is None:
            ax.set_title(f"Unreadable: {fname}", fontsize=8)
            ax.axis('off')
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Downscale for display
        h, w = img_rgb.shape[:2]
        display_w = 1200
        if w > display_w:
            scale = display_w / w
            img_rgb = cv2.resize(img_rgb, (display_w, int(h * scale)))
        else:
            scale = 1.0

        ax.imshow(img_rgb)

        if entry['polygon']:
            poly_pts = [(px * scale, py * scale) for px, py in entry['polygon']]
            poly = MplPolygon(poly_pts, closed=True, alpha=0.35,
                              facecolor='red', edgecolor='red', linewidth=2)
            ax.add_patch(poly)

        short_name = os.path.basename(fname)
        ax.set_title(f"Frame {entry_idx+1}: {short_name}", fontsize=7)
        ax.axis('off')

    # Turn off unused axes
    for plot_idx in range(len(selected), nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row, col].axis('off')

    fig.suptitle("Prior-Coverage (Suppression) Regions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Visualization saved -> {output_path}")


# ---------------------------------------------------------------------------
# Stage 4-5: Dense feature-based reconstruction
# ---------------------------------------------------------------------------

def select_dense_pairs(rec, max_pairs_per_image=3):
    """Select image pairs for dense matching based on shared 3D point count."""
    images = rec.images
    image_ids = sorted(images.keys())

    point_to_images = {}
    for img_id in image_ids:
        img = images[img_id]
        for p2d in img.points2D:
            if p2d.has_point3D():
                pid = p2d.point3D_id
                point_to_images.setdefault(pid, set()).add(img_id)

    pair_scores = {}
    for pid, img_set in point_to_images.items():
        ids = sorted(img_set)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = (ids[i], ids[j])
                pair_scores[pair] = pair_scores.get(pair, 0) + 1

    img_pair_count = {iid: 0 for iid in image_ids}
    selected = []
    for pair, score in sorted(pair_scores.items(), key=lambda x: -x[1]):
        a, b = pair
        if img_pair_count[a] < max_pairs_per_image and img_pair_count[b] < max_pairs_per_image:
            if score >= 20:
                selected.append((a, b, score))
                img_pair_count[a] += 1
                img_pair_count[b] += 1

    return selected


def triangulate_matches(kp1, kp2, matches, K, R1, t1, R2, t2):
    """Triangulate matched keypoints into 3D points.
    Returns (points3d_Nx3, valid_mask_N)."""
    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

    # Projection matrices: P = K @ [R | t]
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    # Triangulate
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  # Nx3

    # Filter: points must be in front of both cameras
    pts_cam1 = (R1 @ pts3d.T + t1.reshape(3, 1)).T
    pts_cam2 = (R2 @ pts3d.T + t2.reshape(3, 1)).T
    valid = (pts_cam1[:, 2] > 0) & (pts_cam2[:, 2] > 0)

    # Filter: reprojection error
    proj1 = (P1 @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    err1 = np.linalg.norm(proj1 - pts1, axis=1)

    proj2 = (P2 @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    err2 = np.linalg.norm(proj2 - pts2, axis=1)

    valid &= (err1 < 5.0) & (err2 < 5.0)

    # Filter: triangulation angle (reject near-degenerate)
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    rays1 = pts3d - c1
    rays2 = pts3d - c2
    cos_angle = np.sum(rays1 * rays2, axis=1) / (
        np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1) + 1e-10
    )
    valid &= (cos_angle < 0.9998)  # at least ~1 degree

    return pts3d, valid


def get_pixel_colors(image_bgr, keypoints, match_indices, scale_from_full):
    """Sample colors from image at keypoint locations."""
    colors = []
    for idx in match_indices:
        x, y = keypoints[idx].pt
        # Scale back to full image coordinates
        x_full = x / scale_from_full
        y_full = y / scale_from_full
        xi, yi = int(round(x_full)), int(round(y_full))
        h, w = image_bgr.shape[:2]
        xi = max(0, min(xi, w - 1))
        yi = max(0, min(yi, h - 1))
        bgr = image_bgr[yi, xi]
        colors.append([bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0])  # RGB
    return np.array(colors)


def run_dense(rec, image_folder, output_dir, scale=0.25, max_pairs_per_image=3):
    """Dense reconstruction via feature matching + triangulation."""
    with timeit("Dense feature matching + triangulation"):
        pairs = select_dense_pairs(rec, max_pairs_per_image=max_pairs_per_image)
        if not pairs:
            print("  No suitable image pairs found for dense matching.")
            return None

        print(f"  Selected {len(pairs)} image pairs")

        all_pts = []
        all_cols = []
        images = rec.images
        cameras = rec.cameras

        # Create SIFT detector for dense matching (more features than SfM)
        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=15)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        for idx, (id_a, id_b, score) in enumerate(pairs):
            img_a = images[id_a]
            img_b = images[id_b]
            cam = cameras[img_a.camera_id]

            K_full = get_camera_matrix(cam)
            Ra, ta = image_pose(img_a)
            Rb, tb = image_pose(img_b)

            path_a = os.path.join(image_folder, img_a.name)
            path_b = os.path.join(image_folder, img_b.name)
            if not os.path.exists(path_a) or not os.path.exists(path_b):
                continue

            bgr_a = cv2.imread(path_a)
            bgr_b = cv2.imread(path_b)
            if bgr_a is None or bgr_b is None:
                continue

            # Downscale for feature extraction
            h0, w0 = bgr_a.shape[:2]
            h, w = int(h0 * scale), int(w0 * scale)
            small_a = cv2.resize(bgr_a, (w, h), interpolation=cv2.INTER_AREA)
            small_b = cv2.resize(bgr_b, (w, h), interpolation=cv2.INTER_AREA)

            gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)

            # Extract features
            kp_a, des_a = sift.detectAndCompute(gray_a, None)
            kp_b, des_b = sift.detectAndCompute(gray_b, None)

            if des_a is None or des_b is None or len(kp_a) < 100 or len(kp_b) < 100:
                continue

            # Match with ratio test
            raw_matches = bf.knnMatch(des_a, des_b, k=2)
            good_matches = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 50:
                print(f"    Pair {idx+1}/{len(pairs)}: {img_a.name} <-> {img_b.name} "
                      f"- too few matches ({len(good_matches)})")
                continue

            # Scale intrinsics to match downscaled images
            K = K_full.copy()
            K[0, :] *= (w / w0)
            K[1, :] *= (h / h0)

            # Fundamental matrix filtering
            pts1 = np.float64([kp_a[m.queryIdx].pt for m in good_matches])
            pts2 = np.float64([kp_b[m.trainIdx].pt for m in good_matches])
            F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                                     ransacReprojThreshold=2.0)
            if inlier_mask is None:
                continue
            inlier_matches = [m for m, keep in zip(good_matches, inlier_mask.ravel()) if keep]

            if len(inlier_matches) < 30:
                continue

            # Triangulate
            pts3d, valid = triangulate_matches(kp_a, kp_b, inlier_matches, K, Ra, ta, Rb, tb)
            valid_pts = pts3d[valid]

            if len(valid_pts) < 10:
                print(f"    Pair {idx+1}/{len(pairs)}: {img_a.name} <-> {img_b.name} "
                      f"- {len(inlier_matches)} inliers -> {len(valid_pts)} triangulated")
                continue

            # Outlier removal by distance from median
            median_pt = np.median(valid_pts, axis=0)
            dists = np.linalg.norm(valid_pts - median_pt, axis=1)
            dist_thresh = np.percentile(dists, 95) * 2.0
            keep = dists < dist_thresh
            valid_pts = valid_pts[keep]

            # Get colors from image A
            valid_match_indices = [inlier_matches[i].queryIdx
                                   for i, v in enumerate(valid) if v]
            valid_match_indices = [vi for vi, k in zip(valid_match_indices, keep) if k]
            # Sample colors at downscaled resolution
            cols = []
            for mi in valid_match_indices:
                x, y = kp_a[mi].pt
                xi, yi = int(round(x)), int(round(y))
                xi = max(0, min(xi, w - 1))
                yi = max(0, min(yi, h - 1))
                bgr_val = small_a[yi, xi]
                cols.append([bgr_val[2] / 255.0, bgr_val[1] / 255.0, bgr_val[0] / 255.0])
            cols = np.array(cols) if cols else np.zeros((0, 3))

            print(f"    Pair {idx+1}/{len(pairs)}: {img_a.name} <-> {img_b.name} "
                  f"({score} shared) -> {len(inlier_matches)} inliers -> "
                  f"{len(valid_pts)} dense points")

            if len(valid_pts) > 0 and len(cols) == len(valid_pts):
                all_pts.append(valid_pts)
                all_cols.append(cols)

        if not all_pts:
            print("  No dense points generated.")
            return None

        all_pts = np.vstack(all_pts)
        all_cols = np.vstack(all_cols)
        print(f"  Total dense points: {len(all_pts)}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_cols)

        # Statistical outlier removal
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"  After outlier removal: {len(pcd.points)} points")

        return pcd


# ---------------------------------------------------------------------------
# Stage 4-5 (alt): COLMAP PatchMatch MVS dense reconstruction
# ---------------------------------------------------------------------------

def check_colmap_cuda():
    """Check if the colmap CLI is installed and built with CUDA.
    Returns (available: bool, has_cuda: bool, message: str)."""
    import shutil
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        return False, False, "colmap binary not found in PATH"
    try:
        result = subprocess.run([colmap_bin, "-h"], capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
    except Exception as e:
        return False, False, f"Failed to run colmap: {e}"
    if "with CUDA" in output and "without CUDA" not in output:
        return True, True, f"{colmap_bin} (with CUDA)"
    elif "without CUDA" in output:
        return True, False, f"{colmap_bin} (without CUDA — patch_match_stereo requires CUDA)"
    # Ambiguous; assume no CUDA
    return True, False, f"{colmap_bin} (CUDA support unknown)"


def run_dense_mvs(rec, image_folder, output_dir):
    """Dense reconstruction via COLMAP PatchMatch MVS + stereo fusion.

    Requires the colmap CLI binary compiled with CUDA support.
    Uses the sparse reconstruction already written to output_dir/sparse/.

    Returns an Open3D PointCloud or None.
    """
    sparse_dir = os.path.join(output_dir, "sparse")
    mvs_dir = os.path.join(output_dir, "dense_mvs")
    fused_ply = os.path.join(mvs_dir, "fused.ply")

    # Step 1: Undistort images
    with timeit("COLMAP image undistortion"):
        ret = subprocess.run([
            "colmap", "image_undistorter",
            "--image_path", image_folder,
            "--input_path", sparse_dir,
            "--output_path", mvs_dir,
            "--output_type", "COLMAP",
        ], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"  ERROR: image_undistorter failed (exit {ret.returncode})")
            print(ret.stderr[-500:] if ret.stderr else "(no stderr)")
            return None
        print(ret.stdout[-300:] if ret.stdout else "  (no stdout)")

    # Step 2: PatchMatch stereo (GPU)
    with timeit("COLMAP PatchMatch stereo (GPU)"):
        ret = subprocess.run([
            "colmap", "patch_match_stereo",
            "--workspace_path", mvs_dir,
            "--PatchMatchStereo.geom_consistency", "true",
        ], capture_output=True, text=True, timeout=3600)
        if ret.returncode != 0:
            print(f"  ERROR: patch_match_stereo failed (exit {ret.returncode})")
            print(ret.stderr[-500:] if ret.stderr else "(no stderr)")
            return None
        print(ret.stdout[-300:] if ret.stdout else "  (no stdout)")

    # Step 3: Stereo fusion
    with timeit("COLMAP stereo fusion"):
        ret = subprocess.run([
            "colmap", "stereo_fusion",
            "--workspace_path", mvs_dir,
            "--output_path", fused_ply,
        ], capture_output=True, text=True, timeout=600)
        if ret.returncode != 0:
            print(f"  ERROR: stereo_fusion failed (exit {ret.returncode})")
            print(ret.stderr[-500:] if ret.stderr else "(no stderr)")
            return None
        print(ret.stdout[-300:] if ret.stdout else "  (no stdout)")

    if not os.path.exists(fused_ply):
        print("  ERROR: fused.ply was not created.")
        return None

    pcd = o3d.io.read_point_cloud(fused_ply)
    n = len(pcd.points)
    print(f"  MVS fused cloud: {n} points")

    if n == 0:
        return None

    # Statistical outlier removal
    if n > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  After outlier removal: {len(pcd.points)} points")

    return pcd


# ---------------------------------------------------------------------------
# Stage 6: Surface reconstruction
# ---------------------------------------------------------------------------

def build_mesh(pcd, output_mesh_path, depth=9):
    """Poisson surface reconstruction from a point cloud."""
    with timeit("Poisson surface reconstruction"):
        n_pts = len(pcd.points)
        print(f"  Input: {n_pts} points")

        # Estimate normals - radius based on point cloud extent
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        nn_radius = extent / 50.0
        print(f"  Scene extent: {extent:.2f}, normal search radius: {nn_radius:.4f}")

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=nn_radius, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        print(f"  Running Poisson reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False,
        )

        # Trim low-density regions
        densities = np.asarray(densities)
        if len(densities) > 0:
            thresh = np.quantile(densities, 0.02)
            vertices_to_remove = densities < thresh
            mesh.remove_vertices_by_mask(vertices_to_remove)

        # Clean up
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        nv = len(mesh.vertices)
        nf = len(mesh.triangles)
        print(f"  Mesh: {nv} vertices, {nf} triangles")

        if nv == 0 or nf == 0:
            print("  WARNING: empty mesh produced.")
            return None

        o3d.io.write_triangle_mesh(output_mesh_path, mesh)
        print(f"  Saved -> {output_mesh_path}")
        return mesh


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def reconstruction_to_pointcloud(rec):
    """Convert pycolmap Reconstruction sparse points to Open3D PointCloud."""
    pts, cols = [], []
    for pid, p3d in rec.points3D.items():
        pts.append(p3d.xyz)
        cols.append(p3d.color / 255.0)
    pcd = o3d.geometry.PointCloud()
    if pts:
        pcd.points = o3d.utility.Vector3dVector(np.array(pts))
        pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
    return pcd


def process_folder(image_folder, output_dir, scale=0.25, max_pairs_per_image=3,
                    coverage_class=None, coverage_file="prior_coverage.csv",
                    dense=True, dense_method="sift", multicam=False,
                    visualize=False, planar=False, planar_kwargs=None):
    """Full pipeline for one image folder.

    If multicam=True (or auto-detected), looks for PORT/STAR/CENTER subfolders
    and processes all cameras jointly in a single SfM reconstruction.

    If planar=True, uses homography-based coverage (skips SfM) - suitable
    for benthic/overhead surveys with mostly flat scenes.
    """
    folder_name = os.path.basename(image_folder.rstrip('/'))

    # Auto-detect multicam
    if not multicam and detect_multicam(image_folder):
        print(f"  Auto-detected multi-camera layout (PORT/STAR/CENTER)")
        multicam = True

    # Planar mode: skip SfM entirely, use homography-based coverage
    if planar_kwargs is None:
        planar_kwargs = {}
    if planar:
        os.makedirs(output_dir, exist_ok=True)
        cclass = coverage_class or "suppressed"
        return run_planar_coverage(
            image_folder, output_dir,
            coverage_file=coverage_file,
            class_name=cclass,
            visualize=visualize,
            multicam=multicam,
            **planar_kwargs,
        )

    print(f"\n{'#'*70}")
    print(f"  Processing: {folder_name} {'[MULTICAM]' if multicam else ''}")
    print(f"  Input:  {image_folder}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}")

    os.makedirs(output_dir, exist_ok=True)

    if multicam:
        image_names = get_multicam_image_files(image_folder)
        if len(image_names) < 2:
            print(f"ERROR: Need at least 2 images, found {len(image_names)}")
            return False
        # Count per camera
        cam_counts = {}
        for name in image_names:
            cam = name.split(os.sep)[0].upper()
            cam_counts[cam] = cam_counts.get(cam, 0) + 1
        print(f"Found {len(image_names)} images across cameras: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(cam_counts.items()))}")
    else:
        image_names = get_image_files(image_folder)
        if len(image_names) < 2:
            print(f"ERROR: Need at least 2 images, found {len(image_names)}")
            return False
        print(f"Found {len(image_names)} images")

    # ---- SfM ----
    rec = run_sfm(image_folder, output_dir, image_names, multicam=multicam)
    if rec is None:
        return False

    # Save sparse point cloud
    sparse_pcd = reconstruction_to_pointcloud(rec)
    sparse_ply = os.path.join(output_dir, "sparse_cloud.ply")
    o3d.io.write_point_cloud(sparse_ply, sparse_pcd)
    print(f"  Saved sparse cloud ({len(sparse_pcd.points)} pts) -> {sparse_ply}")

    # ---- Prior-coverage polygons (optional) ----
    coverage_csv = None
    if coverage_class is not None:
        coverage_csv = os.path.join(output_dir, coverage_file)
        try:
            generate_prior_coverage(rec, coverage_csv, coverage_class)
        except (ImportError, Exception) as e:
            print(f"  kwiver not available ({e}), using standalone CSV writer")
            generate_prior_coverage_standalone(rec, coverage_csv, coverage_class)

        # ---- Visualization (optional) ----
        if visualize:
            vis_path = os.path.join(output_dir,
                                     os.path.splitext(coverage_file)[0] + "_vis.png")
            visualize_coverage(coverage_csv, image_folder, vis_path)

    # ---- Dense feature matching + triangulation (steps 4-6, optional) ----
    dense_ply = None
    mesh_ply = None
    mesh_obj = None

    if dense:
        if dense_method == "mvs":
            dense_pcd = run_dense_mvs(rec, image_folder, output_dir)
        else:
            dense_pcd = run_dense(rec, image_folder, output_dir, scale=scale,
                                  max_pairs_per_image=max_pairs_per_image)

        # Merge sparse + dense
        if dense_pcd is not None and len(dense_pcd.points) > 0:
            combined = sparse_pcd + dense_pcd
            # Remove duplicates via voxel downsampling
            voxel_size = np.linalg.norm(
                combined.get_axis_aligned_bounding_box().get_max_bound() -
                combined.get_axis_aligned_bounding_box().get_min_bound()
            ) / 2000.0
            if voxel_size > 0:
                combined = combined.voxel_down_sample(voxel_size)
            print(f"  Combined cloud after dedup: {len(combined.points)} points")
        else:
            print("  Using sparse cloud only for meshing.")
            combined = sparse_pcd

        if len(combined.points) < 100:
            print("ERROR: Too few points for meshing.")
            return False

        dense_ply = os.path.join(output_dir, "dense_cloud.ply")
        o3d.io.write_point_cloud(dense_ply, combined)
        print(f"  Saved combined cloud ({len(combined.points)} pts) -> {dense_ply}")

        # ---- Mesh ----
        mesh_ply = os.path.join(output_dir, "mesh.ply")
        n_pts = len(combined.points)
        if n_pts < 5000:
            depth = 7
        elif n_pts < 50000:
            depth = 8
        elif n_pts < 500000:
            depth = 9
        else:
            depth = 10

        mesh = build_mesh(combined, mesh_ply, depth=depth)
        if mesh is None:
            return False

        mesh_obj = os.path.join(output_dir, "mesh.obj")
        o3d.io.write_triangle_mesh(mesh_obj, mesh)
        print(f"  Also saved -> {mesh_obj}")

    print(f"\n  SUCCESS: {folder_name}")
    print(f"    Sparse cloud: {sparse_ply}")
    if dense_ply:
        print(f"    Dense cloud:  {dense_ply}")
    if mesh_ply:
        print(f"    Mesh (PLY):   {mesh_ply}")
    if mesh_obj:
        print(f"    Mesh (OBJ):   {mesh_obj}")
    if coverage_csv:
        print(f"    Coverage CSV: {coverage_csv}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def view_file(filepath):
    """Open a PLY/OBJ file in the Open3D interactive viewer."""
    ext = os.path.splitext(filepath)[1].lower()
    name = os.path.basename(filepath)

    # Try loading as mesh first, fall back to point cloud
    if ext in ('.obj', '.stl', '.off', '.gltf', '.glb'):
        load_as = 'mesh'
    elif ext == '.ply':
        # PLY can be either mesh or point cloud - try mesh first
        mesh = o3d.io.read_triangle_mesh(filepath)
        if len(mesh.triangles) > 0:
            load_as = 'mesh'
        else:
            load_as = 'pointcloud'
    else:
        load_as = 'pointcloud'

    if load_as == 'mesh':
        geo = o3d.io.read_triangle_mesh(filepath)
        geo.compute_vertex_normals()
        label = (f"{name}  |  {len(geo.vertices)} vertices, "
                 f"{len(geo.triangles)} triangles")
    else:
        geo = o3d.io.read_point_cloud(filepath)
        label = f"{name}  |  {len(geo.points)} points"

    print(f"Viewing: {filepath}")
    print(f"  {label}")
    print("  Controls: left-drag=rotate, scroll=zoom, middle-drag=pan, "
          "R=reset, Q=close")
    o3d.visualization.draw_geometries([geo], window_name=label,
                                       width=1280, height=720)


def main():
    parser = argparse.ArgumentParser(description="UAS Imagery -> 3D Model")
    parser.add_argument("folder", nargs="?",
                        help="Path to a folder of images (or use --all with --base-dir)")
    parser.add_argument("--all", action="store_true",
                        help="Process every image subfolder under --base-dir")
    parser.add_argument("--base-dir", default=None,
                        help="Base directory containing image subfolders (for --all)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <input_dir>/3d_models/<folder>)")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Image downscale factor for dense matching (default 0.25)")
    parser.add_argument("--max-pairs", type=int, default=3,
                        help="Max dense-stereo pairs per image (default 3)")
    parser.add_argument("--view", "-v", default=None,
                        help="View a PLY/OBJ file instead of running reconstruction")
    parser.add_argument("--dense-method", choices=["sift", "mvs"], default="sift",
                        help="Dense reconstruction method: 'sift' for OpenCV SIFT "
                             "matching (CPU, default), 'mvs' for COLMAP PatchMatch "
                             "stereo (requires CUDA-enabled colmap binary)")
    parser.add_argument("--no-dense", action="store_true",
                        help="Skip dense matching, triangulation, and meshing "
                             "(steps 4-6); only run SfM and optional coverage")
    parser.add_argument("--coverage-class", default=None,
                        help="Generate prior-coverage polygons with this class name "
                             "(e.g. 'suppressed')")
    parser.add_argument("--coverage-file", default="prior_coverage.csv",
                        help="Output filename for prior-coverage CSV "
                             "(default: prior_coverage.csv)")
    parser.add_argument("--multicam", action="store_true",
                        help="Input folder has PORT/STAR/CENTER subfolders "
                             "(auto-detected if present)")
    parser.add_argument("--planar", action="store_true",
                        help="Use homography-based coverage computation for "
                             "planar scenes (benthic/overhead surveys). Skips "
                             "SfM and uses frame-to-frame homographies instead.")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of coverage polygons "
                             "overlaid on images")

    # --- Planar coverage quality options ---
    quality = parser.add_argument_group("Planar coverage quality options")
    quality.add_argument("--match-ratio", type=float, default=0.75,
                         help="Lowe's ratio test threshold (default: 0.75)")
    quality.add_argument("--min-inliers", type=int, default=10,
                         help="Minimum RANSAC inlier count (default: 10)")
    quality.add_argument("--min-inlier-ratio", type=float, default=0.10,
                         help="Minimum inlier/match ratio, 0=disabled (default: 0.10)")
    quality.add_argument("--ransac-thresh", type=float, default=5.0,
                         help="RANSAC reprojection threshold in pixels (default: 5.0)")
    quality.add_argument("--validate-homography", action="store_true",
                         help="Reject homographies with extreme distortion "
                              "(det<0.1 or >10, cond>1e6, clustered inliers)")
    quality.add_argument("--multi-scale", action="store_true",
                         help="Try higher resolution on matching failure")
    quality.add_argument("--search-window", type=int, default=20,
                         help="Max frames to search for chain matches (default: 20)")
    quality.add_argument("--max-chain-cond", type=float, default=0,
                         help="Max condition number for accumulated chain H, "
                              "0=disabled (default: 0)")
    quality.add_argument("--match-scale", type=float, default=0.5,
                         help="Base image scale for feature matching (default: 0.5)")
    quality.add_argument("--cross-cam-trials", type=int, default=5,
                         help="Frame pairs to try for cross-camera matching "
                              "(default: 5)")
    quality.add_argument("--no-adaptive-scale", action="store_true",
                         help="Disable adaptive per-camera scale (water-heavy "
                              "cameras otherwise get 1.5x scale + relaxed ratio)")
    quality.add_argument("--no-interpolate", action="store_true",
                         help="Disable motion interpolation for unchained frames")
    quality.add_argument("--consistency-filter", action="store_true",
                         help="Validate water-frame registrations against "
                              "land-frame motion; replace outliers with estimates")
    quality.add_argument("--matcher", choices=['bf', 'flann'], default='bf',
                         help="Feature matcher (default: bf)")
    quality.add_argument("--homography-method", choices=['ransac', 'lmeds', 'usac'],
                         default='ransac',
                         help="Homography estimation method (default: ransac)")
    quality.add_argument("--always-clahe", action="store_true",
                         help="Always apply CLAHE (default: only low-contrast)")
    quality.add_argument("--root-sift", action="store_true",
                         help="Apply Root-SIFT descriptor normalization")
    quality.add_argument("--affine", action="store_true",
                         help="Use affine model (6 DOF) instead of full "
                              "homography (8 DOF) — more constrained, rejects "
                              "garbage perspective warps over water")
    quality.add_argument("--sift-contrast", type=float, default=0.04,
                         help="SIFT contrast threshold (default: 0.04)")
    quality.add_argument("--clahe-clip", type=float, default=4.0,
                         help="CLAHE clip limit (default: 4.0)")
    quality.add_argument("--loop-closure", action="store_true",
                         help="Recover revisit frames by matching unchained "
                              "non-water frames against temporally-distant "
                              "chained frames across the whole sequence")
    quality.add_argument("--loop-closure-max", type=int, default=24,
                         help="Max candidate frames sampled across the sequence "
                              "for each loop-closure attempt (default: 24)")
    quality.add_argument("--xcam-robust", action="store_true",
                         help="Mode-seeking (cluster) consensus for the cross-"
                              "camera homography instead of median — robust when "
                              "chain drift over water corrupts >50%% of estimates")
    quality.add_argument("--xcam-cluster-tol", type=float, default=300.0,
                         help="Translation tolerance (px) for the cross-camera "
                              "consensus cluster (default: 300)")
    quality.add_argument("--xcam-low-drift", action="store_true",
                         help="Select cross-camera pairs at frames nearest BOTH "
                              "chain anchors (least accumulated drift) so the "
                              "estimates agree — pairs well with --xcam-robust")
    quality.add_argument("--anchor-central", action="store_true",
                         help="Anchor each camera's chain near the sequence "
                              "centre (strong-feature frame closest to the "
                              "middle) to halve worst-case accumulated drift")
    quality.add_argument("--geo-anchor", action="store_true",
                         help="Use flight-log / imagelog.json / EXIF GPS metadata "
                              "to fill unregistered (water) frames with drift-free "
                              "positions and report feature-chain drift vs GPS")
    quality.add_argument("--flight-log", default=None,
                         help="Path to an FMCLOG CSV (2024 format) for --geo-anchor. "
                              "Not needed when an imagelog.json or EXIF GPS is present.")

    parser.add_argument("--install-deps", action="store_true",
                        help="Check and install missing Python dependencies")
    parser.add_argument("--deps-target", default=None,
                        help="pip --target directory for dependency install "
                             "(default: auto-detect VIAME site-packages or user site)")
    args = parser.parse_args()

    # --- Dependency install mode (no imports needed) ---
    if args.install_deps:
        target = args.deps_target or _get_viame_site_packages()
        ensure_dependencies(install=True, target_dir=target)
        return

    # --- Import dependencies (after --install-deps check) ---
    import_dependencies()

    # --- View mode ---
    if args.view:
        view_file(os.path.abspath(args.view))
        return

    folders = []
    base_dir = args.base_dir or os.getcwd()
    if args.all:
        for name in sorted(os.listdir(base_dir)):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p) and name not in ('mosaics', 'mosaics_v2', '3d_models'):
                # Check for images directly or multicam subfolders
                imgs = get_image_files(p)
                if len(imgs) >= 2:
                    folders.append(p)
                elif detect_multicam(p):
                    folders.append(p)
    elif args.folder:
        folders.append(os.path.abspath(args.folder))
    else:
        parser.print_help()
        sys.exit(1)

    if not folders:
        print("No image folders found.")
        sys.exit(1)

    # Validate MVS prerequisites before starting any work
    if args.dense_method == "mvs" and not args.no_dense:
        avail, has_cuda, msg = check_colmap_cuda()
        if not avail:
            print(f"ERROR: --dense-method mvs requires colmap: {msg}")
            sys.exit(1)
        if not has_cuda:
            print(f"ERROR: --dense-method mvs requires CUDA-enabled colmap: {msg}")
            print("  Install a CUDA build of COLMAP (build from source with "
                  "-DCUDA_ENABLED=ON, or use conda-forge).")
            sys.exit(1)
        print(f"  COLMAP MVS: {msg}")

    t_total = time.time()
    results = {}
    planar_kwargs = dict(
        match_ratio=args.match_ratio,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        ransac_thresh=args.ransac_thresh,
        validate_homography=args.validate_homography,
        multi_scale=args.multi_scale,
        search_window=args.search_window,
        max_chain_cond=args.max_chain_cond,
        scale=args.match_scale,
        cross_cam_trials=args.cross_cam_trials,
        classifier=args.classifier if hasattr(args, 'classifier') else 'auto',
        interpolate=not args.no_interpolate,
        consistency_filter=args.consistency_filter,
        adaptive_scale=not args.no_adaptive_scale,
        matcher=args.matcher,
        homography_method=args.homography_method,
        always_clahe=args.always_clahe,
        root_sift=args.root_sift,
        use_affine=args.affine,
        sift_contrast=args.sift_contrast,
        clahe_clip=args.clahe_clip,
        loop_closure=args.loop_closure,
        loop_closure_max=args.loop_closure_max,
        xcam_robust=args.xcam_robust,
        xcam_cluster_tol=args.xcam_cluster_tol,
        xcam_low_drift=args.xcam_low_drift,
        anchor_central=args.anchor_central,
        flight_log=args.flight_log,
        geo_anchor=args.geo_anchor,
    )
    for folder in folders:
        name = os.path.basename(folder.rstrip('/'))
        out = args.output or os.path.join(base_dir, "3d_models", name)
        ok = process_folder(folder, out, scale=args.scale,
                            max_pairs_per_image=args.max_pairs,
                            coverage_class=args.coverage_class,
                            coverage_file=args.coverage_file,
                            dense=not args.no_dense,
                            dense_method=args.dense_method,
                            multicam=args.multicam,
                            visualize=args.visualize,
                            planar=args.planar,
                            planar_kwargs=planar_kwargs)
        results[name] = ok

    dt = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  All done in {dt:.0f}s")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"    {name}: {status}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
