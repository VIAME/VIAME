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

# Shared registration engine (sequential homography chain, GPS metadata,
# geo-anchoring) lives in the OpenCV plugin so other tools can reuse it.
from viame.opencv import registration_utils as _sr
from viame.opencv.registration_utils import (
    get_image_files, detect_multicam, load_pose_metadata,
    compute_homography_pair, _compute_homography_at_scale, _compute_camera_chain,
    _poses_to_enu, _track_headings, _rot2, _fit_similarity_disp,
    _geo_calibrate, _geo_fill, _geo_anchor_cameras,
    _nmea_to_dec, _dms_to_dec, _load_imagelog_json, _load_fmclog_csv, _load_exif_gps,
)


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

# Map of import name -> pip package name. numpy/cv2 are needed for everything;
# pycolmap/open3d are only needed for the COLMAP structure-from-motion and dense
# (MVS) modes, so they are OPTIONAL — the default planar-coverage / registration
# path runs without them (and without VIAME_ENABLE_COLMAP).
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'cv2': 'opencv-python',
}
OPTIONAL_PACKAGES = {
    'pycolmap': 'pycolmap',
    'open3d': 'open3d',
}


def check_dependencies(packages=None):
    """Check which packages are missing. Returns list of (import_name, pip_name)."""
    packages = packages if packages is not None else REQUIRED_PACKAGES
    missing = []
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return missing


def require_colmap():
    """Raise a clear error if pycolmap/open3d (COLMAP mode) are unavailable."""
    missing = check_dependencies(OPTIONAL_PACKAGES)
    if missing:
        print("ERROR: COLMAP 3D-reconstruction mode requires:")
        for import_name, pip_name in missing:
            print(f"  {pip_name} (import {import_name})")
        print("\nInstall them (or build VIAME with -DVIAME_ENABLE_COLMAP=ON), or "
              "use the default --planar mode which does not need COLMAP.")
        sys.exit(1)


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
    """Check deps and optionally install (full set incl. COLMAP). Returns True
    if all present."""
    all_packages = dict(REQUIRED_PACKAGES, **OPTIONAL_PACKAGES)
    missing = check_dependencies(all_packages)
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
    """Import packages. numpy/cv2 are hard requirements; pycolmap/open3d are
    optional (COLMAP mode only) and left as None if unavailable."""
    global np, cv2, pycolmap, o3d
    missing = check_dependencies()
    if missing:
        print("ERROR: Missing required dependencies:")
        for import_name, pip_name in missing:
            print(f"  {pip_name} (import {import_name})")
        print(f"\nRun: python {sys.argv[0]} --install-deps")
        sys.exit(1)
    import numpy as np_
    import cv2 as cv2_
    np = np_
    cv2 = cv2_
    _sr.import_dependencies()  # set engine globals
    try:
        import pycolmap as pycolmap_
        pycolmap = pycolmap_
    except ImportError:
        pycolmap = None
    try:
        import open3d as o3d_
        o3d = o3d_
    except ImportError:
        o3d = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}




# ---------------------------------------------------------------------------
# Multi-camera support
# ---------------------------------------------------------------------------



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




# ---------------------------------------------------------------------------
# Stage 1-3: Structure from Motion via pycolmap
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Prior Coverage
# ---------------------------------------------------------------------------



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
                         flight_log=None, geo_anchor=False,
                         loop_closure_correct=False):
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

        # Step 1a: Loop-closure correction (optional). Detect land revisits and
        # run a 2D pose-graph to redistribute accumulated chain drift so the
        # revisited land lines up across passes.
        if loop_closure_correct:
            print("  Loop-closure correction (pose graph)...")
            for cam_key, cam_imgs in cameras.items():
                chain = cam_chains[cam_key]
                if len(chain) < 4:
                    continue
                loops = _sr.detect_loop_edges(chain, image_folder, cam_imgs)
                if not loops:
                    print(f"    {cam_key}: no land loop closures found")
                    continue
                seq = [(i, j, H) for (i, j), H in cam_pairwise[cam_key].items()
                       if i in chain and j in chain]
                def _resid(poses):
                    # MEDIAN, not mean: the robust optimizer intentionally leaves
                    # rejected outlier edges violated, which would dominate a mean.
                    es = [np.linalg.norm((poses[j] @ H)[:2, 2] - poses[i][:2, 2])
                          for (i, j, H) in loops]
                    return float(np.median(es)) if es else 0.0
                before = _resid(chain)
                cam_chains[cam_key] = _sr.optimize_pose_graph(chain, seq, loops)
                after_poses = cam_chains[cam_key]
                after = _resid(after_poses)
                n_inl = sum(1 for (i, j, H) in loops
                            if np.linalg.norm((after_poses[j] @ H)[:2, 2]
                                              - after_poses[i][:2, 2]) <= 40)
                print(f"    {cam_key}: {len(loops)} loop edges ({n_inl} inliers), "
                      f"median residual {before:.0f}px -> {after:.0f}px")

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

        # Loop-closure correction (optional, single camera).
        if loop_closure_correct and len(chain) >= 4:
            loops = _sr.detect_loop_edges(chain, image_folder, image_names)
            if loops:
                seq = [(i, j, H) for (i, j), H in _pw.items()
                       if i in chain and j in chain]
                def _resid(poses):
                    es = [np.linalg.norm((poses[j] @ H)[:2, 2] - poses[i][:2, 2])
                          for (i, j, H) in loops]
                    return float(np.median(es)) if es else 0.0   # median (robust)
                before = _resid(chain)
                chain = _sr.optimize_pose_graph(chain, seq, loops)
                print(f"  Loop-closure: {len(loops)} edges, median residual "
                      f"{before:.0f}px -> {_resid(chain):.0f}px")
            else:
                print("  Loop-closure: no land revisits found")

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
    temporal_poly_lookup = {}   # purple: short-term overlap from recent prior timesteps
    loop_poly_lookup = {}       # gold:   loop-closure overlap from DISTANT prior frames (revisits)
    crosscam_poly_lookup = {}   # blue:   suppression from same timestep, different camera

    # A prior frame counts as a loop-closure revisit (not short-term drift) when
    # it is more than this many timesteps back yet its footprint still overlaps.
    # Matches detect_loop_edges' min_gap so the colour tracks the actual closures.
    LOOP_GAP = 15

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

    def _project_loop_hull(sources, H_i_inv, wi, hi, min_frac=0.02):
        """Like _project_hull, but for DISTANT (loop-closure) sources: only keep
        those whose projected footprint genuinely overlaps frame i, so far-away
        frames that merely project near the border don't paint spurious overlap."""
        frame_rect = np.float32([[0, 0], [wi, 0], [wi, hi], [0, hi]])
        frame_area = float(wi * hi)
        kept = []
        for fname_j, H_j in sources:
            dims = _get_dims(fname_j)
            if dims is None:
                continue
            hj, wj = dims
            corners_j = np.float32(
                [[0, 0], [wj, 0], [wj, hj], [0, hj]]).reshape(-1, 1, 2)
            H_j_to_i = H_i_inv @ H_j
            try:
                proj = cv2.perspectiveTransform(corners_j, H_j_to_i).reshape(-1, 2)
                inter, _ = cv2.intersectConvexConvex(
                    proj.astype(np.float32), frame_rect)
            except Exception:
                continue
            if inter and inter > min_frac * frame_area:
                p = proj.copy()
                p[:, 0] = np.clip(p[:, 0], 0, wi - 1)
                p[:, 1] = np.clip(p[:, 1], 0, hi - 1)
                kept.append(p)
        if not kept:
            return None
        combined = np.vstack(kept)
        if len(combined) < 3:
            return None
        try:
            hull = cv2.convexHull(combined.astype(np.float32)).reshape(-1, 2)
            return hull if len(hull) >= 3 else None
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

            # Temporal sources from PREVIOUS timesteps, split by how far back:
            #   short-term (recent drift overlap)  vs  loop-closure (distant revisit)
            shortterm_sources = []
            loop_sources = []
            for j in range(idx):
                fname_j, H_j, seq_j, cam_j, _ = all_images[j]
                if seq_j < seq_i:
                    if seq_i - seq_j <= LOOP_GAP:
                        shortterm_sources.append((fname_j, H_j))
                    else:
                        loop_sources.append((fname_j, H_j))

            # Cross-camera sources: ALL images at the SAME timestep from a
            # DIFFERENT camera (bidirectional — look both forward and backward).
            crosscam_sources = []
            for fname_j, H_j, cam_j in timestep_images.get(seq_i, []):
                if cam_j != cam_i:
                    crosscam_sources.append((fname_j, H_j))

            # Compute separate hulls
            temporal_hull = _project_hull(shortterm_sources, H_i_inv, wi, hi)
            loop_hull = _project_loop_hull(loop_sources, H_i_inv, wi, hi)
            crosscam_hull = _project_hull(crosscam_sources, H_i_inv, wi, hi)

            # Combined hull for the CSV (all suppression) — unchanged content
            all_sources = shortterm_sources + loop_sources + crosscam_sources
            combined_hull = _project_hull(all_sources, H_i_inv, wi, hi)

            if temporal_hull is not None:
                temporal_poly_lookup[fname_i] = [(px, py) for px, py in temporal_hull]
            if loop_hull is not None:
                loop_poly_lookup[fname_i] = [(px, py) for px, py in loop_hull]
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

        # Cache everything the grid render needs BEFORE rendering, so a transient
        # matplotlib/PIL failure (seen under concurrent load) doesn't throw away
        # the hours of chain computation — render_grid.py can rebuild the figure
        # from this pickle without recomputing.
        if multicam:
            try:
                import pickle as _pkl
                with open(os.path.join(output_dir, 'grid_data.pkl'), 'wb') as _gf:
                    _pkl.dump(dict(cameras=cameras,
                                   temporal_poly=temporal_poly_lookup,
                                   crosscam_poly=crosscam_poly_lookup,
                                   interp_poly=interp_poly_lookup,
                                   loop_poly=loop_poly_lookup,
                                   all_images=all_images, water_info=water_info,
                                   image_folder=image_folder, vis_path=vis_path), _gf)
            except Exception as _e:
                print(f"  (grid_data cache skipped: {_e})")

        try:
            if multicam:
                _visualize_multicam_grid(
                    cameras, temporal_poly_lookup, crosscam_poly_lookup,
                    interp_poly_lookup, image_base_dir=image_folder,
                    output_path=vis_path, all_images=all_images,
                    water_info=water_info, loop_poly=loop_poly_lookup)
            elif entries:
                poly_lookup = {}
                for e in entries:
                    poly_lookup[e['filename']] = e['polygon']
                _visualize_entries(entries, image_folder, vis_path)
        except Exception as _e:
            print(f"  WARNING: grid render failed ({_e}); computation preserved. "
                  f"Re-render with render_grid.py on grid_data.pkl")

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
                              all_images, water_info=None, loop_poly=None):
    """Create a grid visualization: columns = STAR|CENTER|PORT, rows = timesteps.

    Color coding (the final per-frame transformations, by overlap source):
        Purple: short-term temporal overlap from recent prior frames (last slice)
        Gold:   loop-closure overlap from DISTANT prior frames (a revisit / closed loop)
        Blue:   spatial overlap from same-timestep neighboring cameras (cross-cam)
        Red:    estimated overlap via motion interpolation (unchained frames)

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

    # Cap displayed rows so dense surveys (hundreds of frames) stay legible and
    # render quickly. Prioritise timesteps that actually carry loop-closure
    # overlap (the revisits this view is meant to show), then fill in evenly.
    MAX_ROWS = 70
    if len(sorted_seqs) > MAX_ROWS:
        lp = loop_poly or {}
        loop_seqs = [s for s in sorted_seqs
                     if any(f in lp for f in timesteps[s].values())]
        keep = set(loop_seqs[:MAX_ROWS // 2])
        remaining = [s for s in sorted_seqs if s not in keep]
        if remaining and len(keep) < MAX_ROWS:
            step = max(1, len(remaining) // (MAX_ROWS - len(keep)))
            keep |= set(remaining[::step])
        sorted_seqs = sorted(keep)[:MAX_ROWS]
        print(f"    (grid: showing {len(sorted_seqs)} representative timesteps "
              f"of {len(timesteps)}; {len(loop_seqs)} have loop-closure overlap)")
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

            # 1. Short-term temporal overlap (purple)
            t_poly_raw = temporal_poly.get(fname)
            if t_poly_raw:
                t_pts = [(px * scale, py * scale) for px, py in t_poly_raw]
                poly = MplPolygon(t_pts, closed=True, alpha=0.30,
                                  facecolor='#7B1FA2', edgecolor='#4A148C',
                                  linewidth=1.5)
                ax.add_patch(poly)
                total_supp_area += cv2.contourArea(
                    np.array(t_poly_raw, dtype=np.float32).reshape(-1, 1, 2))

            # 1b. Loop-closure overlap from distant revisited frames (gold)
            l_poly_raw = (loop_poly or {}).get(fname)
            if l_poly_raw:
                l_pts = [(px * scale, py * scale) for px, py in l_poly_raw]
                poly = MplPolygon(l_pts, closed=True, alpha=0.32,
                                  facecolor='#FFC400', edgecolor='#FF6F00',
                                  linewidth=2.0)
                ax.add_patch(poly)

            # 2. Cross-camera spatial overlap (blue)
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
               markersize=12, alpha=0.5, label='Temporal — short-term (recent frames)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFC400',
               markersize=12, alpha=0.6, label='Loop-closure (distant revisit)'),
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




# ---------------------------------------------------------------------------
# Stage 6: Surface reconstruction
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------



def process_folder(image_folder, output_dir, scale=0.25, max_pairs_per_image=3,
                    coverage_class=None, coverage_file="prior_coverage.csv",
                    dense=True, dense_method="sift", multicam=False,
                    visualize=False, planar=False, planar_kwargs=None,
                    sfm_matching='auto'):
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

    # SfM / dense (MVS) modes need COLMAP; planar mode above does not. The SfM
    # implementation lives in the (optional) viame.colmap plugin.
    require_colmap()
    from viame.colmap import reconstruction as _cr
    _cr.import_dependencies()

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
    rec = _cr.run_sfm(image_folder, output_dir, image_names, multicam=multicam,
                      matching=sfm_matching)
    if rec is None:
        return False

    # Save sparse point cloud
    sparse_pcd = _cr.reconstruction_to_pointcloud(rec)
    sparse_ply = os.path.join(output_dir, "sparse_cloud.ply")
    o3d.io.write_point_cloud(sparse_ply, sparse_pcd)
    print(f"  Saved sparse cloud ({len(sparse_pcd.points)} pts) -> {sparse_ply}")

    # ---- Prior-coverage polygons (optional) ----
    coverage_csv = None
    if coverage_class is not None:
        coverage_csv = os.path.join(output_dir, coverage_file)
        try:
            _cr.generate_prior_coverage(rec, coverage_csv, coverage_class)
        except (ImportError, Exception) as e:
            print(f"  kwiver not available ({e}), using standalone CSV writer")
            _cr.generate_prior_coverage_standalone(rec, coverage_csv, coverage_class)

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
            dense_pcd = _cr.run_dense_mvs(rec, image_folder, output_dir)
        else:
            dense_pcd = _cr.run_dense(rec, image_folder, output_dir, scale=scale,
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

        mesh = _cr.build_mesh(combined, mesh_ply, depth=depth)
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
    parser.add_argument("--sfm-matching", choices=['auto', 'exhaustive', 'sequential'],
                        default='auto',
                        help="SfM feature-matching strategy (non-planar mode). "
                             "'exhaustive' finds temporally-distant loop-closure "
                             "pairs (revisits); 'auto' uses it for small/multicam "
                             "sets only.")
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
    quality.add_argument("--loop-closure-correct", action="store_true",
                         help="Detect land revisits and run a 2D pose-graph to "
                              "redistribute chain drift so revisited land lines up "
                              "across passes (the loop-closure correction).")

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

    # --- View mode (point-cloud viewer; needs COLMAP plugin / open3d) ---
    if args.view:
        require_colmap()
        from viame.colmap import reconstruction as _cr
        _cr.import_dependencies()
        _cr.view_file(os.path.abspath(args.view))
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
        loop_closure_correct=args.loop_closure_correct,
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
                            planar_kwargs=planar_kwargs,
                            sfm_matching=args.sfm_matching)
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
