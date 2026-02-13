#!/usr/bin/env python3
"""
reconstruct_3d.py - Convert UAS drone imagery folders into 3D mesh models.

Pipeline:
  1. SIFT feature extraction (pycolmap)
  2. Feature matching (exhaustive <=50 images, sequential otherwise)
  3. Incremental Structure from Motion -> sparse point cloud + cameras
  3b. (Optional) Prior-coverage polygon output (--coverage-class)
  4. Dense feature matching between overlapping image pairs (OpenCV)  [skip with --no-dense]
  5. Triangulate dense matches -> dense colored point cloud            [skip with --no-dense]
  6. Poisson surface reconstruction (Open3D) -> triangle mesh          [skip with --no-dense]
  7. Export PLY point cloud + PLY/OBJ mesh

Usage:
  python reconstruct_3d.py --install-deps          # check/install dependencies
  python reconstruct_3d.py <image_folder> [--output <output_dir>] [--scale <0.25>]
  python reconstruct_3d.py --all                    # process all subfolders
"""

import os
import sys
import json
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

def run_sfm(image_folder, output_dir, image_names):
    """Run incremental SfM.  Returns a pycolmap.Reconstruction or None."""
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    n = len(image_names)

    # --- feature extraction ---
    with timeit(f"Extracting SIFT features from {n} images"):
        ext_opts = pycolmap.FeatureExtractionOptions()
        ext_opts.max_image_size = 3200
        sift = pycolmap.SiftExtractionOptions()
        sift.max_num_features = 8192
        ext_opts.sift = sift

        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_folder,
            image_names=image_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="OPENCV",
            extraction_options=ext_opts,
        )

    # --- matching ---
    with timeit("Matching features"):
        if n <= 50:
            print("  (exhaustive matching)")
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
                    dense=True):
    """Full pipeline for one image folder."""
    folder_name = os.path.basename(image_folder.rstrip('/'))
    print(f"\n{'#'*70}")
    print(f"  Processing: {folder_name}")
    print(f"  Input:  {image_folder}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}")

    os.makedirs(output_dir, exist_ok=True)

    image_names = get_image_files(image_folder)
    if len(image_names) < 2:
        print(f"ERROR: Need at least 2 images, found {len(image_names)}")
        return False

    print(f"Found {len(image_names)} images")

    # ---- SfM ----
    rec = run_sfm(image_folder, output_dir, image_names)
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
        generate_prior_coverage(rec, coverage_csv, coverage_class)

    # ---- Dense feature matching + triangulation (steps 4-6, optional) ----
    dense_ply = None
    mesh_ply = None
    mesh_obj = None

    if dense:
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
    parser.add_argument("--no-dense", action="store_true",
                        help="Skip dense matching, triangulation, and meshing "
                             "(steps 4-6); only run SfM and optional coverage")
    parser.add_argument("--coverage-class", default=None,
                        help="Generate prior-coverage polygons with this class name "
                             "(e.g. 'suppressed')")
    parser.add_argument("--coverage-file", default="prior_coverage.csv",
                        help="Output filename for prior-coverage CSV "
                             "(default: prior_coverage.csv)")
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
    if args.all:
        base_dir = args.base_dir or os.getcwd()
        for name in sorted(os.listdir(base_dir)):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p) and name not in ('mosaics', '3d_models'):
                imgs = get_image_files(p)
                if len(imgs) >= 2:
                    folders.append(p)
    elif args.folder:
        folders.append(os.path.abspath(args.folder))
    else:
        parser.print_help()
        sys.exit(1)

    if not folders:
        print("No image folders found.")
        sys.exit(1)

    t_total = time.time()
    results = {}
    for folder in folders:
        name = os.path.basename(folder.rstrip('/'))
        out = args.output or os.path.join(base_dir, "3d_models", name)
        ok = process_folder(folder, out, scale=args.scale,
                            max_pairs_per_image=args.max_pairs,
                            coverage_class=args.coverage_class,
                            coverage_file=args.coverage_file,
                            dense=not args.no_dense)
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
