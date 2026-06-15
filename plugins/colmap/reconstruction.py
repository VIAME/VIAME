#!/usr/bin/env python3
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""COLMAP structure-from-motion + dense reconstruction (``viame.colmap``).

Extracted from ``reconstruct_3d.py``; used only for the non-planar SfM / dense
(MVS) modes, which require COLMAP. Built only when VIAME is configured with
``VIAME_ENABLE_COLMAP=ON``. The planar registration path does NOT use this.

``numpy`` / ``cv2`` / ``pycolmap`` / ``open3d`` are imported lazily via
:func:`import_dependencies` (module globals ``np`` / ``cv2`` / ``pycolmap`` /
``o3d``). Call it once before using any function here.
"""

import os
import sys
import time
import re
import subprocess
import glob

# Populated by import_dependencies()
np = None
cv2 = None
pycolmap = None
o3d = None


def import_dependencies():
    """Import the SfM/dense dependencies into module globals."""
    global np, cv2, pycolmap, o3d
    import numpy as np_
    import cv2 as cv2_
    import pycolmap as pycolmap_
    import open3d as o3d_
    np = np_
    cv2 = cv2_
    pycolmap = pycolmap_
    o3d = o3d_


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
