#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Stereo camera calibration tool.

Estimates intrinsic and extrinsic parameters for a stereo camera rig
from images of a chessboard calibration target.
"""

import numpy as np
import cv2
import os
import struct
import sys
import glob
import argparse
import json


def parse_ptscal(filepath):
    """Parse a PtsCAL binary file (3D reference points).

    Returns:
        dict with keys:
          points: dict of {label: (x, y, z)} in mm
          distances: list of (label1, label2, distance_mm) tuples
    """
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'MBL\x00':
            raise ValueError(f"Not a PtsCAL file (expected MBL header): {filepath}")

        count = struct.unpack('<I', f.read(4))[0]

        points = {}
        point_ids = {}
        for _ in range(count):
            entry = f.read(146)
            if len(entry) < 146:
                break

            pbl_magic = entry[:4]
            if pbl_magic != b'PBL\x01':
                raise ValueError(f"Expected PBL entry header, got {pbl_magic!r}")
            pt_id = struct.unpack('<I', entry[4:8])[0]
            label = str(pt_id)

            # 3 coordinate sub-records at offsets 8, 49, 90 (41 bytes each)
            coords = []
            for dim in range(3):
                base = 8 + dim * 41
                coord = struct.unpack('<d', entry[base:base+8])[0]
                coords.append(coord)

            points[label] = tuple(coords)
            point_ids[pt_id] = label

        distances = []
        bdi_count_raw = f.read(4)
        if len(bdi_count_raw) == 4:
            bdi_count = struct.unpack('<I', bdi_count_raw)[0]
            for _ in range(bdi_count):
                bdi_entry = f.read(45)
                if len(bdi_entry) < 45:
                    break
                id1 = struct.unpack('<I', bdi_entry[4:8])[0]
                id2 = struct.unpack('<I', bdi_entry[8:12])[0]
                dist_val = struct.unpack('<d', bdi_entry[12:20])[0]
                label1 = point_ids.get(id1, str(id1))
                label2 = point_ids.get(id2, str(id2))
                distances.append((label1, label2, dist_val))

    return {
        'points': points,
        'distances': distances,
    }


def print_progress(current, total, prefix='Progress', suffix='', bar_length=40):
    """Print a progress bar to stderr"""
    if total == 0:
        return
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = '=' * filled + '-' * (bar_length - filled)
    percent = fraction * 100
    # Pad suffix to fixed width to overwrite previous longer text
    line = f'\r{prefix}: [{bar}] {percent:5.1f}% ({current}/{total}) {suffix:<30}'
    sys.stderr.write(line)
    sys.stderr.flush()
    if current == total:
        sys.stderr.write('\n')


def make_object_points(grid_size=(6,5)):
    """construct the array of object points for camera calibration"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
    return objp


def detect_grid_auto(image, max_dim=5000, min_size=4, max_size=15):
    """Automatically detect grid size by trying different configurations.

    Args:
        image: Grayscale input image
        max_dim: Maximum dimension for detection (larger images are downscaled)
        min_size: Minimum grid dimension to try
        max_size: Maximum grid dimension to try

    Returns:
        Tuple of (corners, grid_size) if found, (None, None) otherwise
    """
    # Try various grid sizes, starting with common ones
    common_sizes = [(6, 5), (7, 6), (8, 6), (9, 6), (5, 4), (8, 5), (7, 5)]

    # Add other sizes systematically
    all_sizes = list(common_sizes)
    for x in range(min_size, max_size + 1):
        for y in range(min_size, x + 1):  # y <= x to avoid duplicates like (5,6) and (6,5)
            if (x, y) not in all_sizes and (y, x) not in all_sizes:
                all_sizes.append((x, y))

    for grid_size in all_sizes:
        corners = detect_grid_image(image, grid_size, max_dim)
        if corners is not None:
            return corners, grid_size
        # Also try transposed
        if grid_size[0] != grid_size[1]:
            corners = detect_grid_image(image, (grid_size[1], grid_size[0]), max_dim)
            if corners is not None:
                return corners, (grid_size[1], grid_size[0])

    return None, None


def detect_grid_image(image, grid_size=(6,5), max_dim=5000):
    """Detect a grid in a grayscale image.

    For large images, detection is performed on a downscaled version first,
    then corners are refined at full resolution.
    """
    min_len = min(image.shape)
    scale = 1.0
    while scale * min_len > max_dim:
        scale /= 2.0

    # termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chess board corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH
    if scale < 1.0:
        small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        ret, corners = cv2.findChessboardCorners(small, grid_size, flags=flags)
        if ret:
            cv2.cornerSubPix(small, corners, (11, 11), (-1, -1), criteria)
            corners /= scale
    else:
        ret, corners = cv2.findChessboardCorners(image, grid_size, flags=flags)

    if ret:
        # refine the location of the corners at full resolution
        cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
        return corners
    else:
        return None


def detect_dots_image(image, max_dim=5000, min_area=30.0, max_area=5000.0,
                      min_circularity=0.65):
    """Detect white dots on a dark background using blob detection.

    Args:
        image: Grayscale input image
        max_dim: Maximum dimension for detection (larger images downscaled)
        min_area: Minimum blob area in pixels
        max_area: Maximum blob area in pixels
        min_circularity: Minimum circularity threshold (0-1)

    Returns:
        Nx1x2 float32 array of dot centers (matching chessboard corner format),
        or None if fewer than 3 dots found.
    """
    min_len = min(image.shape)
    scale = 1.0
    while scale * min_len > max_dim:
        scale /= 2.0

    if scale < 1.0:
        work = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    else:
        work = image

    # Invert: white dots become dark blobs for SimpleBlobDetector
    inverted = cv2.bitwise_not(work)

    # Scale area thresholds
    scale2 = scale * scale
    scaled_min_area = min_area * scale2
    scaled_max_area = max_area * scale2

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 40
    params.maxThreshold = 220
    params.thresholdStep = 10
    params.minRepeatability = 2
    params.filterByArea = True
    params.minArea = scaled_min_area
    params.maxArea = scaled_max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByConvexity = True
    params.minConvexity = 0.70
    params.filterByInertia = True
    params.minInertiaRatio = 0.40
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inverted)

    if len(keypoints) < 3:
        return None, None

    # Extract centers and sizes, scale back to full resolution
    centers = np.array([[[kp.pt[0] / scale, kp.pt[1] / scale]]
                        for kp in keypoints], dtype=np.float32)
    sizes = np.array([kp.size / scale for kp in keypoints], dtype=np.float32)

    # Sub-pixel refinement via intensity-weighted centroid
    refine_dot_centers(image, centers)

    return centers, sizes


def refine_dot_centers(gray, centers, window_radius=7):
    """Refine dot centers using intensity-weighted centroid.

    Args:
        gray: Full-resolution grayscale image
        centers: Nx1x2 float32 array (modified in place)
        window_radius: Half-size of the refinement window
    """
    h, w = gray.shape[:2]
    for i in range(len(centers)):
        cx = int(round(centers[i, 0, 0]))
        cy = int(round(centers[i, 0, 1]))

        x0 = max(0, cx - window_radius)
        y0 = max(0, cy - window_radius)
        x1 = min(w, cx + window_radius + 1)
        y1 = min(h, cy + window_radius + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        roi = gray[y0:y1, x0:x1].astype(np.float64)
        lo, hi = roi.min(), roi.max()

        if hi - lo < 10:
            continue

        threshold = lo + 0.5 * (hi - lo)
        mask = roi >= threshold

        weights = (roi - lo) * mask
        total_w = weights.sum()
        if total_w > 0:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            centers[i, 0, 0] = (weights * xs).sum() / total_w
            centers[i, 0, 1] = (weights * ys).sum() / total_w


def filter_by_roi(corners, roi, sizes=None):
    """Filter detected corners/dots to keep only those within a region of interest.

    Args:
        corners: Nx1x2 float32 array of point positions
        roi: (x1, y1, x2, y2) bounding box
        sizes: Optional Nx1 float32 array of blob sizes (dots mode)

    Returns:
        Filtered corners (and sizes if provided), or (None, None) if all filtered out
    """
    if corners is None or roi is None:
        return corners, sizes
    x1, y1, x2, y2 = roi
    keep = ((corners[:, 0, 0] >= x1) & (corners[:, 0, 0] <= x2) &
            (corners[:, 0, 1] >= y1) & (corners[:, 0, 1] <= y2))
    if keep.sum() == 0:
        return None, None
    filtered_corners = corners[keep]
    filtered_sizes = sizes[keep] if sizes is not None else None
    n_removed = len(corners) - len(filtered_corners)
    if n_removed > 0:
        print(f"    ROI filter removed {n_removed} points "
              f"({len(filtered_corners)} remaining)")
    return filtered_corners, filtered_sizes


def draw_dots(image, centers, color=(0, 255, 0), radius=8, thickness=2):
    """Draw detected dots on an image for visualization.

    Args:
        image: BGR color image to draw on (modified in place)
        centers: Nx1x2 float32 array of dot centers
        color: Drawing color (BGR)
        radius: Circle radius
        thickness: Line thickness
    """
    if centers is None:
        return
    for i in range(len(centers)):
        pt = (int(round(centers[i, 0, 0])), int(round(centers[i, 0, 1])))
        # Circle
        cv2.circle(image, pt, radius, color, thickness)
        # Crosshair
        cv2.line(image, (pt[0] - radius, pt[1]), (pt[0] + radius, pt[1]),
                 color, max(1, thickness // 2))
        cv2.line(image, (pt[0], pt[1] - radius), (pt[0], pt[1] + radius),
                 color, max(1, thickness // 2))
    # Count overlay
    cv2.putText(image, f"{len(centers)} dots", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def filter_target_cluster(centers, sizes=None, min_neighbors=3,
                           neighbor_factor=3.0):
    """Filter detected dots to keep only the main target cluster.

    Removes isolated background dots (labels, reflections, bright spots)
    that would confuse the matching algorithm.  Target dots form a dense
    cluster with regular spacing; background dots are typically isolated.

    Args:
        centers: Nx1x2 float32 array of dot centers
        sizes: Nx float32 array of blob sizes (optional, filtered in parallel)
        min_neighbors: minimum nearby neighbors required to keep a dot
        neighbor_factor: search radius = factor * median nearest-neighbor dist

    Returns:
        (filtered_centers, filtered_sizes) tuple
    """
    from scipy.spatial import cKDTree

    if centers is None or len(centers) < 10:
        return centers, sizes

    pts = centers[:, 0, :]  # Nx2
    tree = cKDTree(pts)

    # Median nearest-neighbor distance characterizes target dot spacing
    nn_dists, _ = tree.query(pts, k=2)  # k=2: self + nearest
    median_nn = np.median(nn_dists[:, 1])

    # Count neighbors within radius
    radius = neighbor_factor * median_nn
    neighbor_lists = tree.query_ball_point(pts, radius)
    neighbor_counts = np.array([len(c) - 1 for c in neighbor_lists])

    keep = neighbor_counts >= min_neighbors

    if keep.sum() < 10:
        return centers, sizes  # Don't filter if too few would remain

    filtered_centers = centers[keep]
    filtered_sizes = sizes[keep] if sizes is not None else None

    n_removed = len(centers) - len(filtered_centers)
    if n_removed > 0:
        print(f"    Filtered {n_removed} background dots "
              f"({len(filtered_centers)} remaining)")

    return filtered_centers, filtered_sizes


def match_dots_to_pts(image_dots, pts_data, image_shape, dot_sizes=None):
    """Match detected image dots to PtsCAL 3D world points.

    Strategy: Sequential face matching.
      1. Split world points into two faces by Z coordinate.
      2. Match face 1 against ALL image dots using affine + homography.
      3. Remove matched image dots.
      4. Match face 2 against REMAINING dots using affine + homography.
      5. Try both orderings (A-first and B-first), pick best via PnP.

    This prevents cross-face confusion since the second face can only match
    to dots not already claimed by the first face.

    Args:
        image_dots: Nx1x2 float32 array from detect_dots_image()
        pts_data: dict from parse_ptscal() with 'points' key
        image_shape: (width, height) tuple
        dot_sizes: Nx float32 array of blob diameters (unused, kept for API compat)

    Returns:
        (matched_image_pts, matched_world_pts, matched_labels) or None
    """
    from scipy.spatial import cKDTree

    # Filter out isolated background dots before matching
    image_dots, dot_sizes = filter_target_cluster(image_dots, dot_sizes)
    if image_dots is None or len(image_dots) < 10:
        print("  Too few dots after background filtering")
        return None

    labels = list(pts_data['points'].keys())
    world_3d = np.array([pts_data['points'][l] for l in labels], dtype=np.float32)
    world_xy = world_3d[:, :2].copy()
    image_2d = image_dots[:, 0, :]  # Nx2

    n_world = len(world_3d)
    n_image = len(image_2d)
    w, h = image_shape

    # Separate world points into two faces by Z coordinate
    z_values = world_3d[:, 2]
    z_median = np.median(z_values)
    face_a_mask = z_values > z_median   # higher Z face
    face_b_mask = ~face_a_mask          # lower Z face

    def affine_match_face(face_mask, img_subset_mask):
        """Match one face's XY pattern to a subset of image dots.

        Returns list of (world_idx, image_idx) pairs.
        """
        face_xy = world_xy[face_mask]
        face_idx = np.where(face_mask)[0]
        n_face = len(face_xy)

        img_sub_idx = np.where(img_subset_mask)[0]
        img_sub = image_2d[img_sub_idx]
        if len(img_sub) < 4:
            return []

        # Normalize face XY
        f_median = np.median(face_xy, axis=0)
        f_centered = face_xy - f_median
        f_mad = np.median(np.sqrt((f_centered ** 2).sum(axis=1)))
        if f_mad < 1e-6:
            return []
        f_norm = f_centered / f_mad

        # Normalize image subset
        img_med = np.median(img_sub, axis=0)
        img_c = img_sub - img_med
        img_m = np.median(np.sqrt((img_c ** 2).sum(axis=1)))
        if img_m < 1e-6:
            return []
        i_norm = img_c / img_m
        tree = cKDTree(i_norm)

        def nn_cost(pts):
            dists, _ = tree.query(pts)
            return np.sort(dists)[:n_face].sum()

        best_cost = float('inf')
        best_angle = 0
        best_mirror = False
        best_aspect = 1.0

        for mirror in [False, True]:
            fn_base = f_norm.copy()
            if mirror:
                fn_base[:, 0] = -fn_base[:, 0]
            for aspect in np.arange(0.3, 3.5, 0.2):
                fn = fn_base.copy()
                fn[:, 0] *= aspect
                for deg in range(0, 360, 5):
                    rad = np.radians(deg)
                    co, si = np.cos(rad), np.sin(rad)
                    rot = fn @ np.array([[co, si], [-si, co]], dtype=np.float32)
                    cost = nn_cost(rot)
                    if cost < best_cost:
                        best_cost = cost
                        best_angle = deg
                        best_mirror = mirror
                        best_aspect = aspect

        fn_base = f_norm.copy()
        if best_mirror:
            fn_base[:, 0] = -fn_base[:, 0]
        for aspect in np.arange(max(0.1, best_aspect - 0.1),
                                best_aspect + 0.15, 0.02):
            fn = fn_base.copy()
            fn[:, 0] *= aspect
            for fine_deg in np.arange(best_angle - 3, best_angle + 3, 0.5):
                rad = np.radians(fine_deg)
                co, si = np.cos(rad), np.sin(rad)
                rot = fn @ np.array([[co, si], [-si, co]], dtype=np.float32)
                cost = nn_cost(rot)
                if cost < best_cost:
                    best_cost = cost
                    best_angle = fine_deg
                    best_aspect = aspect

        fn = f_norm.copy()
        if best_mirror:
            fn[:, 0] = -fn[:, 0]
        fn[:, 0] *= best_aspect
        rad = np.radians(best_angle)
        co, si = np.cos(rad), np.sin(rad)
        rot = fn @ np.array([[co, si], [-si, co]], dtype=np.float32)

        dists, indices = tree.query(rot)
        order = np.argsort(dists)
        max_pairs = min(n_face, 20)
        pairs = []
        for rank in range(max_pairs):
            i = order[rank]
            if dists[i] < 0.5:
                pairs.append((face_idx[i], img_sub_idx[indices[i]], dists[i]))

        seen = {}
        for wi, im, d in pairs:
            if im not in seen or d < seen[im][1]:
                seen[im] = (wi, d)
        return [(wi, im) for im, (wi, _) in seen.items()]

    def homography_match_face(face_mask_cur, face_pairs, img_subset_mask):
        """Use homography to project all face points and match to image dots.

        Only searches within img_subset_mask dots.
        Returns dict of {image_idx: (world_idx, dist)}.
        """
        if len(face_pairs) < 4:
            return {}

        face_world_xy = np.array([world_xy[p[0]] for p in face_pairs], dtype=np.float64)
        face_image_pts = np.array([image_2d[p[1]] for p in face_pairs], dtype=np.float64)

        H, h_mask = cv2.findHomography(face_world_xy, face_image_pts,
                                       cv2.RANSAC, 20.0)
        if H is None or h_mask is None or int(h_mask.sum()) < 4:
            return {}

        face_idx_all = np.where(face_mask_cur)[0]
        face_xy_all = world_xy[face_mask_cur].astype(np.float64)

        sub_idx = np.where(img_subset_mask)[0]
        sub_pts = image_2d[sub_idx]
        if len(sub_pts) < 4:
            return {}
        tree_px = cKDTree(sub_pts)

        # Project all face points, match generously, refit H
        pts_in = face_xy_all.reshape(-1, 1, 2)
        pts_out = cv2.perspectiveTransform(pts_in, H)
        projected = pts_out.reshape(-1, 2)
        dists, indices = tree_px.query(projected)

        seen = {}
        for i in range(len(projected)):
            if dists[i] < 50.0:
                im_idx = sub_idx[indices[i]]
                if im_idx not in seen or dists[i] < seen[im_idx][1]:
                    seen[im_idx] = (face_idx_all[i], dists[i])

        if len(seen) >= 6:
            match_xy = np.array([world_xy[w] for w, _ in seen.values()],
                                dtype=np.float64)
            match_im = np.array([image_2d[im] for im in seen.keys()],
                                dtype=np.float64)
            H_new, _ = cv2.findHomography(match_xy, match_im, cv2.RANSAC, 15.0)
            if H_new is not None:
                H = H_new

        # Final projection
        pts_out = cv2.perspectiveTransform(pts_in, H)
        projected = pts_out.reshape(-1, 2)
        dists, indices = tree_px.query(projected)

        result = {}
        for i in range(len(projected)):
            if dists[i] < 30.0:
                im_idx = sub_idx[indices[i]]
                if im_idx not in result or dists[i] < result[im_idx][1]:
                    result[im_idx] = (face_idx_all[i], dists[i])
        return result

    # --- Strategy: Per-face homography → PnP bootstrap → reproject all ---
    # 1. Match each face separately against all dots using homography
    # 2. Use the better face to estimate camera pose via solvePnP
    # 3. Project ALL 81 world points through the estimated pose
    # 4. Match projected points to image dots (PnP handles both Z-planes)

    all_mask = np.ones(n_image, dtype=bool)

    # Match each face independently against all dots
    pairs_a = affine_match_face(face_a_mask, all_mask)
    pairs_b = affine_match_face(face_b_mask, all_mask)
    matched_a = homography_match_face(face_a_mask, pairs_a, all_mask)
    matched_b = homography_match_face(face_b_mask, pairs_b, all_mask)

    # Estimate focal length from homography (Zhang's method).
    # For each face's homography, the orthogonality constraint r1^T*r2=0
    # gives: h1^T * K^{-T} * K^{-1} * h2 = 0, which we solve for f.
    def estimate_focal_from_H(face_pairs):
        """Estimate focal length from a planar homography."""
        if len(face_pairs) < 6:
            return None
        src = np.array([world_xy[p[0]] for p in face_pairs], dtype=np.float64)
        dst = np.array([image_2d[p[1]] for p in face_pairs], dtype=np.float64)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)
        if H is None:
            return None
        h1, h2 = H[:, 0], H[:, 1]
        cx, cy = w / 2.0, h / 2.0
        # Orthogonality: h1^T * B * h2 = 0 where B = K^{-T} K^{-1}
        # With K = [[f,0,cx],[0,f,cy],[0,0,1]]:
        # Simplified: (h1[0]*h2[0] + h1[1]*h2[1])/f^2
        #   - cx*(h1[0]*h2[2]+h1[2]*h2[0])/f^2
        #   - cy*(h1[1]*h2[2]+h1[2]*h2[1])/f^2
        #   + (cx^2+cy^2)*h1[2]*h2[2]/f^2 + h1[2]*h2[2] = 0
        num = (h1[0]*h2[0] + h1[1]*h2[1]
               - cx*(h1[0]*h2[2] + h1[2]*h2[0])
               - cy*(h1[1]*h2[2] + h1[2]*h2[1])
               + (cx**2 + cy**2) * h1[2]*h2[2])
        den = h1[2] * h2[2]
        if abs(den) < 1e-12:
            return None
        f2 = -num / den
        if f2 <= 0:
            return None
        return np.sqrt(f2)

    # Estimate f from both faces' homographies
    f_estimates = []
    for face_pairs in [pairs_a, pairs_b]:
        f_est = estimate_focal_from_H(face_pairs)
        if f_est is not None and 200 < f_est < 5000:
            f_estimates.append(f_est)

    # Build candidate focal lengths: homography estimates + fallbacks
    f_candidates = set()
    for f_est in f_estimates:
        f_candidates.add(f_est)
    # Always include some fallbacks
    for f_mult in [0.45, 0.55, 0.7, 0.85, 1.0]:
        f_candidates.add(max(w, h) * f_mult)
    f_candidates = sorted(f_candidates)

    # Try PnP from each face's matches + each focal length candidate
    # PnP uses full 3D and handles both Z-planes correctly
    best_all_matched = {}
    best_pnp_score = -1

    for face_name, face_matched, face_mask in [
            ('A', matched_a, face_a_mask), ('B', matched_b, face_b_mask)]:
        if len(face_matched) < 6:
            continue

        seed_w = np.array([world_3d[wi] for wi, _ in face_matched.values()],
                          dtype=np.float64)
        seed_i = np.array([image_2d[im] for im in face_matched.keys()],
                          dtype=np.float64)

        for f_est in f_candidates:
            K_approx = np.array([[f_est, 0, w / 2.0],
                                 [0, f_est, h / 2.0],
                                 [0, 0, 1]], dtype=np.float64)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                seed_w, seed_i, K_approx, None,
                reprojectionError=20.0, iterationsCount=3000,
                flags=cv2.SOLVEPNP_ITERATIVE)

            if not success or inliers is None or len(inliers) < 4:
                continue

            # Refine PnP with inliers only
            inl = inliers.ravel()
            try:
                success2, rvec, tvec = cv2.solvePnP(
                    seed_w[inl], seed_i[inl], K_approx, None,
                    rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                pass

            # Iterative PnP refinement: project → match → re-solve PnP
            tree_all = cKDTree(image_2d)
            for pnp_iter in range(3):
                proj_all, _ = cv2.projectPoints(
                    world_3d.astype(np.float64), rvec, tvec, K_approx, None)
                proj_all = proj_all.reshape(-1, 2)

                dists, indices = tree_all.query(proj_all)
                threshold = 30.0 if pnp_iter == 0 else 20.0

                pnp_matched = {}
                for i in range(n_world):
                    if dists[i] < threshold:
                        im_idx = indices[i]
                        if im_idx not in pnp_matched or dists[i] < pnp_matched[im_idx][1]:
                            pnp_matched[im_idx] = (i, dists[i])

                if len(pnp_matched) < 6:
                    break

                # Re-solve PnP with updated matches
                new_w = np.array([world_3d[wi] for wi, _ in pnp_matched.values()],
                                 dtype=np.float64)
                new_i = np.array([image_2d[im] for im in pnp_matched.keys()],
                                 dtype=np.float64)
                s2, rv2, tv2, inl2 = cv2.solvePnPRansac(
                    new_w, new_i, K_approx, None,
                    rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                    reprojectionError=15.0, iterationsCount=2000)
                if s2 and inl2 is not None and len(inl2) >= 6:
                    rvec, tvec = rv2, tv2

            # Final projection at tight threshold
            proj_all, _ = cv2.projectPoints(
                world_3d.astype(np.float64), rvec, tvec, K_approx, None)
            proj_all = proj_all.reshape(-1, 2)
            dists, indices = tree_all.query(proj_all)

            pnp_matched = {}
            for i in range(n_world):
                if dists[i] < 20.0:
                    im_idx = indices[i]
                    if im_idx not in pnp_matched or dists[i] < pnp_matched[im_idx][1]:
                        pnp_matched[im_idx] = (i, dists[i])

            # Score: number of matches from BOTH faces
            n_a = sum(1 for wi, _ in pnp_matched.values() if face_a_mask[wi])
            n_b = sum(1 for wi, _ in pnp_matched.values() if face_b_mask[wi])
            score = min(n_a, n_b) * 2 + max(n_a, n_b)

            if score > best_pnp_score:
                best_pnp_score = score
                best_all_matched = pnp_matched

    best_total = len(best_all_matched)

    final_world_idx = [w for w, _ in best_all_matched.values()]
    final_image_idx = list(best_all_matched.keys())

    matched_image_pts = image_dots[final_image_idx].copy()
    matched_world_pts = world_3d[final_world_idx].copy()
    matched_labels = [labels[i] for i in final_world_idx]

    n_front = sum(1 for i in final_world_idx if face_a_mask[i])
    n_back = sum(1 for i in final_world_idx if face_b_mask[i])
    size_info = " (size-split)" if dot_sizes is not None else ""
    print(f"  Matched {best_total} of {n_world} "
          f"({n_front} front + {n_back} back, {n_image} dots{size_info})")

    return matched_image_pts, matched_world_pts, matched_labels


def draw_dot_matches(image, matched_pts, matched_labels, unmatched_dots=None,
                     n_world_total=None):
    """Draw dot match results on an image for visualization.

    Args:
        image: BGR color image to draw on (modified in place)
        matched_pts: Kx1x2 float32 array of matched dot centers
        matched_labels: list of K world-point label strings
        unmatched_dots: Nx1x2 float32 array of unmatched image dots (optional)
        n_world_total: total number of world points for overlay (optional)
    """
    # Draw unmatched dots in red
    if unmatched_dots is not None and len(unmatched_dots) > 0:
        for i in range(len(unmatched_dots)):
            pt = (int(round(unmatched_dots[i, 0, 0])),
                  int(round(unmatched_dots[i, 0, 1])))
            cv2.circle(image, pt, 8, (0, 0, 255), 2)

    # Draw matched dots in green with labels
    n_matched = 0
    if matched_pts is not None and len(matched_pts) > 0:
        n_matched = len(matched_pts)
        for i in range(n_matched):
            pt = (int(round(matched_pts[i, 0, 0])),
                  int(round(matched_pts[i, 0, 1])))
            cv2.circle(image, pt, 8, (0, 255, 0), 2)
            cv2.line(image, (pt[0] - 8, pt[1]), (pt[0] + 8, pt[1]),
                     (0, 255, 0), 1)
            cv2.line(image, (pt[0], pt[1] - 8), (pt[0], pt[1] + 8),
                     (0, 255, 0), 1)
            cv2.putText(image, matched_labels[i],
                        (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Count overlay
    if n_world_total is not None:
        total = n_world_total
    else:
        total = n_matched + (len(unmatched_dots) if unmatched_dots is not None else 0)
    cv2.putText(image, f"{n_matched}/{total} matched",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


def track_and_match_all_frames(frame_data, frame_sizes, pts_data, image_shape,
                                camera_name="camera"):
    """Match dots to world points with cross-frame tracking and consensus.

    Improves per-frame matching by:
      1. Independently matching each frame's dots to world points
      2. Tracking dots across consecutive frames via nearest-neighbor
      3. Using majority voting across frames for consistent world-point identity

    Args:
        frame_data: dict of {frame_number: Nx1x2 dots}
        frame_sizes: dict of {frame_number: Nx float32 sizes}
        pts_data: PtsCAL data dict
        image_shape: (width, height)
        camera_name: for display

    Returns:
        dict of {frame_number: (imgpts Kx1x2, objpts Kx3, labels list)}
    """
    from scipy.spatial import cKDTree
    from collections import Counter

    frames = sorted(frame_data.keys())
    n_frames = len(frames)

    # --- Phase 1: Independent per-frame matching ---
    per_frame = {}
    for f in frames:
        print(f"  Frame {f}:")
        result = match_dots_to_pts(frame_data[f], pts_data, image_shape,
                                    frame_sizes.get(f))
        if result is not None:
            per_frame[f] = result
    print(f"  {camera_name}: {len(per_frame)}/{n_frames} frames matched")

    if n_frames < 3 or len(per_frame) < 3:
        return per_frame

    # --- Phase 2: Track dots across consecutive frames ---
    # Uses centroid-compensated mutual nearest-neighbor matching so that
    # even large global target motions between frames are handled.
    print(f"  Tracking dots across {n_frames} frames...")
    max_disp = 80.0  # max residual displacement after centroid compensation
    next_track = 0
    track_map = {}  # frame -> {dot_idx: track_id}

    f0 = frames[0]
    n0 = len(frame_data[f0])
    track_map[f0] = {i: i for i in range(n0)}
    next_track = n0

    for fi in range(1, n_frames):
        prev_f = frames[fi - 1]
        curr_f = frames[fi]
        prev_pts = frame_data[prev_f][:, 0, :]
        curr_pts = frame_data[curr_f][:, 0, :]

        if len(prev_pts) < 3 or len(curr_pts) < 3:
            track_map[curr_f] = {i: next_track + i
                                  for i in range(len(curr_pts))}
            next_track += len(curr_pts)
            continue

        # Compensate for global motion (centroid shift)
        prev_center = np.mean(prev_pts, axis=0)
        curr_center = np.mean(curr_pts, axis=0)
        shift = curr_center - prev_center
        prev_shifted = prev_pts + shift

        # Mutual nearest-neighbor matching (both directions must agree)
        tree_ps = cKDTree(prev_shifted)
        tree_cp = cKDTree(curr_pts)
        dists_fwd, idx_fwd = tree_ps.query(curr_pts)
        dists_bwd, idx_bwd = tree_cp.query(prev_shifted)

        track_map[curr_f] = {}
        for ci in range(len(curr_pts)):
            pi = idx_fwd[ci]
            if dists_fwd[ci] <= max_disp and idx_bwd[pi] == ci:
                prev_track = track_map[prev_f].get(pi)
                if prev_track is not None:
                    track_map[curr_f][ci] = prev_track

        # New tracks for unmatched dots
        for ci in range(len(curr_pts)):
            if ci not in track_map[curr_f]:
                track_map[curr_f][ci] = next_track
                next_track += 1

    # --- Phase 3: Collect world-point votes per track ---
    track_votes = {}
    for f, result in per_frame.items():
        m_img, m_obj, m_labels = result
        frame_dots = frame_data[f][:, 0, :]
        matched_pts = m_img[:, 0, :]

        if len(matched_pts) == 0 or len(frame_dots) == 0:
            continue

        # Map matched image points back to dot indices via position
        tree_frame = cKDTree(frame_dots)
        for mi in range(len(matched_pts)):
            dist, di = tree_frame.query(matched_pts[mi])
            if dist < 3.0:
                track_id = track_map.get(f, {}).get(di)
                if track_id is not None:
                    track_votes.setdefault(track_id, []).append(
                        m_labels[mi])

    # Only trust tracks where ALL votes agree (confirmed identity)
    confirmed_label = {}
    n_confirmed = 0
    n_conflicting = 0
    for track_id, votes in track_votes.items():
        counts = Counter(votes)
        if len(counts) == 1:  # All frames agree on this label
            confirmed_label[track_id] = votes[0]
            n_confirmed += 1
        else:
            n_conflicting += 1

    # Resolve duplicate labels among confirmed tracks
    label_to_track = {}
    for tid in sorted(confirmed_label.keys()):
        label = confirmed_label[tid]
        if label in label_to_track:
            other_tid = label_to_track[label]
            if len(track_votes[tid]) > len(track_votes[other_tid]):
                del confirmed_label[other_tid]
                label_to_track[label] = tid
            else:
                del confirmed_label[tid]
        else:
            label_to_track[label] = tid

    print(f"  {n_confirmed} confirmed, {n_conflicting} conflicting tracks")

    # --- Phase 4: Hybrid results (confirmed tracks + per-frame fallback) ---
    world_map = {l: np.array(pts_data['points'][l], dtype=np.float32)
                 for l in pts_data['points']}

    results = {}
    for f in frames:
        dots = frame_data[f]
        assignments = track_map.get(f, {})

        mdots, mworld, mlabels = [], [], []
        used_labels = set()
        used_dots = set()

        # First: use confirmed track identities (reliable across frames)
        for di, tid in sorted(assignments.items()):
            label = confirmed_label.get(tid)
            if label is not None and label not in used_labels:
                used_labels.add(label)
                used_dots.add(di)
                mdots.append(dots[di])
                mworld.append(world_map[label])
                mlabels.append(label)

        # Second: fill gaps with per-frame matches that don't conflict
        if f in per_frame:
            pf_img, pf_obj, pf_labels = per_frame[f]
            frame_pts = dots[:, 0, :]
            if len(frame_pts) > 0:
                tree_f = cKDTree(frame_pts)
                for mi in range(len(pf_labels)):
                    if pf_labels[mi] not in used_labels:
                        dist, di = tree_f.query(pf_img[mi, 0, :])
                        if dist < 3.0 and di not in used_dots:
                            used_labels.add(pf_labels[mi])
                            used_dots.add(di)
                            mdots.append(dots[di])
                            mworld.append(world_map[pf_labels[mi]])
                            mlabels.append(pf_labels[mi])

        if len(mdots) >= 6:
            results[f] = (
                np.array(mdots, dtype=np.float32),
                np.array(mworld, dtype=np.float32),
                mlabels
            )

    if results:
        avg_matches = np.mean([len(r[2]) for r in results.values()])
        n_confirmed_used = len(confirmed_label)
        print(f"  Result: {len(results)}/{n_frames} frames, "
              f"avg {avg_matches:.0f} matches/frame "
              f"({n_confirmed_used} confirmed + per-frame fill)")
    else:
        return per_frame

    return results


DEFAULT_VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
DEFAULT_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def to_grayscale(image, bayer=False):
    """Convert an image to grayscale, handling color, grayscale, and Bayer inputs.

    Args:
        image: Input image (BGR, grayscale, or Bayer)
        bayer: If True, treat as Bayer pattern image

    Returns:
        Grayscale image
    """
    if bayer:
        # Bayer images: extract first channel and debayer
        if len(image.shape) == 3:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_BayerBG2GRAY)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BayerBG2GRAY)
    elif len(image.shape) == 2:
        # Already grayscale
        return image
    elif image.shape[2] == 1:
        # Single channel but with extra dimension
        return image[:, :, 0]
    else:
        # Color image (BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_color(image, bayer=False):
    """Convert an image to BGR color for display.

    Args:
        image: Input image (BGR, grayscale, or Bayer)
        bayer: If True, treat as Bayer pattern image

    Returns:
        BGR color image
    """
    if bayer:
        if len(image.shape) == 3:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_BayerBG2BGR)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
    elif len(image.shape) == 2:
        # Grayscale to BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        # Already BGR
        return image.copy()


def is_video_file(filepath, video_extensions=None):
    """Check if a file is a video (not an image) by extension"""
    if video_extensions is None:
        video_extensions = DEFAULT_VIDEO_EXTENSIONS
    _, ext = os.path.splitext(filepath.lower())
    return ext in video_extensions


def get_image_files(path, image_extensions=None):
    """Get list of image files from a path (file, directory, or glob pattern)"""
    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS

    if os.path.isdir(path):
        files = []
        for ext in image_extensions:
            pattern = '*' + ext if ext.startswith('.') else '*.' + ext
            files.extend(glob.glob(os.path.join(path, pattern)))
            files.extend(glob.glob(os.path.join(path, pattern.upper())))
        return sorted(files)
    elif os.path.isfile(path):
        return [path]
    else:
        return sorted(glob.glob(path))


def image_frames(input_path, frame_step=1, image_extensions=None, show_progress=True):
    """Yield frames from image file(s) - single file, directory, or glob pattern"""
    files = get_image_files(input_path, image_extensions)

    if len(files) == 0:
        raise ValueError(f"No images found. Input '{input_path}' is not a valid "
                         f"directory or glob pattern matching any images.")

    total_files = len(files)
    print(f"processing {total_files} image(s)")
    frames_yielded = 0
    for n, f in enumerate(files):
        if n % frame_step != 0:
            continue
        frame = cv2.imread(f)
        if frame is None:
            raise ValueError(f"Failed to read image file: {f}")
        frames_yielded += 1
        if show_progress:
            print_progress(n + 1, total_files, prefix='Reading images')
        yield frame, n + 1

    if show_progress:
        print_progress(total_files, total_files, prefix='Reading images')

    if frames_yielded == 0:
        raise ValueError(f"No frames yielded from '{input_path}'. "
                         f"frame_step ({frame_step}) may be too large for {len(files)} file(s).")


def video_frames(video_file, frame_step=1, show_progress=True):
    """Yield frames from a video file"""
    vf = cv2.VideoCapture(video_file)
    if not vf.isOpened():
        vf.release()
        raise ValueError(f"Failed to open video file: {video_file}")

    total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"opened video: {video_file} ({total_frames} frames)")

    frame_number = 0
    frames_yielded = 0
    while True:
        ret, frame = vf.read()
        if not ret:
            break
        frame_number += 1
        if (frame_number - 1) % frame_step == 0:
            frames_yielded += 1
            if show_progress and total_frames > 0:
                print_progress(frame_number, total_frames, prefix='Reading video')
            yield frame, frame_number
    vf.release()

    if show_progress and total_frames > 0:
        print_progress(total_frames, total_frames, prefix='Reading video')

    if frames_yielded == 0:
        raise ValueError(f"No frames yielded from '{video_file}'. "
                         f"Video may be empty or frame_step ({frame_step}) is too large.")


def stereo_frames_separate(left_path, right_path, frame_step=1, show_progress=True):
    """Yield paired frames from separate left and right image sources"""
    # Check if inputs are videos by extension
    left_is_video = os.path.isfile(left_path) and is_video_file(left_path)
    right_is_video = os.path.isfile(right_path) and is_video_file(right_path)

    if left_is_video and right_is_video:
        # Both are videos
        left_cap = cv2.VideoCapture(left_path)
        right_cap = cv2.VideoCapture(right_path)
        if not left_cap.isOpened():
            left_cap.release()
            right_cap.release()
            raise ValueError(f"Failed to open left video: {left_path}")
        if not right_cap.isOpened():
            left_cap.release()
            right_cap.release()
            raise ValueError(f"Failed to open right video: {right_path}")

        total_frames = min(int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                           int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        print(f"opened left video: {left_path}")
        print(f"opened right video: {right_path}")
        print(f"processing up to {total_frames} stereo frame(s)")

        frame_number = 0
        size_validated = False
        while True:
            ret_l, left_frame = left_cap.read()
            ret_r, right_frame = right_cap.read()
            if not ret_l or not ret_r:
                break
            # Validate image sizes match on first frame
            if not size_validated:
                if left_frame.shape != right_frame.shape:
                    left_cap.release()
                    right_cap.release()
                    raise ValueError(f"Left and right video frame sizes do not match: "
                                     f"left={left_frame.shape[:2]}, right={right_frame.shape[:2]}")
                size_validated = True
            frame_number += 1
            if (frame_number - 1) % frame_step == 0:
                if show_progress and total_frames > 0:
                    print_progress(frame_number, total_frames, prefix='Reading stereo video')
                yield left_frame, right_frame, frame_number
        left_cap.release()
        right_cap.release()

        if show_progress and total_frames > 0:
            print_progress(total_frames, total_frames, prefix='Reading stereo video')
    elif left_is_video or right_is_video:
        raise ValueError("Both left and right must be videos, or both must be images. "
                         f"Left is {'video' if left_is_video else 'images'}, "
                         f"right is {'video' if right_is_video else 'images'}.")
    else:
        # Image lists
        left_files = get_image_files(left_path)
        right_files = get_image_files(right_path)

        if len(left_files) == 0:
            raise ValueError(f"No images found in left path: {left_path}")
        if len(right_files) == 0:
            raise ValueError(f"No images found in right path: {right_path}")

        if len(left_files) != len(right_files):
            print(f"Warning: left ({len(left_files)}) and right ({len(right_files)}) "
                  f"image counts differ. Using minimum count.")

        num_frames = min(len(left_files), len(right_files))
        print(f"processing {num_frames} stereo image pair(s)")
        size_validated = False
        for n in range(0, num_frames, frame_step):
            left_frame = cv2.imread(left_files[n])
            right_frame = cv2.imread(right_files[n])
            if left_frame is None:
                raise ValueError(f"Failed to read left image: {left_files[n]}")
            if right_frame is None:
                raise ValueError(f"Failed to read right image: {right_files[n]}")
            # Validate image sizes match on first frame
            if not size_validated:
                if left_frame.shape != right_frame.shape:
                    raise ValueError(f"Left and right image sizes do not match: "
                                     f"left={left_frame.shape[:2]}, right={right_frame.shape[:2]}")
                size_validated = True
            if show_progress:
                print_progress(n + 1, num_frames, prefix='Reading stereo images')
            yield left_frame, right_frame, n + 1

        if show_progress:
            print_progress(num_frames, num_frames, prefix='Reading stereo images')


def detect_grid_stereo_separate(left_path, right_path, grid_size=(6,5),
                                 frame_step=1, gui=False, bayer=False, auto_grid=False,
                                 max_frames=0, dots=False, pts_data=None,
                                 left_roi=None, right_roi=None):
    """Detect a grid in each frame from separate left/right image sources

    Args:
        left_path: Path to left camera images
        right_path: Path to right camera images
        grid_size: Tuple of (width, height) inner corners
        frame_step: Process every Nth frame
        gui: Show detection results in GUI
        bayer: Input is Bayer pattern
        auto_grid: Automatically detect grid size from first successful detection
        max_frames: Maximum number of frames to use (0=unlimited)
        dots: Detect dots instead of chessboard corners
        pts_data: PtsCAL data for dot matching (if provided with dots=True)

    Returns:
        Tuple of (img_shape, left_data, right_data, detected_grid_size)
    """
    # Dicts to store corner points from all the images.
    left_data = {}
    right_data = {}
    left_sizes = {}   # frame -> Nx1 float32 blob sizes (dots mode only)
    right_sizes = {}
    img_shape = None
    detected_grid_size = grid_size

    print(f"left: {left_path}")
    print(f"right: {right_path}")

    for left_frame, right_frame, frame_number in stereo_frames_separate(
            left_path, right_path, frame_step):
        # Check if we've reached max_frames limit
        if max_frames > 0 and len(left_data) >= max_frames and len(right_data) >= max_frames:
            print(f"Reached max_frames limit ({max_frames}), stopping detection")
            break

        left_gray = to_grayscale(left_frame, bayer)
        right_gray = to_grayscale(right_frame, bayer)

        if img_shape is None:
            img_shape = left_gray.shape[::-1]

        left_sz = None
        right_sz = None
        if dots:
            # Dot detection mode
            left_corners, left_sz = detect_dots_image(left_gray)
            right_corners, right_sz = detect_dots_image(right_gray)
        elif auto_grid and not left_data and not right_data:
            # Auto-detect grid size on first frame if requested
            print("Auto-detecting grid size...")
            left_corners, detected_size = detect_grid_auto(left_gray)
            if left_corners is not None:
                detected_grid_size = detected_size
                print(f"Detected grid size: {detected_grid_size[0]}x{detected_grid_size[1]}")
                right_corners = detect_grid_image(right_gray, detected_grid_size)
            else:
                right_corners, detected_size = detect_grid_auto(right_gray)
                if right_corners is not None:
                    detected_grid_size = detected_size
                    print(f"Detected grid size: {detected_grid_size[0]}x{detected_grid_size[1]}")
                else:
                    print("Warning: Could not auto-detect grid size, using default")
                    left_corners = None
                    right_corners = None
        else:
            left_corners = detect_grid_image(left_gray, detected_grid_size)
            right_corners = detect_grid_image(right_gray, detected_grid_size)

        # Apply per-camera ROI filter if specified
        if left_roi is not None:
            left_corners, left_sz = filter_by_roi(left_corners, left_roi, left_sz)
        if right_roi is not None:
            right_corners, right_sz = filter_by_roi(right_corners, right_roi, right_sz)

        target_label = "dots" if dots else "checkerboard"

        # If found, add object points, image points (after refining them)
        if left_corners is not None:
            print(f"found {target_label} in frame {frame_number} left"
                  + (f" ({len(left_corners)} pts)" if dots else ""))
            left_data[frame_number] = left_corners
            if left_sz is not None:
                left_sizes[frame_number] = left_sz
        if right_corners is not None:
            print(f"found {target_label} in frame {frame_number} right"
                  + (f" ({len(right_corners)} pts)" if dots else ""))
            right_data[frame_number] = right_corners
            if right_sz is not None:
                right_sizes[frame_number] = right_sz
        if left_corners is None and right_corners is None:
            print(f"{target_label} not found in frame {frame_number}")

        # Draw and display the corners
        if gui:
            left_color = to_color(left_frame, bayer)
            right_color = to_color(right_frame, bayer)
            if dots and pts_data is not None:
                n_world = len(pts_data['points'])
                # Run matching and show matched labels
                for corners, color_img, side in [
                        (left_corners, left_color, "left"),
                        (right_corners, right_color, "right")]:
                    if corners is not None:
                        result = match_dots_to_pts(corners, pts_data, img_shape)
                        if result is not None:
                            m_img, m_obj, m_labels = result
                            # Find unmatched image dots
                            matched_set = set(map(tuple, m_img[:, 0, :].tolist()))
                            unmatched = np.array(
                                [c for c in corners if tuple(c[0].tolist()) not in matched_set],
                                dtype=np.float32)
                            if len(unmatched) == 0:
                                unmatched = None
                            draw_dot_matches(color_img, m_img, m_labels,
                                             unmatched, n_world)
                        else:
                            draw_dots(color_img, corners, color=(0, 0, 255))
                    else:
                        cv2.putText(color_img, "no dots", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            elif dots:
                draw_dots(left_color, left_corners)
                draw_dots(right_color, right_corners)
            else:
                if left_corners is not None:
                    cv2.drawChessboardCorners(left_color, detected_grid_size, left_corners, True)
                if right_corners is not None:
                    cv2.drawChessboardCorners(right_color, detected_grid_size, right_corners, True)
            # Draw ROI rectangles if specified
            if left_roi is not None:
                rx1, ry1, rx2, ry2 = left_roi
                cv2.rectangle(left_color, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            if right_roi is not None:
                rx1, ry1, rx2, ry2 = right_roi
                cv2.rectangle(right_color, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            # Stack images side by side for display
            combined = np.hstack([left_color, right_color])
            cv2.imshow('img', combined)
            cv2.waitKey(-1)

    print("done")
    if gui:
        cv2.destroyAllWindows()

    if img_shape is None:
        raise ValueError(f"No frames were processed from left='{left_path}' and "
                         f"right='{right_path}'. Check that the input paths are correct.")

    return img_shape, left_data, right_data, detected_grid_size, left_sizes, right_sizes


def detect_grid_video(input_path, grid_size=(6,5), frame_step=1, gui=False, bayer=False,
                      auto_grid=False, max_frames=0, dots=False, pts_data=None,
                      left_roi=None, right_roi=None):
    """Detect a grid in each frame of video or image(s)

    Args:
        input_path: Path to video file or image glob pattern
        grid_size: Tuple of (width, height) inner corners
        frame_step: Process every Nth frame
        gui: Show detection results in GUI
        bayer: Input is Bayer pattern
        auto_grid: Automatically detect grid size from first successful detection
        max_frames: Maximum number of frames to use (0=unlimited)
        dots: Detect dots instead of chessboard corners
        pts_data: PtsCAL data for dot matching (if provided with dots=True)

    Returns:
        Tuple of (img_shape, left_data, right_data, detected_grid_size)
    """
    # Dicts to store corner points from all the images.
    left_data = {}
    right_data = {}
    left_sizes = {}   # frame -> Nx1 float32 blob sizes (dots mode only)
    right_sizes = {}
    img_shape = None
    detected_grid_size = grid_size

    # Choose appropriate frame source based on input type
    if os.path.isfile(input_path) and is_video_file(input_path):
        frame_source = video_frames(input_path, frame_step)
    else:
        frame_source = image_frames(input_path, frame_step)

    for frame, frame_number in frame_source:
        # Check if we've reached max_frames limit
        if max_frames > 0 and len(left_data) >= max_frames and len(right_data) >= max_frames:
            print(f"Reached max_frames limit ({max_frames}), stopping detection")
            break

        left_img = frame[:, 0:frame.shape[1] // 2]
        right_img = frame[:, frame.shape[1] // 2:]
        left_gray = to_grayscale(left_img, bayer)
        right_gray = to_grayscale(right_img, bayer)
        if img_shape is None:
            img_shape = left_gray.shape[::-1]

        left_sz = None
        right_sz = None
        if dots:
            # Dot detection mode
            left_corners, left_sz = detect_dots_image(left_gray)
            right_corners, right_sz = detect_dots_image(right_gray)
        elif auto_grid and not left_data and not right_data:
            # Auto-detect grid size on first frame if requested
            print("Auto-detecting grid size...")
            left_corners, detected_size = detect_grid_auto(left_gray)
            if left_corners is not None:
                detected_grid_size = detected_size
                print(f"Detected grid size: {detected_grid_size[0]}x{detected_grid_size[1]}")
            else:
                right_corners, detected_size = detect_grid_auto(right_gray)
                if right_corners is not None:
                    detected_grid_size = detected_size
                    print(f"Detected grid size: {detected_grid_size[0]}x{detected_grid_size[1]}")
                    left_corners = detect_grid_image(left_gray, detected_grid_size)
                else:
                    print("Warning: Could not auto-detect grid size, using default")
                    left_corners = None
                    right_corners = None
        else:
            left_corners = detect_grid_image(left_gray, detected_grid_size)
            right_corners = detect_grid_image(right_gray, detected_grid_size)

        if not dots and (not auto_grid or left_data or right_data):
            # Normal detection with known grid size
            if left_corners is None:
                left_corners = detect_grid_image(left_gray, detected_grid_size)
            if right_corners is None:
                right_corners = detect_grid_image(right_gray, detected_grid_size)

        # Apply per-camera ROI filter if specified
        if left_roi is not None:
            left_corners, left_sz = filter_by_roi(left_corners, left_roi, left_sz)
        if right_roi is not None:
            right_corners, right_sz = filter_by_roi(right_corners, right_roi, right_sz)

        target_label = "dots" if dots else "checkerboard"

        # If found, add object points, image points (after refining them)
        if left_corners is not None:
            print(f"found {target_label} in frame {frame_number} left"
                  + (f" ({len(left_corners)} pts)" if dots else ""))
            left_data[frame_number] = left_corners
            if left_sz is not None:
                left_sizes[frame_number] = left_sz
        if right_corners is not None:
            print(f"found {target_label} in frame {frame_number} right"
                  + (f" ({len(right_corners)} pts)" if dots else ""))
            right_data[frame_number] = right_corners
            if right_sz is not None:
                right_sizes[frame_number] = right_sz
        if left_corners is None and right_corners is None:
            print(f"{target_label} not found in frame {frame_number}")

        # Draw and display the corners
        if gui:
            color_frame = to_color(frame, bayer)
            if dots and pts_data is not None:
                n_world = len(pts_data['points'])
                # Run matching and show matched labels for each side
                x_offset = img_shape[0]  # right half offset in stitched frame
                for corners, x_off, side in [
                        (left_corners, 0, "left"),
                        (right_corners, x_offset, "right")]:
                    if corners is not None:
                        result = match_dots_to_pts(corners, pts_data, img_shape)
                        # Shift coordinates for drawing on stitched frame
                        shift = np.array([[[x_off, 0]]], dtype=np.float32)
                        if result is not None:
                            m_img, m_obj, m_labels = result
                            matched_set = set(map(tuple, m_img[:, 0, :].tolist()))
                            unmatched = np.array(
                                [c for c in corners if tuple(c[0].tolist()) not in matched_set],
                                dtype=np.float32)
                            if len(unmatched) == 0:
                                unmatched = None
                            else:
                                unmatched = unmatched + shift
                            draw_dot_matches(color_frame, m_img + shift,
                                             m_labels, unmatched, n_world)
                        else:
                            draw_dots(color_frame, corners + shift,
                                      color=(0, 0, 255))
            elif dots:
                if left_corners is not None:
                    draw_dots(color_frame, left_corners)
                if right_corners is not None:
                    offset = np.repeat(np.array([[[img_shape[0], 0]]],
                                                dtype=right_corners.dtype),
                                       right_corners.shape[0], axis=0)
                    draw_dots(color_frame, right_corners + offset)
            else:
                if left_corners is not None:
                    cv2.drawChessboardCorners(color_frame, detected_grid_size, left_corners, True)
                if right_corners is not None:
                    offset = np.repeat(np.array([[[img_shape[0], 0]]],
                                                dtype=right_corners.dtype),
                                       right_corners.shape[0], axis=0)
                    shift_right_corners = right_corners + offset
                    cv2.drawChessboardCorners(color_frame, detected_grid_size,
                                              shift_right_corners, True)
            # Draw ROI rectangles if specified
            x_off = img_shape[0]
            if left_roi is not None:
                rx1, ry1, rx2, ry2 = left_roi
                cv2.rectangle(color_frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            if right_roi is not None:
                rx1, ry1, rx2, ry2 = right_roi
                cv2.rectangle(color_frame, (rx1 + x_off, ry1), (rx2 + x_off, ry2),
                              (255, 255, 0), 2)
            cv2.imshow('img', color_frame)
            cv2.waitKey(-1)
    print("done")
    if gui:
        cv2.destroyAllWindows()

    if img_shape is None:
        raise ValueError(f"No frames were processed from '{input_path}'. "
                         f"Check that the input video or image path is correct.")

    return img_shape, left_data, right_data, detected_grid_size, left_sizes, right_sizes


def calibrate_single_camera(data, object_points, img_shape, camera_name="camera"):
    """Calibrate a single camera with automatic model selection.

    Iteratively tests simpler distortion models and selects the simplest
    model that doesn't significantly increase reprojection error.

    Args:
        data: dict of {frame_number: Nx1x2 image points} OR list of Nx1x2 arrays
        object_points: shared Mx3 array (checkerboard) OR list of per-frame Mx3 arrays
        img_shape: (width, height) tuple
        camera_name: name for progress display
    """
    if isinstance(object_points, list):
        objpoints = object_points  # Per-frame (dots+pts mode)
    else:
        objpoints = [object_points] * len(data)  # Shared (checkerboard mode)
    if isinstance(data, list):
        imgpoints = data
    else:
        frames = list(data.keys())
        imgpoints = list(data.values())

    # For non-regular grids (dots+pts), do iterative outlier rejection.
    # The matching may produce ~30% wrong correspondences which would
    # throw off calibration. We iteratively calibrate, compute per-point
    # reprojection errors, and remove points with large errors.
    if isinstance(object_points, list):
        for outlier_iter in range(5):
            # Quick calibration to identify outliers
            sample_obj = objpoints[0]
            z_range = sample_obj[:, 2].max() - sample_obj[:, 2].min()
            is_np = z_range > 1.0
            w, h = img_shape
            f_i = float(max(w, h))
            K_q = np.array([[f_i, 0, w/2.0], [0, f_i, h/2.0], [0, 0, 1]],
                           dtype=np.float64)
            d_q = np.array([0, 0, 0, 0], dtype=np.float64)
            fl_q = 0
            if is_np:
                fl_q = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
            try:
                ret, mtx, dist_c, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, img_shape, K_q, d_q, flags=fl_q)
            except cv2.error:
                break

            # Compute per-point reprojection errors
            total_removed = 0
            new_objpoints = []
            new_imgpoints = []
            for i in range(len(objpoints)):
                proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                            mtx, dist_c)
                proj = proj.reshape(-1, 2)
                actual = imgpoints[i].reshape(-1, 2)
                errors = np.sqrt(((proj - actual) ** 2).sum(axis=1))
                # Remove points with error > 2.5× median (or > 30px absolute)
                med_err = np.median(errors)
                threshold = max(min(2.5 * med_err, 50.0), 15.0)
                keep = errors < threshold
                if keep.sum() >= 6:
                    new_objpoints.append(objpoints[i][keep])
                    new_imgpoints.append(imgpoints[i][keep])
                    total_removed += (~keep).sum()

            if total_removed == 0 or len(new_objpoints) < 3:
                break
            objpoints = new_objpoints
            imgpoints = new_imgpoints

    # Define calibration steps for progress reporting
    steps = [
        ("Initial calibration", None),
        ("Testing aspect ratio", cv2.CALIB_FIX_ASPECT_RATIO),
        ("Testing principal point", cv2.CALIB_FIX_PRINCIPAL_POINT),
        ("Testing tangential distortion", cv2.CALIB_ZERO_TANGENT_DIST),
        ("Testing K3 distortion", cv2.CALIB_FIX_K3),
        ("Testing K2 distortion", cv2.CALIB_FIX_K2),
        ("Testing K1 distortion", cv2.CALIB_FIX_K1),
    ]
    total_steps = len(steps)

    flags = 0

    # Check if object points are non-planar (e.g., dots+pts with two Z-planes).
    # OpenCV requires CALIB_USE_INTRINSIC_GUESS for non-planar rigs because it
    # can't initialize K from homographies when points aren't coplanar.
    sample_obj = objpoints[0]
    z_range = sample_obj[:, 2].max() - sample_obj[:, 2].min()
    non_planar = z_range > 1.0  # more than 1mm Z variation

    w, h = img_shape
    f_init = float(max(w, h))
    K = np.array([[f_init, 0, w / 2.0],
                  [0, f_init, h / 2.0],
                  [0, 0, 1]], dtype=np.float64)
    d = np.array([0, 0, 0, 0], dtype=np.float64)

    if non_planar:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # Fix principal point for non-planar rigs with sparse points to prevent
        # the optimizer from diverging
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    def try_calibrate(objp, imgp, shape, K_init, d_init, cal_flags):
        """Wrapper that catches OpenCV calibration errors."""
        try:
            return cv2.calibrateCamera(objp, imgp, shape, K_init, d_init,
                                       flags=cal_flags)
        except cv2.error:
            return None

    # Step 1: Initial calibration
    print_progress(1, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[0][0])
    cal_result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)

    if cal_result is None:
        raise ValueError(f"Initial calibration failed for {camera_name}. "
                         f"Insufficient or poor quality point matches.")

    ret, mtx, dist, rvecs, tvecs = cal_result

    # Step 2: Test aspect ratio
    print_progress(2, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[1][0])
    aspect_ratio = mtx[0,0] / mtx[1,1]
    if 1.0 - min(aspect_ratio, 1.0/aspect_ratio) < 0.01:
        flags += cv2.CALIB_FIX_ASPECT_RATIO
        result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)
        if result is not None:
            cal_result = result
            ret, mtx, dist, rvecs, tvecs = cal_result

    # Step 3: Test principal point (skip if already fixed for non-planar)
    print_progress(3, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[2][0])
    if not (flags & cv2.CALIB_FIX_PRINCIPAL_POINT):
        pp = np.array([mtx[0,2], mtx[1,2]])
        rel_pp_diff = (pp - np.array(img_shape)/2) / np.array(img_shape)
        if max(abs(rel_pp_diff)) < 0.05:
            flags += cv2.CALIB_FIX_PRINCIPAL_POINT
            result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)
            if result is not None:
                cal_result = result

    # set a threshold 25% more than the baseline error
    error_threshold = 1.25 * cal_result[0]
    last_result = (cal_result, flags)

    # Step 4: Test tangential distortion
    print_progress(4, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[3][0])
    flags += cv2.CALIB_ZERO_TANGENT_DIST
    result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)
    if result is None or result[0] > error_threshold:
        print_progress(total_steps, total_steps, prefix=f'Calibrating {camera_name}', suffix='Done')
        return last_result
    cal_result = result
    last_result = (cal_result, flags)

    # Step 5: Test K3
    print_progress(5, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[4][0])
    flags += cv2.CALIB_FIX_K3
    result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)
    if result is None or result[0] > error_threshold:
        print_progress(total_steps, total_steps, prefix=f'Calibrating {camera_name}', suffix='Done')
        return last_result
    last_result = (result, flags)

    # Step 6: Test K2
    print_progress(6, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[5][0])
    flags += cv2.CALIB_FIX_K2
    result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)
    if result is None or result[0] > error_threshold:
        print_progress(total_steps, total_steps, prefix=f'Calibrating {camera_name}', suffix='Done')
        return last_result
    last_result = (result, flags)

    # Step 7: Test K1
    print_progress(7, total_steps, prefix=f'Calibrating {camera_name}', suffix=steps[6][0])
    flags += cv2.CALIB_FIX_K1
    result = try_calibrate(objpoints, imgpoints, img_shape, K, d, flags)
    print_progress(total_steps, total_steps, prefix=f'Calibrating {camera_name}', suffix='Done')
    if result is None or result[0] > error_threshold:
        return last_result
    return (result, flags)


def estimate_essential_from_stereo_frames(left_path, right_path, input_path,
                                           K_left, dist_left, K_right, dist_right,
                                           frame_step=1, max_samples=60):
    """Estimate Essential matrix from stereo frame pairs using SIFT/ORB features.

    Works with both separate stereo (left_path + right_path) and stitched
    video (input_path) sources. Returns a list of per-frame estimates with
    R, T, and quality metrics.

    Args:
        left_path: Path to left camera images/video (separate mode), or None
        right_path: Path to right camera images/video (separate mode), or None
        input_path: Path to stitched video/images, or None
        K_left, dist_left: Left camera intrinsics and distortion
        K_right, dist_right: Right camera intrinsics and distortion
        frame_step: Process every Nth frame
        max_samples: Maximum number of successful E estimates to collect

    Returns:
        List of dicts with 'R', 'T', 'n_inliers', 'n_pose' keys
    """
    # Try SIFT first, fall back to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=3000)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        use_sift = True
        print("  Using SIFT features")
    except Exception:
        detector = cv2.ORB_create(nfeatures=3000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        use_sift = False
        print("  Using ORB features")

    all_results = []

    def process_pair(left_gray, right_gray, frame_id):
        """Process a single stereo pair for Essential matrix estimation."""
        kp1, des1 = detector.detectAndCompute(left_gray, None)
        kp2, des2 = detector.detectAndCompute(right_gray, None)

        if des1 is None or des2 is None or len(kp1) < 30 or len(kp2) < 30:
            return None

        if use_sift:
            raw_matches = matcher.knnMatch(des1, des2, k=2)
            matches = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        matches.append(m)
            matches = sorted(matches, key=lambda m: m.distance)
        else:
            matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)

        if len(matches) < 20:
            return None

        n_match = min(500, len(matches))
        matches = matches[:n_match]

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)

        # Undistort points
        pts1_ud = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K_left, dist_left)
        pts2_ud = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K_right, dist_right)

        E, mask = cv2.findEssentialMat(
            pts1_ud, pts2_ud, np.eye(3),
            method=cv2.RANSAC, prob=0.999, threshold=0.001)

        if E is None or mask is None:
            return None

        n_inliers = int(mask.sum())
        if n_inliers < 15:
            return None

        _, R, T, mask_pose = cv2.recoverPose(
            E, pts1_ud, pts2_ud, np.eye(3), mask=mask)
        n_pose = int(mask_pose.sum()) if mask_pose is not None else 0

        if n_pose < 10:
            return None

        return {
            'frame': frame_id, 'R': R.copy(), 'T': T.copy(),
            'n_inliers': n_inliers, 'n_pose': n_pose,
            'n_matches': len(matches)
        }

    # Iterate over stereo frames
    if left_path is not None and right_path is not None:
        # Separate stereo mode
        try:
            for left_frame, right_frame, frame_num in stereo_frames_separate(
                    left_path, right_path, frame_step, show_progress=False):
                if len(all_results) >= max_samples:
                    break
                left_gray = to_grayscale(left_frame)
                right_gray = to_grayscale(right_frame)
                result = process_pair(left_gray, right_gray, frame_num)
                if result is not None:
                    all_results.append(result)
                    print_progress(len(all_results), max_samples,
                                   prefix='  E estimation',
                                   suffix=f'frame {frame_num}: {result["n_inliers"]} inliers')
        except (ValueError, StopIteration):
            pass
    elif input_path is not None:
        # Stitched video/image mode
        try:
            if os.path.isfile(input_path) and is_video_file(input_path):
                frame_source = video_frames(input_path, frame_step, show_progress=False)
            else:
                frame_source = image_frames(input_path, frame_step, show_progress=False)

            for frame, frame_num in frame_source:
                if len(all_results) >= max_samples:
                    break
                left_img = frame[:, 0:frame.shape[1] // 2]
                right_img = frame[:, frame.shape[1] // 2:]
                left_gray = to_grayscale(left_img)
                right_gray = to_grayscale(right_img)
                result = process_pair(left_gray, right_gray, frame_num)
                if result is not None:
                    all_results.append(result)
                    print_progress(len(all_results), max_samples,
                                   prefix='  E estimation',
                                   suffix=f'frame {frame_num}: {result["n_inliers"]} inliers')
        except (ValueError, StopIteration):
            pass

    if all_results:
        print_progress(max_samples, max_samples,
                       prefix='  E estimation', suffix='Done')
    print(f"  Collected {len(all_results)} Essential matrix estimates")
    return all_results


def robust_consensus_RT(all_results):
    """Compute robust consensus R, T direction from multiple Essential matrix estimates.

    Uses median Rodrigues vectors with outlier filtering on both rotation and
    translation direction. Returns unit-norm T direction (scale is unknown from E).

    Args:
        all_results: List of dicts with 'R' and 'T' keys (from estimate_essential)

    Returns:
        (R, T_dir) where T_dir has unit norm, or (None, None) if insufficient data
    """
    if len(all_results) < 3:
        return None, None

    R_list = [r['R'] for r in all_results]
    T_list = [r['T'] for r in all_results]

    rvecs = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_list])
    tvecs = np.array([T.flatten() for T in T_list])

    # Median rotation vector
    med_rvec = np.median(rvecs, axis=0)

    # Filter rotation outliers (>3x median deviation or >0.02 rad minimum)
    rvec_dists = np.linalg.norm(rvecs - med_rvec, axis=1)
    thresh = max(3.0 * np.median(rvec_dists), 0.02)
    inliers = rvec_dists < thresh

    print(f"  Rotation inliers: {inliers.sum()}/{len(inliers)}")

    # Also filter T direction outliers
    t_norms = np.linalg.norm(tvecs, axis=1, keepdims=True)
    t_dirs = tvecs / np.maximum(t_norms, 1e-10)
    med_tdir = np.median(t_dirs[inliers], axis=0)
    med_tdir /= np.linalg.norm(med_tdir)

    t_angle_dists = 1.0 - (t_dirs @ med_tdir)  # ~0 for aligned, ~2 for opposite
    t_thresh = max(3.0 * np.median(t_angle_dists[inliers]), 0.01)
    combined_inliers = inliers & (t_angle_dists < t_thresh)

    print(f"  Combined inliers: {combined_inliers.sum()}/{len(combined_inliers)}")

    if combined_inliers.sum() < 3:
        combined_inliers = inliers  # Fall back to rotation-only filtering

    R_consensus, _ = cv2.Rodrigues(np.median(rvecs[combined_inliers], axis=0))
    T_consensus = np.median(tvecs[combined_inliers], axis=0).reshape(3, 1)
    T_consensus = T_consensus / np.linalg.norm(T_consensus)

    return R_consensus, T_consensus


def cross_label_right_dots(left_matched, K_left, dist_left, K_right, dist_right,
                            R, T, pts_data, right_raw_dots, img_shape,
                            max_dist=15.0):
    """Label right camera dots by projecting world points through left PnP + stereo RT.

    For each frame where left camera dots are matched to world points:
      1. Solve PnP for left camera pose
      2. Compute right camera pose from left pose + stereo RT
      3. Project all world points into right camera
      4. Match projected positions to detected right camera dots

    This bypasses the unreliable independent dot-to-PtsCAL matching for the
    right camera, using the stereo geometry to transfer labels instead.

    Args:
        left_matched: dict {frame: (imgpts, objpts, labels)} from left camera
        K_left, dist_left: Left camera intrinsics
        K_right, dist_right: Right camera intrinsics
        R, T: Stereo rotation and translation (left→right)
        pts_data: PtsCAL data with 'points' dict
        right_raw_dots: dict {frame: Nx1x2 detected dots} for right camera
        img_shape: (width, height)
        max_dist: Maximum pixel distance for dot-to-projection matching

    Returns:
        dict {frame: (imgpts Kx1x2, objpts Kx3, labels list)} for right camera
    """
    all_3d = np.array([pts_data['points'][lbl]
                       for lbl in sorted(pts_data['points'].keys())],
                      dtype=np.float64)
    all_labels = sorted(pts_data['points'].keys())
    w, h = img_shape

    right_matched = {}
    for f in sorted(left_matched.keys()):
        if f not in right_raw_dots:
            continue
        l_img, l_obj, l_labels = left_matched[f]
        if len(l_labels) < 10:
            continue

        # Solve PnP for left camera
        ok, rvec_l, tvec_l, inliers = cv2.solvePnPRansac(
            l_obj.astype(np.float64),
            l_img.reshape(-1, 1, 2).astype(np.float64),
            K_left, dist_left, iterationsCount=300, reprojectionError=6.0)
        if not ok or inliers is None or len(inliers) < 6:
            continue

        # Refine with inliers
        idx = inliers.flatten()
        ok, rvec_l, tvec_l = cv2.solvePnP(
            l_obj[idx].astype(np.float64),
            l_img[idx].reshape(-1, 1, 2).astype(np.float64),
            K_left, dist_left, rvec=rvec_l, tvec=tvec_l,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        # Compute right camera pose: R_right = R_stereo @ R_left, T_right = R_stereo @ T_left + T_stereo
        R_L, _ = cv2.Rodrigues(rvec_l)
        R_right = R @ R_L
        T_right = R @ tvec_l + T
        rvec_right, _ = cv2.Rodrigues(R_right)

        # Project all world points into right camera
        proj, _ = cv2.projectPoints(all_3d, rvec_right, T_right, K_right, dist_right)
        proj = proj.reshape(-1, 2)

        # Match projections to detected right camera dots
        r_pts = right_raw_dots[f].reshape(-1, 2)
        matched_img, matched_obj, matched_labels = [], [], []
        used = set()
        for j in range(len(all_3d)):
            p = proj[j]
            if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h:
                continue
            dists = np.sqrt(((r_pts - p) ** 2).sum(axis=1))
            mi = np.argmin(dists)
            if dists[mi] < max_dist and mi not in used:
                matched_img.append(r_pts[mi])
                matched_obj.append(all_3d[j])
                matched_labels.append(all_labels[j])
                used.add(mi)

        if len(matched_labels) >= 8:
            right_matched[f] = (
                np.array(matched_img, np.float32).reshape(-1, 1, 2),
                np.array(matched_obj, np.float32),
                matched_labels)

    return right_matched


def optimize_translation_scale(K_left, dist_left, K_right, dist_right,
                                R, T_dir, left_matched, right_raw_dots,
                                pts_data, img_shape):
    """Find optimal translation scale using PtsCAL known distances.

    Sweeps over candidate baseline magnitudes (coarse then fine), at each
    scale cross-labeling the right camera dots and triangulating matched
    points to compute 3D distances. The scale that minimizes median distance
    error relative to PtsCAL ground truth is selected.

    Args:
        K_left, dist_left: Left camera intrinsics
        K_right, dist_right: Right camera intrinsics
        R: Stereo rotation matrix
        T_dir: Unit-norm translation direction
        left_matched: Left camera matched frames
        right_raw_dots: Right camera raw dot detections
        pts_data: PtsCAL data with 'points' and 'distances'
        img_shape: (width, height)

    Returns:
        (best_scale, median_error_pct, mean_error_pct) or (None, None, None)
    """
    from collections import defaultdict

    known_dists = pts_data.get('distances', [])
    if not known_dists:
        print("  WARNING: No known distances in PtsCAL data, cannot optimize scale")
        return None, None, None

    def evaluate_scale(scale):
        T = T_dir * scale
        rc = cross_label_right_dots(
            left_matched, K_left, dist_left, K_right, dist_right,
            R, T, pts_data, right_raw_dots, img_shape, max_dist=20.0)
        if len(rc) < 3:
            return 1000.0, []

        P_L = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_R = K_right @ np.hstack([R, T])

        tri = defaultdict(list)
        for f in sorted(set(left_matched.keys()) & set(rc.keys())):
            l_im, l_ob, l_la = left_matched[f]
            r_im, r_ob, r_la = rc[f]
            common = set(l_la) & set(r_la)
            lm = {la: i for i, la in enumerate(l_la)}
            rm = {la: i for i, la in enumerate(r_la)}
            for lb in common:
                lp = cv2.undistortPoints(
                    l_im[lm[lb]:lm[lb]+1].astype(np.float64),
                    K_left, dist_left, P=K_left)
                rp = cv2.undistortPoints(
                    r_im[rm[lb]:rm[lb]+1].astype(np.float64),
                    K_right, dist_right, P=K_right)
                p4 = cv2.triangulatePoints(
                    P_L, P_R,
                    lp.reshape(2, 1).astype(np.float64),
                    rp.reshape(2, 1).astype(np.float64))
                tri[lb].append((p4[:3] / p4[3]).ravel())

        avg = {lb: np.median(np.array(ps), axis=0) for lb, ps in tri.items()}
        errors = []
        for l1, l2, gt in known_dists:
            if l1 in avg and l2 in avg:
                m = np.linalg.norm(avg[l2] - avg[l1])
                errors.append(abs(100.0 * (m - gt) / gt))

        if not errors:
            return 1000.0, errors
        return np.median(errors), errors

    # Estimate reasonable scale range from known distances
    # The baseline is typically 5-50% of the target-to-camera distance
    # Use a wide range to be safe
    print("  Coarse scale search...")
    scales = np.linspace(50, 500, 90)
    best_scale = None
    best_err = float('inf')

    for s in scales:
        err, _ = evaluate_scale(s)
        if err < best_err:
            best_err = err
            best_scale = s

    if best_scale is None:
        print("  No valid scale found in coarse search")
        return None, None, None

    print(f"  Coarse: scale={best_scale:.0f}mm, median_err={best_err:.1f}%")

    # Fine search around best coarse scale
    print("  Fine scale search...")
    fine_scales = np.linspace(best_scale - 25, best_scale + 25, 100)
    for s in fine_scales:
        err, _ = evaluate_scale(s)
        if err < best_err:
            best_err = err
            best_scale = s

    # Get final error details
    _, errs_pct = evaluate_scale(best_scale)
    mean_err = np.mean(errs_pct) if errs_pct else 999.0
    print(f"  Fine: scale={best_scale:.1f}mm, median_err={best_err:.1f}%, "
          f"mean_err={mean_err:.1f}%")

    return best_scale, best_err, mean_err


def feature_based_stereo_calibration(left_path, right_path, input_path,
                                      K_left, dist_left,
                                      left_matched, right_raw_dots,
                                      pts_data, img_shape, frame_step=1):
    """Alternative stereo calibration using Essential matrix from natural features.

    When standard stereoCalibrate fails (due to inconsistent dot-to-PtsCAL
    matching between cameras), this approach:
      1. Estimates R, T direction from SIFT/ORB features across stereo frames
      2. Optimizes translation scale using PtsCAL known distances
      3. Cross-labels right camera dots via left PnP + stereo projection
      4. Optionally re-calibrates right camera with cross-labeled correspondences

    This bypasses the problematic independent label matching for each camera.

    Args:
        left_path, right_path: Separate stereo paths (or None)
        input_path: Stitched video path (or None)
        K_left, dist_left: Calibrated left camera intrinsics
        left_matched: Left camera {frame: (imgpts, objpts, labels)}
        right_raw_dots: Right camera {frame: Nx1x2 raw dot detections}
        pts_data: PtsCAL data
        img_shape: (width, height)
        frame_step: Frame step for video processing

    Returns:
        dict with 'K_right', 'dist_right', 'R', 'T', 'rms_stereo',
        'mean_dist_err', 'median_dist_err', 'baseline_mm', or None on failure
    """
    print("\n" + "-" * 60)
    print("Feature-based stereo calibration (Essential matrix approach)")
    print("-" * 60)

    # Use left camera intrinsics as initial right camera estimate
    K_right = K_left.copy()
    dist_right = dist_left.copy()

    # Phase 1: Estimate R, T direction from feature matches
    print("\nPhase 1: Essential matrix estimation from stereo frames")
    all_estimates = estimate_essential_from_stereo_frames(
        left_path, right_path, input_path,
        K_left, dist_left, K_right, dist_right,
        frame_step=max(frame_step, 5), max_samples=60)

    if len(all_estimates) < 5:
        print("  Insufficient Essential matrix estimates, aborting")
        return None

    R, T_dir = robust_consensus_RT(all_estimates)
    if R is None:
        print("  Failed to compute consensus R, T")
        return None

    print(f"  R diagonal: [{R[0,0]:.6f}, {R[1,1]:.6f}, {R[2,2]:.6f}]")
    print(f"  T direction: [{T_dir[0,0]:.4f}, {T_dir[1,0]:.4f}, {T_dir[2,0]:.4f}]")

    # Phase 2: Optimize translation scale
    print("\nPhase 2: Translation scale optimization")
    best_scale, median_err, mean_err = optimize_translation_scale(
        K_left, dist_left, K_right, dist_right,
        R, T_dir, left_matched, right_raw_dots, pts_data, img_shape)

    if best_scale is None:
        print("  Scale optimization failed, aborting")
        return None

    T = T_dir * best_scale
    baseline = np.linalg.norm(T)
    print(f"  T: [{T[0,0]:.2f}, {T[1,0]:.2f}, {T[2,0]:.2f}]")
    print(f"  Baseline: {baseline:.2f} mm")

    # Phase 3: Cross-label right camera and re-calibrate
    print("\nPhase 3: Cross-labeling right camera dots")
    right_cross = cross_label_right_dots(
        left_matched, K_left, dist_left, K_right, dist_right,
        R, T, pts_data, right_raw_dots, img_shape)

    if len(right_cross) < 3:
        print(f"  Only {len(right_cross)} frames cross-labeled, aborting")
        return None

    avg_matches = np.mean([len(r[2]) for r in right_cross.values()])
    print(f"  Cross-labeled {len(right_cross)} frames, avg {avg_matches:.0f} matches/frame")

    # Re-calibrate right camera with cross-labeled correspondences
    print("\nPhase 4: Re-calibrating right camera with cross-labeled points")
    right_sorted = sorted(right_cross.keys())
    right_objpoints = [right_cross[f][1] for f in right_sorted]
    right_imgpoints = [right_cross[f][0] for f in right_sorted]

    try:
        (right_rms, K_right_new, dist_right_new, _, _), _ = calibrate_single_camera(
            right_imgpoints, right_objpoints, img_shape, "right (cross-labeled)")
        print(f"  Right camera RMS: {right_rms:.4f} px")

        # Re-estimate E with updated right intrinsics
        print("\nPhase 5: Re-estimating E with updated right intrinsics")
        all_estimates2 = estimate_essential_from_stereo_frames(
            left_path, right_path, input_path,
            K_left, dist_left, K_right_new, dist_right_new,
            frame_step=max(frame_step, 5), max_samples=60)

        if len(all_estimates2) >= 5:
            R2, T_dir2 = robust_consensus_RT(all_estimates2)
            if R2 is not None:
                # Re-optimize scale with updated intrinsics
                best_scale2, median_err2, mean_err2 = optimize_translation_scale(
                    K_left, dist_left, K_right_new, dist_right_new,
                    R2, T_dir2, left_matched, right_raw_dots, pts_data, img_shape)

                if best_scale2 is not None and median_err2 < median_err:
                    R, T_dir, T = R2, T_dir2, T_dir2 * best_scale2
                    K_right, dist_right = K_right_new, dist_right_new
                    median_err, mean_err, best_scale = median_err2, mean_err2, best_scale2
                    baseline = np.linalg.norm(T)
                    print(f"  Improved: median_err={median_err:.1f}%, baseline={baseline:.1f}mm")
                else:
                    K_right, dist_right = K_right_new, dist_right_new
                    print(f"  Using updated intrinsics (scale not improved)")
            else:
                K_right, dist_right = K_right_new, dist_right_new
        else:
            K_right, dist_right = K_right_new, dist_right_new
    except (ValueError, cv2.error) as e:
        print(f"  Right re-calibration failed ({e}), using initial estimate")

    return {
        'K_right': K_right,
        'dist_right': dist_right,
        'R': R,
        'T': T,
        'rms_stereo': -1.0,  # Not from stereoCalibrate
        'mean_dist_err': mean_err,
        'median_dist_err': median_err,
        'baseline_mm': float(baseline),
    }


def main():
    description = "Estimate stereo calibration from calibration target images."
    epilog = """
Input modes:
  1. Stitched stereo (default): Provide a single video file or image
     glob pattern where left and right images are horizontally
     concatenated (left on left half, right on right half).
     Example: %(prog)s calibration_video.mp4
     Example: %(prog)s 'calibration_images/*.png'

  2. Separate stereo: Use --left and --right options to specify
     separate paths for left and right camera images. Each path
     can be a video file, directory, or glob pattern.
     Example: %(prog)s --left left_video.mp4 --right right_video.mp4
     Example: %(prog)s --left ./left_images/ --right ./right_images/
     Example: %(prog)s --left 'left/*.png' --right 'right/*.png'
"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", nargs='?', default=None,
                        help="stitched video or image path/glob pattern")
    parser.add_argument("-l", "--left", default=None, dest="left_path",
                        help="left camera images (video, directory, or glob pattern)")
    parser.add_argument("-r", "--right", default=None, dest="right_path",
                        help="right camera images (video, directory, or glob pattern)")
    parser.add_argument("-b", "--bayer", action="store_true", default=False,
                        help="input images are Bayer patterned")
    parser.add_argument("-c", "--corners-file", default=None, dest="corners_file",
                        help="corner file input/output path for caching detections")
    parser.add_argument("-i", "--intr-file", default=None, dest="intr_file",
                        help="input intrinsics file if only recomputing extrinsics")
    parser.add_argument("--pts", default=None, metavar="PATH",
                        help="PtsCAL file providing 3D reference point coordinates (mm). "
                             "When provided, these replace the chessboard grid object points "
                             "and --square-size is ignored. Grid size must match point arrangement.")
    parser.add_argument("-q", "--square-size", type=float, default=80,
                        help="width of a single calibration square in mm (default: 80)")
    parser.add_argument("-x", "--grid-x", type=int, default=6,
                        help="number of inner corners in grid width (default: 6)")
    parser.add_argument("-y", "--grid-y", type=int, default=5,
                        help="number of inner corners in grid height (default: 5)")
    parser.add_argument("-d", "--dots", action="store_true", default=False,
                        help="detect white circular dots instead of chessboard corners")
    parser.add_argument("-a", "--auto-grid", action="store_true", default=False,
                        help="automatically detect grid size from first image")
    parser.add_argument("-s", "--frame-step", type=int, default=1,
                        help="process every Nth frame (default: 1)")
    parser.add_argument("-m", "--max-frames", type=int, default=0,
                        help="maximum number of frames to use for calibration (0=unlimited)")
    parser.add_argument("-g", "--gui", action="store_true", default=False,
                        help="visualize detection results in a GUI")
    parser.add_argument("-o", "--output", default="calibration.json", dest="json_file",
                        help="output json file path (default: calibration.json)")
    parser.add_argument("-n", "--npz", default=None, dest="npz_file",
                        help="optional npz output file")
    parser.add_argument("--roi-left", default=None, metavar="X1,Y1,X2,Y2",
                        help="only keep left camera detections within this image region "
                             "(x1,y1 top-left, x2,y2 bottom-right)")
    parser.add_argument("--roi-right", default=None, metavar="X1,Y1,X2,Y2",
                        help="only keep right camera detections within this image region "
                             "(x1,y1 top-left, x2,y2 bottom-right)")

    args = parser.parse_args()

    # Parse ROIs if provided
    def parse_roi_arg(roi_str, name):
        if roi_str is None:
            return None
        try:
            parts = [int(x.strip()) for x in roi_str.split(',')]
            if len(parts) != 4:
                raise ValueError
            print(f"Detection ROI ({name}): ({parts[0]},{parts[1]}) to ({parts[2]},{parts[3]})")
            return tuple(parts)
        except ValueError:
            parser.error(f"--roi-{name} must be four comma-separated integers: X1,Y1,X2,Y2")

    left_roi = parse_roi_arg(args.roi_left, "left")
    right_roi = parse_roi_arg(args.roi_right, "right")

    # Determine input mode
    use_separate_stereo = args.left_path is not None or args.right_path is not None

    if use_separate_stereo:
        if args.left_path is None or args.right_path is None:
            parser.error("Both --left and --right must be specified for separate stereo mode")
        if args.input is not None:
            parser.error("Cannot specify positional argument with --left/--right options")
    else:
        if args.input is None:
            parser.error("Must specify either a stitched video/image path or --left and --right options")

    grid_size = (args.grid_x, args.grid_y)

    # Parse PtsCAL early so it's available for GUI visualization during detection
    gui_pts_data = None
    if args.pts and args.dots and args.gui:
        gui_pts_data = parse_ptscal(args.pts)

    img_shape = None
    if args.corners_file:
        if os.path.exists(args.corners_file):
            data = np.load(args.corners_file, allow_pickle=True)
            img_shape = tuple(data["img_shape"])
            left_data = data["left_data"].item()
            right_data = data["right_data"].item()
            # Load saved grid size if available (for auto-detection)
            if "grid_size" in data:
                grid_size = tuple(data["grid_size"])

    left_sizes_dict = {}
    right_sizes_dict = {}

    if img_shape is None:
        if use_separate_stereo:
            (img_shape, left_data, right_data, detected_grid_size,
             left_sizes_dict, right_sizes_dict) = detect_grid_stereo_separate(
                args.left_path, args.right_path, grid_size,
                args.frame_step, args.gui, args.bayer, args.auto_grid, args.max_frames,
                args.dots, gui_pts_data, left_roi, right_roi)
        else:
            (img_shape, left_data, right_data, detected_grid_size,
             left_sizes_dict, right_sizes_dict) = detect_grid_video(
                args.input, grid_size, args.frame_step,
                args.gui, args.bayer, args.auto_grid, args.max_frames,
                args.dots, gui_pts_data, left_roi, right_roi)
        # Use detected grid size if auto-detection was enabled
        if args.auto_grid and detected_grid_size != grid_size:
            grid_size = detected_grid_size
        if args.corners_file:
            np.savez(args.corners_file, img_shape=img_shape,
                     left_data=left_data, right_data=right_data,
                     grid_size=grid_size)

    # Validate we have enough detections
    MIN_DETECTIONS = 3
    if len(left_data) < MIN_DETECTIONS:
        raise ValueError(f"Insufficient left camera detections: {len(left_data)} "
                         f"(minimum {MIN_DETECTIONS} required)")
    if len(right_data) < MIN_DETECTIONS:
        raise ValueError(f"Insufficient right camera detections: {len(right_data)} "
                         f"(minimum {MIN_DETECTIONS} required)")

    # Check for common frames (needed for stereo calibration)
    common_frames = set(left_data.keys()).intersection(set(right_data.keys()))
    if len(common_frames) < MIN_DETECTIONS:
        raise ValueError(f"Insufficient common stereo detections: {len(common_frames)} "
                         f"(minimum {MIN_DETECTIONS} required). "
                         f"Left has {len(left_data)}, right has {len(right_data)} detections.")

    print("computing calibration")
    use_feature_based = False
    feat_result = None
    if args.dots and args.pts:
        # Dots + PtsCAL mode: match each frame's dots to 3D world points
        ptscal_data = parse_ptscal(args.pts)
        print(f"Loaded {len(ptscal_data['points'])} PtsCAL reference points")

        # Match dots to world points with frame-to-frame tracking for consistency
        print("Matching left camera dots to world points...")
        left_matched = track_and_match_all_frames(
            left_data, left_sizes_dict, ptscal_data, img_shape, "left")

        print("Matching right camera dots to world points...")
        right_matched = track_and_match_all_frames(
            right_data, right_sizes_dict, ptscal_data, img_shape, "right")

        MIN_MATCHED = 3
        if len(left_matched) < MIN_MATCHED:
            raise ValueError(f"Insufficient left matched frames: {len(left_matched)} "
                             f"(minimum {MIN_MATCHED} required)")
        if len(right_matched) < MIN_MATCHED:
            raise ValueError(f"Insufficient right matched frames: {len(right_matched)} "
                             f"(minimum {MIN_MATCHED} required)")

        # Build per-frame object/image point lists for single-camera calibration
        left_sorted = sorted(left_matched.keys())
        left_objpoints = [left_matched[f][1] for f in left_sorted]
        left_imgpoints = [left_matched[f][0] for f in left_sorted]

        right_sorted = sorted(right_matched.keys())
        right_objpoints = [right_matched[f][1] for f in right_sorted]
        right_imgpoints = [right_matched[f][0] for f in right_sorted]

        (left_rms, K_left, dist_left, _, _), _ = calibrate_single_camera(
            left_imgpoints, left_objpoints, img_shape, "left")
        (right_rms, K_right, dist_right, _, _), _ = calibrate_single_camera(
            right_imgpoints, right_objpoints, img_shape, "right")

        # write the intrinsics file
        if not args.intr_file:
            fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
            if fs.isOpened():
                fs.write("M1", K_left)
                fs.write("D1", dist_left)
                fs.write("M2", K_right)
                fs.write("D2", dist_right)
            fs.release()
        else:
            npz_dict = dict(np.load(args.intr_file))
            K_left = npz_dict['cameraMatrixL']
            K_right = npz_dict['cameraMatrixR']
            dist_left = npz_dict['distCoeffsL']
            dist_right = npz_dict['distCoeffsR']

        # For stereo: intersect matched labels per frame
        stereo_frames = sorted(set(left_matched.keys()) & set(right_matched.keys()))
        if len(stereo_frames) < MIN_MATCHED:
            raise ValueError(f"Insufficient common stereo frames with matches: "
                             f"{len(stereo_frames)} (minimum {MIN_MATCHED} required)")

        stereo_objpoints = []
        stereo_left_pts = []
        stereo_right_pts = []
        for f in stereo_frames:
            l_img, l_obj, l_labels = left_matched[f]
            r_img, r_obj, r_labels = right_matched[f]
            common_labels = set(l_labels) & set(r_labels)
            if len(common_labels) < 6:
                continue
            # Filter both to common labels only
            l_indices = [i for i, lab in enumerate(l_labels) if lab in common_labels]
            r_index_map = {lab: i for i, lab in enumerate(r_labels)}
            # Reorder right to match left's label order for common labels
            common_l_labels = [l_labels[i] for i in l_indices]
            r_indices = [r_index_map[lab] for lab in common_l_labels]

            stereo_objpoints.append(l_obj[l_indices])
            stereo_left_pts.append(l_img[l_indices])
            stereo_right_pts.append(r_img[r_indices])

        if len(stereo_objpoints) < MIN_MATCHED:
            raise ValueError(f"Insufficient stereo frames after label intersection: "
                             f"{len(stereo_objpoints)} (minimum {MIN_MATCHED} required)")

        print(f"Stereo calibration using {len(stereo_objpoints)} frames with common matches")
        print_progress(1, 2, prefix='Stereo calibration', suffix='Computing extrinsics')
        ret = cv2.stereoCalibrate(stereo_objpoints, stereo_left_pts, stereo_right_pts,
                                  K_left, dist_left, K_right, dist_right, img_shape,
                                  flags=cv2.CALIB_FIX_INTRINSIC)
        frames = stereo_frames  # for summary output
        standard_rms = ret[0]
        print(f"\nStandard stereoCalibrate RMS: {standard_rms:.2f} pixels")

        # Also try feature-based Essential matrix approach as alternative.
        # This is especially useful when dot-to-PtsCAL matching is inconsistent
        # between cameras (different label assignments for same physical dots).
        feat_result = None
        high_rms_threshold = 20.0  # stereoCalibrate RMS indicating poor matching
        try:
            feat_result = feature_based_stereo_calibration(
                left_path=args.left_path, right_path=args.right_path,
                input_path=args.input,
                K_left=K_left, dist_left=dist_left,
                left_matched=left_matched, right_raw_dots=right_data,
                pts_data=ptscal_data, img_shape=img_shape,
                frame_step=args.frame_step)
        except Exception as e:
            print(f"Feature-based calibration failed: {e}")

        use_feature_based = False
        if feat_result is not None:
            fb_median = feat_result['median_dist_err']
            fb_baseline = feat_result['baseline_mm']
            print(f"\nComparing calibration approaches:")
            print(f"  Standard stereoCalibrate: RMS={standard_rms:.2f}px")
            print(f"  Feature-based:            median_dist_err={fb_median:.1f}%, "
                  f"baseline={fb_baseline:.1f}mm")

            # Use feature-based if standard RMS is high (bad matching),
            # or if feature-based produces reasonable distance errors
            if standard_rms > high_rms_threshold:
                use_feature_based = True
                print(f"  -> Using feature-based (standard RMS {standard_rms:.1f} > "
                      f"{high_rms_threshold} threshold)")
            elif fb_median < 30.0 and standard_rms > 5.0:
                use_feature_based = True
                print(f"  -> Using feature-based (median distance error "
                      f"{fb_median:.1f}% with lower stereo RMS)")
            else:
                print(f"  -> Using standard stereoCalibrate")

        if use_feature_based:
            K_right = feat_result['K_right']
            dist_right = feat_result['dist_right']
            right_rms = -1.0  # Not available from feature-based approach
            # Override stereoCalibrate result with feature-based R, T
            ret = list(ret)
            ret[0] = feat_result['rms_stereo']
            ret[5] = feat_result['R']
            ret[6] = feat_result['T']
            ret = tuple(ret)
            # Re-write intrinsics.yml with updated right camera intrinsics
            if not args.intr_file:
                fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
                if fs.isOpened():
                    fs.write("M1", K_left)
                    fs.write("D1", dist_left)
                    fs.write("M2", K_right)
                    fs.write("D2", dist_right)
                fs.release()
    else:
        # Checkerboard mode (existing path) or pts-only with checkerboard
        if args.pts:
            ptscal_data = parse_ptscal(args.pts)
            pts_3d = np.array(list(ptscal_data['points'].values()), dtype=np.float32)
            n_pts = grid_size[0] * grid_size[1]
            if len(pts_3d) != n_pts:
                raise ValueError(
                    f"PtsCAL has {len(pts_3d)} points but grid size "
                    f"{grid_size[0]}x{grid_size[1]} expects {n_pts} points. "
                    f"Adjust --grid-x/--grid-y to match.")
            objp = pts_3d
            print(f"Using {len(pts_3d)} PtsCAL reference points as object points")
        else:
            objp = make_object_points(grid_size) * args.square_size
        (left_rms, K_left, dist_left, _, _), _ = calibrate_single_camera(
            left_data, objp, img_shape, "left")
        (right_rms, K_right, dist_right, _, _), _ = calibrate_single_camera(
            right_data, objp, img_shape, "right")

        # write the intrinsics file
        if not args.intr_file:
            fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
            if fs.isOpened():
                fs.write("M1", K_left)
                fs.write("D1", dist_left)
                fs.write("M2", K_right)
                fs.write("D2", dist_right)
            fs.release()
        else:
            npz_dict = dict(np.load(args.intr_file))
            K_left = npz_dict['cameraMatrixL']
            K_right = npz_dict['cameraMatrixR']
            dist_left = npz_dict['distCoeffsL']
            dist_right = npz_dict['distCoeffsR']

        # find frames that detected the target in both left and right views
        frames = set(left_data.keys()).intersection(set(right_data.keys()))
        left_points = [left_data[f] for f in frames]
        right_points = [right_data[f] for f in frames]
        objpoints = [objp] * len(frames)

        print_progress(1, 2, prefix='Stereo calibration', suffix='Computing extrinsics')
        ret = cv2.stereoCalibrate(objpoints, left_points, right_points,
                                  K_left, dist_left, K_right, dist_right, img_shape,
                                  flags=cv2.CALIB_FIX_INTRINSIC)
    stereo_rms_error = ret[0]
    R, T = ret[5:7]

    print_progress(2, 2, prefix='Stereo calibration', suffix='Computing rectification')
    ret2 = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, img_shape, R, T,
                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q = ret2[:5]
    print_progress(2, 2, prefix='Stereo calibration', suffix='Done')

    # write the extrinsics file
    fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_WRITE)
    if (fs.isOpened()):
        fs.write("R", R)
        fs.write("T", T)
        fs.write("R1", R1)
        fs.write("R2", R2)
        fs.write("P1", P1)
        fs.write("P2", P2)
        fs.write("Q", Q)
    fs.release()

    # write KWIVER camera_rig_io-compatible json file (default output)
    json_dict = dict()

    # Image dimensions and grid
    json_dict['image_width'] = img_shape[0]
    json_dict['image_height'] = img_shape[1]
    json_dict['grid_width'] = grid_size[0]
    json_dict['grid_height'] = grid_size[1]
    json_dict['square_size_mm'] = args.square_size

    # Calibration quality metrics
    json_dict['rms_error_left'] = float(left_rms)
    json_dict['rms_error_right'] = float(right_rms)
    json_dict['rms_error_stereo'] = float(stereo_rms_error)

    # Distance error metrics (from feature-based approach, if used)
    if args.dots and args.pts and use_feature_based and feat_result is not None:
        json_dict['mean_distance_error_pct'] = float(feat_result['mean_dist_err'])
        json_dict['median_distance_error_pct'] = float(feat_result['median_dist_err'])
        json_dict['baseline_mm'] = float(feat_result['baseline_mm'])

    # Extrinsics
    json_dict['T'] = T.flatten().tolist()
    json_dict['R'] = R.flatten().tolist()

    # Intrinsics and distortion for each camera
    for (m, d, side) in ([K_left, dist_left, 'left'], [K_right, dist_right, 'right']):
        json_dict[f'fx_{side}'] = float(m[0][0])
        json_dict[f'fy_{side}'] = float(m[1][1])
        json_dict[f'cx_{side}'] = float(m[0][2])
        json_dict[f'cy_{side}'] = float(m[1][2])

        # Flatten distortion coefficients to handle various array shapes
        d_flat = d.flatten()
        json_dict[f'k1_{side}'] = float(d_flat[0]) if len(d_flat) > 0 else 0.0
        json_dict[f'k2_{side}'] = float(d_flat[1]) if len(d_flat) > 1 else 0.0
        json_dict[f'p1_{side}'] = float(d_flat[2]) if len(d_flat) > 2 else 0.0
        json_dict[f'p2_{side}'] = float(d_flat[3]) if len(d_flat) > 3 else 0.0
        json_dict[f'k3_{side}'] = float(d_flat[4]) if len(d_flat) > 4 else 0.0

    with open(args.json_file, 'w') as fh:
        fh.write(json.dumps(json_dict, indent=2))

    # optionally write npz file
    if args.npz_file:
        npz_dict = dict()
        npz_dict['cameraMatrixL'] = K_left
        npz_dict['cameraMatrixR'] = K_right
        npz_dict['distCoeffsL'] = dist_left
        npz_dict['distCoeffsR'] = dist_right
        npz_dict['R'] = R
        npz_dict['T'] = T
        np.savez(args.npz_file, **npz_dict)

    # Print summary statistics
    baseline = np.linalg.norm(T)
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Image size:                 {img_shape[0]} x {img_shape[1]}")
    if args.dots and args.pts:
        mode = "dots + PtsCAL"
        if use_feature_based:
            mode += " (feature-based stereo)"
        print(f"Mode:                       {mode}")
        print(f"PtsCAL points:              {len(ptscal_data['points'])}")
    else:
        grid_label = "Grid size (auto-detected):" if args.auto_grid else "Grid size:"
        print(f"{grid_label:27} {grid_size[0]} x {grid_size[1]}")
        print(f"Square size:                {args.square_size} mm")
    print("-" * 60)
    print(f"Left camera detections:     {len(left_data)}")
    print(f"Right camera detections:    {len(right_data)}")
    print(f"Common stereo detections:   {len(frames)}")
    print("-" * 60)
    print(f"Left camera RMS error:      {left_rms:.4f} pixels")
    print(f"Right camera RMS error:     {right_rms:.4f} pixels")
    print(f"Stereo RMS error:           {stereo_rms_error:.4f} pixels")
    print(f"Baseline distance:          {baseline:.2f} mm")
    if args.dots and args.pts and use_feature_based and feat_result is not None:
        print(f"Distance error (mean):      {feat_result['mean_dist_err']:.1f}%")
        print(f"Distance error (median):    {feat_result['median_dist_err']:.1f}%")
    print("-" * 60)
    print("Output files:")
    print(f"  - {args.json_file}")
    print(f"  - intrinsics.yml")
    print(f"  - extrinsics.yml")
    if args.npz_file:
        print(f"  - {args.npz_file}")
    if args.corners_file:
        print(f"  - {args.corners_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
