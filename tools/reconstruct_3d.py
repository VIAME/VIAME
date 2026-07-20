#!/usr/bin/env python3
"""
reconstruct_3d.py - Convert UAS drone imagery folders into 3D mesh models.

Pipeline:
  1. SIFT feature extraction (pycolmap)
  2. Feature matching (exhaustive <=50 images, sequential otherwise)
  3. Incremental Structure from Motion -> sparse point cloud + cameras
  4. Densification (one of two methods):                                [skip with --no-dense]
     a. --dense-method sift (default): OpenCV SIFT matching + triangulation (CPU)
     b. --dense-method mvs: COLMAP PatchMatch stereo + fusion (requires CUDA)
  5. Poisson surface reconstruction (Open3D) -> triangle mesh          [skip with --no-dense]
  6. Export PLY point cloud + PLY/OBJ mesh

Multi-camera rigs (PORT/STAR/CENTER subfolders) are auto-detected and
reconstructed jointly.

NOTE: 2D planar prior-coverage / registration ("--planar" in older versions)
now lives in detect_prior_coverage.py, which also handles revisit detection,
cross-camera overlap and rig-constrained SfM coverage (--method sfm-rig).
This tool is purely for 3D reconstruction.

Usage:
  python reconstruct_3d.py --install-deps          # check/install dependencies
  python reconstruct_3d.py <image_folder> [--output <output_dir>] [--scale <0.25>]
  python reconstruct_3d.py <image_folder> --dense-method mvs   # use COLMAP MVS (GPU)
  python reconstruct_3d.py --all                    # process all subfolders
"""

import os
import sys
import argparse
import time
import subprocess
import importlib

# Shared registration engine (image listing, multicam detection) lives in the
# OpenCV plugin so other tools can reuse it.
from viame.opencv import registration_utils as _sr
from viame.opencv.registration_utils import get_image_files, detect_multicam


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

# Map of import name -> pip package name. numpy/cv2 are needed for everything;
# pycolmap/open3d are needed for the SfM / dense (MVS) reconstruction modes.
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'cv2': 'opencv-python',
}
OPTIONAL_PACKAGES = {
    'pycolmap': 'pycolmap',
    'open3d': 'open3d-cpu (Linux) or open3d (Windows/Mac)',
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
        print("\nInstall them with pip (pycolmap also ships with "
              "-DVIAME_ENABLE_COLMAP=ON builds; open3d is never bundled "
              "due to its size). "
              "For 2D planar coverage/registration use detect_prior_coverage.py, "
              "which does not need COLMAP.")
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
# Multi-camera support
# ---------------------------------------------------------------------------

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


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
# COLMAP CLI validation (MVS mode)
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
# Pipeline
# ---------------------------------------------------------------------------

def process_folder(image_folder, output_dir, scale=0.25, max_pairs_per_image=3,
                    dense=True, dense_method="sift", multicam=False,
                    sfm_matching='auto'):
    """Full 3D-reconstruction pipeline for one image folder.

    If multicam=True (or auto-detected), looks for PORT/STAR/CENTER subfolders
    and processes all cameras jointly in a single SfM reconstruction.
    """
    folder_name = os.path.basename(image_folder.rstrip('/'))

    # Auto-detect multicam
    if not multicam and detect_multicam(image_folder):
        print(f"  Auto-detected multi-camera layout (PORT/STAR/CENTER)")
        multicam = True

    # The SfM implementation lives in the (optional) viame.colmap plugin.
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
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UAS Imagery -> 3D Model",
        epilog="For 2D planar prior-coverage, registration and revisit "
               "detection use detect_prior_coverage.py instead.")
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
                             "(steps 4-6); only run SfM")
    parser.add_argument("--multicam", action="store_true",
                        help="Input folder has PORT/STAR/CENTER subfolders "
                             "(auto-detected if present)")
    parser.add_argument("--sfm-matching", choices=['auto', 'exhaustive', 'sequential'],
                        default='auto',
                        help="SfM feature-matching strategy. 'exhaustive' finds "
                             "temporally-distant loop-closure pairs (revisits); "
                             "'auto' uses it for small/multicam sets only.")
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
    for folder in folders:
        name = os.path.basename(folder.rstrip('/'))
        out = args.output or os.path.join(base_dir, "3d_models", name)
        ok = process_folder(folder, out, scale=args.scale,
                            max_pairs_per_image=args.max_pairs,
                            dense=not args.no_dense,
                            dense_method=args.dense_method,
                            multicam=args.multicam,
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
