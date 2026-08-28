# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Build a pixel-stitched mosaic for a survey site without the compiled
KWIVER stabilization pipeline (``many_image_stabilizer`` / ``kw_write_homography``).

Reuses the same registration engine as ``detect_prior_coverage.py``
(``viame.opencv.registration_utils``) to build a per-camera homography chain,
GPS-anchors/fills frames that failed direct registration, then runs a
pose-graph optimization pass using ``detect_loop_edges`` /
``optimize_pose_graph`` so temporally-distant revisits (loop closures) pull
the chain back into alignment instead of drifting. The resulting per-camera
homographies are written in ``create_mosaic.py``'s file format and stitched
with it.

Usage::

    build_pixel_mosaic.py SITE_FOLDER --flight-logs DIR --output OUT_DIR
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Source JPEGs in this dataset are slightly truncated (trailing bytes missing,
# harmless -- OpenCV loads them with just a warning) but PIL/skimage, which
# create_mosaic.py reads images with, raises OSError on them by default.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from viame.core import survey_metadata as smd
import create_mosaic as cm
from viame.opencv import registration_utils as ru
from viame.opencv.registration_utils import (
    compute_homography_pair,
    detect_loop_edges,
    optimize_pose_graph,
)

def _find_anchor(chain):
    """Recover the anchor index _compute_camera_chain seeded with identity."""
    return min(chain.keys(), key=lambda k: np.linalg.norm(chain[k] - np.eye(3)))


def build_camera_chain(site_folder, cam, rels, water_info, args):
    """Return (final_chain: index -> H_to_anchor, anchor_idx, n_loop_edges, pairwise_H)."""
    # Affine (not full homography) chains: perspective terms compound over
    # 100+-frame chains and drive the far end's scale toward zero (observed:
    # median frame scale 0.02 on PINNACLE ROCK), which renders as a smear.
    # detect_prior_coverage.py uses affine for the same reason.
    ch, pw = ru._compute_camera_chain(
        site_folder, rels, label=str(cam), water_info=water_info,
        match_ratio=args.match_ratio, min_inliers=args.min_inliers,
        scale=args.match_scale, use_affine=True, consistency_filter=True,
    )
    if not ch:
        return {}, None, 0, {}

    anchor_idx = _find_anchor(ch)

    loop_edges = []
    if len(ch) >= 3 and not args.no_loop_closure:
        loop_edges = detect_loop_edges(
            ch, site_folder, rels, min_gap=args.loop_min_gap,
            overlap_thresh=args.loop_overlap_thresh,
        )

    seq_edges = [(i, j, H) for (i, j), H in pw.items()]
    corrected = optimize_pose_graph(
        ch, seq_edges, loop_edges=loop_edges, anchor=anchor_idx,
    )
    print(f'    {cam}: pose graph = {len(seq_edges)} sequential edges, '
          f'{len(loop_edges)} loop-closure edges')
    return corrected, anchor_idx, len(loop_edges), pw


def gps_fill_gaps(chains, cams, poses_by_cam, pairwise_by_cam):
    """Fill frames with no direct registration via GPS dead-reckoning, sharing
    the rig's metres-to-pixels scale (same approach detect_prior_coverage.py
    uses), so create_mosaic.py has a full-length homography file per camera.
    No-op (frames stay dropped) for sites without flight-log/EXIF GPS.
    """
    if not any(poses_by_cam.get(cam) for cam in cams):
        print('    No GPS metadata available; unregistered frames stay dropped')
        return
    ru._geo_anchor_cameras(chains, cams, poses_by_cam, pairwise_by_cam)


def write_homog_file(path, chain, n):
    """Write chain (index -> H in cv2 x,y convention) in create_mosaic.py's
    on-disk format. create_mosaic.read_homog_file conjugates each matrix by
    SWAP_XY on read (``swap_xy @ M_file @ swap_xy``) to get its own internal
    Y,X-ordered representation, so the FILE itself must hold the raw cv2
    x,y-convention matrix unmodified (conjugating twice would cancel out and
    silently swap the mosaic's X/Y axes) -- write `chain[i]` as-is.

    Frames missing from chain are dropped; caller must also drop the
    matching lines from the image list so indices realign. `tof` is only
    used by create_mosaic.py as a same-batch consistency tag (across ALL
    cameras of a multi-cam mosaic), not for any computation, so a shared
    constant (0) is used for every line.

    ALL chained frames are kept, INCLUDING open water: this is a sea-lion
    survey and animals swimming in open water must appear in the mosaic.
    """
    kept = [i for i in range(n) if i in chain]
    with open(path, 'w') as f:
        for line_no, i in enumerate(kept):
            vals = ' '.join(f'{x:.10g}' for x in chain[i].flatten())
            f.write(f'{vals} {line_no} 0\n')
    return kept


def write_image_list(path, site_folder, rels, kept_indices):
    with open(path, 'w') as f:
        for i in kept_indices:
            f.write(os.path.join(site_folder, rels[i]) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('site', help='Site folder (single camera, or rig with '
                    'PORT/CENTER/STAR subfolders)')
    ap.add_argument('--flight-logs', default=None,
                    help='FMCLOG CSV or directory of them')
    ap.add_argument('--output', required=True, help='Output directory')
    ap.add_argument('--water-method', choices=['auto', 'svm', 'sift'], default='auto')
    ap.add_argument('--match-ratio', type=float, default=0.80)
    ap.add_argument('--match-scale', type=float, default=0.5)
    ap.add_argument('--min-inliers', type=int, default=10)
    ap.add_argument('--loop-min-gap', type=int, default=15,
                    help='Minimum frame separation to consider a loop-closure edge')
    ap.add_argument('--loop-overlap-thresh', type=float, default=0.15)
    ap.add_argument('--no-loop-closure', action='store_true',
                    help='Disable loop-closure pose-graph correction (sequential-only chain)')
    ap.add_argument('--zoom', type=float, default=0.15,
                    help='Mosaic output scale factor (full-res 5168x3448 source frames '
                    'need a small value to keep memory/runtime reasonable)')
    ap.add_argument('--step', type=int, default=1, help='Draw every Nth frame')
    ap.add_argument('--optimize-fit', action='store_true',
                    help='Apply create_mosaic.py distortion-minimizing global fit. '
                    'Off by default: its 8-parameter projective search can '
                    'degenerate on long GPS-anchored multi-camera chains.')
    args = ap.parse_args()

    ru.import_dependencies()
    os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    site_tag = os.path.basename(os.path.normpath(args.site))
    print(f'=== {site_tag}: building pixel mosaic ===')

    records, cams = smd.build_image_records(
        args.site, flight_logs=args.flight_logs, read_exif=True)
    all_rels = [r for rels in cams.values() for r in rels]

    print(f'  Classifying water/land ({args.water_method})...')
    water_info = ru.classify_images_fast(args.site, all_rels, method=args.water_method)

    poses_by_cam = {
        cam: {r: records[r] for r in rels if records.get(r, {}).get('lat') is not None}
        for cam, rels in cams.items()
    }

    chains, anchors, pairwise_by_cam = {}, {}, {}
    total_loop_edges = 0
    print('  Computing per-camera homography chains + loop-closure pose graph...')
    for cam, rels in cams.items():
        chain, anchor_idx, n_loop, pw = build_camera_chain(args.site, cam, rels, water_info, args)
        if not chain:
            print(f'    {cam}: no frames chained, skipping camera')
            continue
        chains[cam], anchors[cam], pairwise_by_cam[cam] = chain, anchor_idx, pw
        total_loop_edges += n_loop

    if not chains:
        print('  No camera produced a usable chain; aborting.')
        return

    print('  GPS dead-reckoning fill for unregistered frames...')
    gps_fill_gaps(chains, {c: cams[c] for c in chains}, poses_by_cam, pairwise_by_cam)

    homog_and_lists = []
    for cam, rels in cams.items():
        if cam not in chains:
            continue
        chain = chains[cam]
        anchor_idx = anchors[cam]
        tag = cam or 'CAM'
        hpath = os.path.join(args.output, f'{tag}_homog.txt')
        lpath = os.path.join(args.output, f'{tag}_images.txt')
        kept = write_homog_file(hpath, chain, len(rels))
        write_image_list(lpath, args.site, rels, kept)
        print(f'    {cam}: {len(kept)}/{len(rels)} frames written '
              f'(anchor=#{anchor_idx})')
        homog_and_lists += [hpath, lpath]

    out_file = os.path.join(args.output, f'{site_tag}_mosaic.jpg')
    print(f'  Stitching mosaic -> {out_file} (zoom={args.zoom}, step={args.step})...')
    cm.main_multi(
        out_file, [(h, l) for h, l in zip(homog_and_lists[0::2], homog_and_lists[1::2])],
        optimize_fit=args.optimize_fit, zoom=args.zoom, step=args.step,
    )
    print(f'  Done: {total_loop_edges} loop-closure edges used, '
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
