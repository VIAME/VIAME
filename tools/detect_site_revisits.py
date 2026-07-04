#!/usr/bin/env python3
"""Detect site revisits (loop-closure candidates) in a survey image sequence.

A "revisit" is when the platform LEAVES a location and later RETURNS to image the
same ground again — the loop-closure events that matter for drift correction and
for flagging duplicate coverage. Two detection methods are provided:

  --method metadata      Uses per-frame GPS metadata only (fast, no image
                         matching). A revisit is a pair of frames that are close
                         on the ground but separated in time, with the platform
                         having travelled away in between (so slow/stationary
                         motion is not mistaken for a revisit). Metadata is read
                         via the same loaders as reconstruct_3d.py:
                         --flight-log CSV, an imagelog.json in the folder, or
                         embedded EXIF GPS.

  --method registration  Uses image registration. Builds the within-sequence
                         homography chain, projects each frame's ground
                         footprint, finds temporally-distant frames whose
                         footprints overlap, then CONFIRMS each candidate by a
                         direct feature match (a successful match between two
                         far-apart frames is a confirmed loop closure). Works
                         with no metadata at all.

  --method both          Run both and cross-reference (default when metadata is
                         present, else registration only).

Outputs a CSV of revisit events and prints a summary. Shares all geometry/IO
helpers with reconstruct_3d.py (imported as a library).
"""

import os
import sys
import csv
import argparse
import math

# Shared registration engine (metadata loaders, homography chain, geo-anchoring,
# dependency import) lives in the VIAME OpenCV plugin.
from viame.opencv import registration_utils as r3d


def _focal_px_from_exif(image_path, width):
    """Focal length in pixels from EXIF 35mm-equivalent focal length.
    focal_px = f35mm / 36mm * image_width. Returns None if unavailable."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        ex = {TAGS.get(k, k): v for k, v in (Image.open(image_path)._getexif() or {}).items()}
        f35 = ex.get('FocalLengthIn35mmFilm')
        if f35:
            return float(f35) / 36.0 * width
    except Exception:
        pass
    return None


def _list_images(image_folder):
    """Return (image_folder_for_metadata, ordered relative image paths). For a
    multicam rig the CENTER camera is used as the revisit reference."""
    if r3d.detect_multicam(image_folder):
        for cam in ('CENTER', 'PORT', 'STAR'):
            d = os.path.join(image_folder, cam)
            if os.path.isdir(d):
                imgs = sorted(f for f in os.listdir(d)
                              if os.path.splitext(f)[1].lower() in r3d.IMAGE_EXTS)
                return [os.path.join(cam, f) for f in imgs]
    return r3d.get_image_files(image_folder)


# ---------------------------------------------------------------------------
# Metadata method
# ---------------------------------------------------------------------------

def detect_metadata(image_folder, image_list, poses, args):
    """Find revisits from GPS positions. Returns list of event dicts."""
    np = r3d.np
    enu, _yaw = r3d._poses_to_enu(poses, image_list)
    n = len(image_list)

    # Ground footprint radius (m). Prefer EXIF-derived GSD * altitude; else use
    # --footprint-radius. half-width = altitude * 18 / f35 (independent of width).
    radius = args.footprint_radius
    if radius is None:
        f35 = None
        for rel in image_list:
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                ex = {TAGS.get(k, k): v for k, v in
                      (Image.open(os.path.join(image_folder, rel))._getexif() or {}).items()}
                if ex.get('FocalLengthIn35mmFilm'):
                    f35 = float(ex['FocalLengthIn35mmFilm']); break
            except Exception:
                continue
        alts = [poses[image_list[i]].get('alt_agl') for i in range(n)
                if not np.isnan(enu[i, 0]) and poses[image_list[i]].get('alt_agl')]
        alt = float(np.median([a for a in alts if a and a > 5])) if any(
            a and a > 5 for a in alts) else None
        if f35 and alt:
            radius = alt * 18.0 / f35
            print(f"  Footprint radius from EXIF/altitude: {radius:.0f} m "
                  f"(f35={f35:.0f}mm, alt={alt:.0f}m)")
        else:
            radius = 50.0
            print(f"  Footprint radius defaulting to {radius:.0f} m "
                  f"(no EXIF focal / altitude; set --footprint-radius)")

    overlap_dist = radius * 2.0 * (1.0 - args.overlap_thresh)  # closer => more overlap
    depart_dist = max(radius * args.departure_factor, overlap_dist * 1.5)

    valid = [i for i in range(n) if not np.isnan(enu[i, 0])]
    events = []
    used = set()
    for a_idx in range(len(valid)):
        i = valid[a_idx]
        for b_idx in range(a_idx + 1, len(valid)):
            j = valid[b_idx]
            if j - i < args.min_gap:
                continue
            d_ij = float(np.linalg.norm(enu[j] - enu[i]))
            if d_ij > overlap_dist:
                continue
            # Confirm the platform actually LEFT between i and j.
            between = [enu[k] for k in valid if i < k < j and not np.isnan(enu[k, 0])]
            if not between:
                continue
            max_depart = max(float(np.linalg.norm(p - enu[i])) for p in between)
            if max_depart < depart_dist:
                continue  # never went away -> slow/stationary, not a revisit
            if (i, j) in used:
                continue
            used.add((i, j))
            events.append({
                'frame_i': i, 'frame_j': j, 'gap': j - i,
                'distance_m': round(d_ij, 1),
                'max_departure_m': round(max_depart, 1),
                'method': 'metadata', 'confirmed': '',
            })
    return _dedupe_events(events)


# ---------------------------------------------------------------------------
# Registration method
# ---------------------------------------------------------------------------

def _footprint_corners(H, w, h):
    np = r3d.np
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return r3d.cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _poly_overlap(a, b):
    """Approximate overlap fraction of two convex quads (intersection / min area)."""
    cv2 = r3d.cv2; np = r3d.np
    a = a.astype(np.float32); b = b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter is None or inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a); area_b = cv2.contourArea(b)
    if min(area_a, area_b) < 1:
        return 0.0
    return float(inter / min(area_a, area_b))


def detect_registration(image_folder, image_list, args):
    """Find revisits by footprint overlap of the registered chain, confirmed by a
    direct feature match between the far-apart frames."""
    np = r3d.np
    print("  Building homography chain for footprints...")
    chain, _pw = r3d._compute_camera_chain(
        image_folder, image_list, label="frames", use_affine=args.affine,
        scale=args.match_scale, consistency_filter=False)
    reg = sorted(chain.keys())
    if len(reg) < 3:
        print("  Too few registered frames for registration-based detection.")
        return []
    # Footprint corners per registered frame
    dims = {}
    corners = {}
    for i in reg:
        p = os.path.join(image_folder, image_list[i])
        img = r3d.cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        dims[i] = (w, h)
        corners[i] = _footprint_corners(chain[i], w, h)

    candidates = []
    for ai in range(len(reg)):
        i = reg[ai]
        if i not in corners:
            continue
        for bi in range(ai + 1, len(reg)):
            j = reg[bi]
            if j - i < args.min_gap or j not in corners:
                continue
            ov = _poly_overlap(corners[i], corners[j])
            if ov >= args.overlap_thresh:
                candidates.append((i, j, ov))

    print(f"  {len(candidates)} footprint-overlap candidates; confirming by matching...")
    events = []
    for i, j, ov in candidates:
        confirmed = ''
        if not args.no_confirm:
            H, _ = r3d.compute_homography_pair(
                os.path.join(image_folder, image_list[i]),
                os.path.join(image_folder, image_list[j]),
                scale=min(args.match_scale * 1.5, 1.0), nfeatures=12000,
                use_affine=args.affine, min_inliers=args.min_inliers)
            confirmed = 'yes' if H is not None else 'no'
        events.append({
            'frame_i': i, 'frame_j': j, 'gap': j - i,
            'overlap': round(ov, 2), 'method': 'registration',
            'confirmed': confirmed,
        })
    if not args.no_confirm:
        events = [e for e in events if e['confirmed'] == 'yes']
    return _dedupe_events(events)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _dedupe_events(events):
    """Collapse near-duplicate (overlapping index-range) revisit pairs into the
    single most-separated representative per cluster."""
    if not events:
        return events
    events = sorted(events, key=lambda e: (e['frame_i'], e['frame_j']))
    kept = []
    for e in events:
        merged = False
        for k in kept:
            if abs(e['frame_i'] - k['frame_i']) <= 3 and abs(e['frame_j'] - k['frame_j']) <= 3:
                merged = True
                break
        if not merged:
            kept.append(e)
    return kept


def _write_csv(events, image_list, out_path):
    cols = ['method', 'frame_i', 'frame_j', 'image_i', 'image_j', 'gap',
            'distance_m', 'max_departure_m', 'overlap', 'confirmed']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for e in events:
            e = dict(e)
            e['image_i'] = image_list[e['frame_i']]
            e['image_j'] = image_list[e['frame_j']]
            w.writerow(e)


def main():
    ap = argparse.ArgumentParser(
        description="Detect site revisits (loop-closure candidates) from metadata "
                    "and/or image registration.")
    ap.add_argument("image_folder", help="Folder of images (single camera or a "
                    "PORT/STAR/CENTER multicam rig; CENTER is used as reference)")
    ap.add_argument("--method", choices=['metadata', 'registration', 'both'],
                    default='both', help="Detection method (default: both)")
    ap.add_argument("--flight-log", default=None,
                    help="FMCLOG CSV for the metadata method (not needed if an "
                         "imagelog.json or EXIF GPS is present)")
    ap.add_argument("--output", default=None, help="Output CSV (default: "
                    "<image_folder>/site_revisits.csv)")
    ap.add_argument("--min-gap", type=int, default=10,
                    help="Min temporal frame separation to count as a revisit "
                         "(default: 10)")
    ap.add_argument("--overlap-thresh", type=float, default=0.3,
                    help="Min ground-footprint overlap fraction (default: 0.3)")
    ap.add_argument("--footprint-radius", type=float, default=None,
                    help="Ground footprint radius in metres for the metadata "
                         "method (default: derived from EXIF focal + altitude)")
    ap.add_argument("--departure-factor", type=float, default=3.0,
                    help="The platform must travel at least this many footprint "
                         "radii away between i and j to count as a revisit "
                         "(default: 3.0)")
    ap.add_argument("--affine", action="store_true", default=True,
                    help="Use the affine model for registration matching "
                         "(default: on — matches the recommended config)")
    ap.add_argument("--no-affine", dest="affine", action="store_false",
                    help="Use the full homography model instead of affine")
    ap.add_argument("--match-scale", type=float, default=0.5,
                    help="Image scale for registration matching (default: 0.5)")
    ap.add_argument("--min-inliers", type=int, default=15,
                    help="Min inliers to confirm a registration revisit "
                         "(default: 15)")
    ap.add_argument("--no-confirm", action="store_true",
                    help="Skip the direct-match confirmation step (registration "
                         "method) — report footprint-overlap candidates only")
    ap.add_argument("--install-deps", action="store_true",
                    help="Install missing Python dependencies")
    args = ap.parse_args()

    if args.install_deps:
        r3d.ensure_dependencies(install=True)
        return
    r3d.import_dependencies()

    folder = args.image_folder
    image_list = _list_images(folder)
    if len(image_list) < 3:
        print(f"ERROR: need >=3 images, found {len(image_list)}")
        sys.exit(1)
    print(f"Detecting site revisits in {folder} ({len(image_list)} frames, "
          f"method={args.method})")

    site_name = __import__('re').sub(r'^\d{8}_', '',
                                     os.path.basename(folder.rstrip('/'))).replace('_', ' ')

    all_events = []
    want_meta = args.method in ('metadata', 'both')
    want_reg = args.method in ('registration', 'both')

    if want_meta:
        poses = r3d.load_pose_metadata(folder, image_list,
                                       flight_log=args.flight_log, site_name=site_name)
        if poses:
            print("  [metadata] detecting...")
            ev = detect_metadata(folder, image_list, poses, args)
            print(f"  [metadata] {len(ev)} revisit event(s)")
            all_events += ev
        else:
            print("  [metadata] no GPS metadata found for this folder; skipping.")
            if args.method == 'metadata':
                sys.exit(0)

    if want_reg:
        print("  [registration] detecting...")
        ev = detect_registration(folder, image_list, args)
        print(f"  [registration] {len(ev)} confirmed revisit event(s)")
        all_events += ev

    out = args.output or os.path.join(folder, 'site_revisits.csv')
    _write_csv(all_events, image_list, out)
    print(f"\n{len(all_events)} total revisit event(s) -> {out}")
    for e in sorted(all_events, key=lambda x: x['frame_i'])[:30]:
        extra = (f"dist={e.get('distance_m')}m" if e['method'] == 'metadata'
                 else f"overlap={e.get('overlap')} confirmed={e.get('confirmed')}")
        print(f"  [{e['method']:12s}] frame {e['frame_i']} <-> {e['frame_j']} "
              f"(gap {e['gap']})  {extra}")


if __name__ == "__main__":
    main()
