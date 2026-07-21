# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Pseudo-evaluation of ``detect_prior_coverage.py`` loop-closure recall.

Independently derives "expected" loop-closure events for a site from raw
flight-log GPS + altitude (ground footprint overlap between an image and any
earlier-pass/earlier-day image, using nothing from the imagery pipeline under
test), then compares that set against the ``image`` column of the tool's own
``revisits.csv`` to report what fraction of the expected revisits it
recovered.

This is a coverage/recall check, not ground truth in the strict sense -- the
flight-log-derived "expected" set is itself an estimate (footprint size is a
model, not measured), so treat the percentage as an approximate pseudo-score,
not an exact metric.

Usage::

    evaluate_loop_closures.py SITE_FOLDER --flight-logs DIR \\
        --revisits-csv OUT_DIR/revisits.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from viame.core import survey_metadata as smd


def _quad_overlap_frac(a, b):
    """Fraction of the smaller quad's area covered by the intersection."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if not inter or inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    return inter / max(min(area_a, area_b), 1e-6)


def expected_loop_closures(site_folder, flight_logs, overlap_thresh=0.15,
                           min_pass_gap=True, xcam_offset_frac=0.9,
                           min_frame_gap=0):
    """Return {rel_path: [(source_rel_path, source_pass, source_day, overlap_frac), ...]}
    for images whose GPS/altitude-derived footprint overlaps an earlier
    pass's (or earlier day's, for --all multi-site runs) footprint.

    Footprints come from ``survey_metadata.build_footprints``, which applies the
    PORT/STAR across-track cant: those cameras look ~0.9 footprint-widths off
    the aircraft track, so placing their footprint at the aircraft position
    would misplace it by ~95 m and mis-state which frames really share ground.
    """
    fps, cams = smd.build_footprints(site_folder, flight_logs=flight_logs,
                                     xcam_offset_frac=xcam_offset_frac,
                                     read_exif=True, verbose=True)
    day = smd.folder_date(site_folder) or ''
    if not fps:
        return {}

    rels = list(fps.keys())
    passes = {}
    order = {}
    for idx, rel in enumerate(rels):
        try:
            passes[rel] = int(fps[rel].get('pass') or 1)
        except (TypeError, ValueError):
            passes[rel] = 1
        fr = fps[rel].get('frame')
        order[rel] = fr if fr is not None else idx

    # When the data has no pass structure (a single continuous flight, e.g. the
    # 2025 single-camera UAS surveys where every frame is pass 1), a "loop
    # closure" is the platform returning to re-image the same ground after a
    # long gap. Detect it by a large capture-order separation + footprint
    # overlap, so ordinary consecutive-frame overlap (the sequential swath)
    # does not count. Auto-enabled when only one pass is present.
    single_pass = len(set(passes.values())) <= 1
    use_frame_gap = min_frame_gap > 0 and single_pass

    expected = {}
    for i, rel_i in enumerate(rels):
        for rel_j in rels[:i]:
            if use_frame_gap:
                if abs(order[rel_i] - order[rel_j]) < min_frame_gap:
                    continue  # too close in time -> sequential overlap, not a revisit
            elif min_pass_gap and passes[rel_j] >= passes[rel_i]:
                continue      # only re-covering an EARLIER pass is a revisit
            frac = _quad_overlap_frac(fps[rel_i]['quad'], fps[rel_j]['quad'])
            if frac >= overlap_thresh:
                expected.setdefault(rel_i, []).append(
                    (rel_j, passes[rel_j], day, frac))
    return expected


def _is_confirmed(row):
    return str(row.get('confirmed', '')).strip().lower() in ('true', '1', 'yes')


def load_detected_images(revisits_csv):
    detected = {}
    if not os.path.isfile(revisits_csv):
        return detected
    with open(revisits_csv) as f:
        for row in csv.DictReader(f):
            detected.setdefault(row['image'], []).append(row)
    return detected


def evaluate(site_folder, flight_logs, revisits_csv, overlap_thresh=0.15,
             min_frame_gap=0):
    expected = expected_loop_closures(site_folder, flight_logs, overlap_thresh,
                                      min_frame_gap=min_frame_gap)
    detected = load_detected_images(revisits_csv)

    n_expected = len(expected)
    n_found = sum(1 for rel in expected if rel in detected)
    recall = (n_found / n_expected) if n_expected else None

    # Rig-level (camera-agnostic) recall. STRICT recall above requires the exact
    # (camera, frame) image to be flagged. On a PORT/CENTER/STAR rig the pipeline
    # often credits a re-covered swath to a different camera than the footprint
    # model expects (e.g. it flags PORT#144 while the expected revisit is
    # STAR#144 - same trigger, same ground). That reads as 0% strict even though
    # the rig DID detect the re-coverage. Here an expected revisit is credited if
    # the rig flagged a revisit at the same TRIGGER (frame number) from any
    # camera. All three rig cameras share the frame number at a trigger.
    def _frame(rel):
        return smd.parse_image_filename(rel).get('frame')
    detected_frames = {_frame(rel) for rel in detected if _frame(rel) is not None}
    n_found_rig = sum(1 for rel in expected if _frame(rel) in detected_frames)
    recall_rig = (n_found_rig / n_expected) if n_expected else None

    # Independent (image-registration-confirmed) subset: of the expected
    # loop-closures the tool detected, how many did it verify by a direct
    # land-to-land feature match rather than GPS geometry alone. This is the
    # honest, non-circular signal -- GPS recall is partly structural because
    # the tool's hybrid detector reads the same flight log this evaluator does.
    n_conf = sum(1 for rel in expected
                 if any(_is_confirmed(r) for r in detected.get(rel, [])))
    conf_rate = (n_conf / n_found) if n_found else None

    missed = sorted(set(expected) - set(detected))
    return {
        'site': os.path.basename(os.path.normpath(site_folder)),
        'n_expected_images': n_expected,
        'n_detected_of_expected': n_found,
        'recall': recall,
        'n_detected_rig': n_found_rig,
        'recall_rig': recall_rig,
        'n_reg_confirmed_of_detected': n_conf,
        'confirm_rate': conf_rate,
        'missed_images': missed,
        'n_detected_total': len(detected),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('site', help='Site folder')
    ap.add_argument('--flight-logs', default=None)
    ap.add_argument('--revisits-csv', required=True,
                    help='revisits.csv produced by detect_prior_coverage.py')
    ap.add_argument('--overlap-thresh', type=float, default=0.15,
                    help='Minimum footprint overlap fraction to count as an '
                    'expected loop-closure/revisit (default 0.15)')
    ap.add_argument('--show-missed', type=int, default=10,
                    help='Print up to N missed expected-revisit image names')
    ap.add_argument('--min-frame-gap', type=int, default=0,
                    help='For single-pass flights (e.g. 2025 UAS), the capture-'
                    'order gap beyond which an overlapping footprint counts as a '
                    'revisit rather than sequential swath overlap (0 = off; '
                    'try 20). Auto-applies only when the site has one pass.')
    args = ap.parse_args()

    result = evaluate(args.site, args.flight_logs, args.revisits_csv,
                      args.overlap_thresh, min_frame_gap=args.min_frame_gap)

    print(f"=== {result['site']} ===")
    print(f"  Expected revisit images (flight-log GPS footprint overlap, "
          f">= {args.overlap_thresh*100:.0f}% overlap with an earlier pass): "
          f"{result['n_expected_images']}")
    print(f"  Detected by tool (revisits.csv, any images): {result['n_detected_total']}")
    if result['recall'] is None:
        print("  No expected revisits for this site (single pass, or passes are "
              "offset transects with no ground overlap) -- recall not applicable")
    else:
        print(f"  Strict (per-camera) recall: {result['n_detected_of_expected']}/"
              f"{result['n_expected_images']} = {result['recall']*100:.1f}%")
        print(f"  Rig-level (camera-agnostic) recall: {result['n_detected_rig']}/"
              f"{result['n_expected_images']} = {result['recall_rig']*100:.1f}%")
        if result['confirm_rate'] is not None:
            print(f"  ...of which image-registration-confirmed (independent): "
                  f"{result['n_reg_confirmed_of_detected']}/"
                  f"{result['n_detected_of_expected']} = "
                  f"{result['confirm_rate']*100:.1f}%")
    if result['missed_images'] and args.show_missed:
        print(f"  Missed (first {args.show_missed}):")
        for m in result['missed_images'][:args.show_missed]:
            print(f"    {m}")


if __name__ == '__main__':
    main()
