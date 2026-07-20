# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Catalogue registration/matching FAILURE CASES for a survey site.

Aerial sea-lion survey imagery is dominated by two terrain types that break
feature matching:

  * open ocean / seaweed water - no stable features; SIFT locks onto waves,
    glint and foam that move between frames, so a "successful" match is
    meaningless.
  * tussock grass / vegetated island interior ("all_land") - highly repetitive
    self-similar texture, so SIFT aliases onto the WRONG patch of grass and
    produces a confident but geometrically wrong match.

Both failures look like a successful registration from inside the pipeline.
This tool exposes them by cross-checking every match the pipeline CLAIMED
against the flight-log GPS, which is independent of the imagery:

    if the images say "these two frames see the same ground"
    but the GPS says the aircraft was 400 m away
    -> the match is false, and we can name the exact frame it matched against.

Inputs are a site folder plus the ``revisits.csv`` that
``detect_prior_coverage.py`` wrote for it. Phase 1 (metadata only, ~free)
flags the false/missed matches. Phase 2 (``--visualize``) re-runs SIFT on the
flagged pairs to record inlier counts and render side-by-side match images, so
the failure can be eyeballed later.

Usage::

    analyze_match_failures.py SITE --flight-logs DIR --revisits-csv OUT/revisits.csv \\
        --output OUT/failures [--visualize]
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from viame.core import survey_metadata as smd

# SVM background-classifier labels -> coarse terrain bucket. On these islands
# 'all_land' is predominantly tussock grass / low vegetation, which is the
# repetitive texture that causes aliased matches.
TERRAIN = {
    'open_water': 'ocean',
    'seaweed_water': 'ocean',
    'cloudy': 'cloud',
    'coastal': 'coastal',
    'all_land': 'grass_land',
}

# A claimed match between frames whose GPS positions differ by more than this
# cannot be real: image footprints are ~105 m across at survey altitude, so
# frames >150 m apart share no ground at all.
FOOTPRINT_M = 105.0


def _terrain(label):
    return TERRAIN.get(label, 'unknown')


def _pair_terrain(ta, tb):
    if ta == tb:
        return ta
    if 'ocean' in (ta, tb) and 'coastal' in (ta, tb):
        return 'coastal'
    return f'{ta}+{tb}'


def load_site_geo(site_folder, flight_logs, xcam_offset_frac=0.9):
    """Per-image ground FOOTPRINT (not aircraft position) in a shared ENU frame.

    The rig's PORT/STAR cameras are canted across-track, so their footprints sit
    ~0.9 footprint-widths off the aircraft track. Testing shared ground with raw
    aircraft positions would both invent false failures (PORT/STAR pairs whose
    footprints really do overlap despite the aircraft being far apart) and hide
    real ones (PORT vs STAR at the same trigger: aircraft distance 0, but ~190 m
    between footprints).
    """
    return smd.build_footprints(site_folder, flight_logs=flight_logs,
                                xcam_offset_frac=xcam_offset_frac,
                                read_exif=False, verbose=False)


def _footprint_overlap(qa, qb):
    """Fraction of the smaller footprint covered by the intersection (0 = no
    shared ground)."""
    import cv2
    a = np.array(qa, dtype=np.float32)
    b = np.array(qb, dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if not inter or inter <= 0:
        return 0.0
    aa, ab = cv2.contourArea(a), cv2.contourArea(b)
    return float(inter / max(min(aa, ab), 1e-6))


def classify_terrain(site_folder, rels, method='auto'):
    """SVM/SIFT water-land label per image; {} if the classifier is unavailable."""
    try:
        from viame.opencv import registration_utils as ru
        ru.import_dependencies()
        return ru.classify_images_fast(site_folder, rels, method=method)
    except Exception as e:
        print(f'    terrain classification unavailable ({e})')
        return {}


def load_coverage_claims(coverage_csv, geo):
    """Claimed shared-ground pairs from a prior_coverage.csv.

    Each row is one previously-seen region of an image, tagged with the source
    frame that saw it (``(note) src=CAM#FRAME``) and a class suffix
    (_sequential / _cross_camera / _revisit). Yields deduped
    (image, source_image, kind) triples. This covers the whole registration
    chain -- notably the SEQUENTIAL matches over open water, which are where a
    mosaic actually tears -- rather than only the revisit events.
    """
    if not coverage_csv or not os.path.isfile(coverage_csv):
        return []
    # (cam, frame) -> rel path, to resolve the src=CAM#FRAME tag.
    by_cam_frame = {}
    for rel, g in geo.items():
        by_cam_frame[(str(g.get('cam')), str(g.get('frame')))] = rel

    claims, seen = [], set()
    with open(coverage_csv) as f:
        for line in f:
            if line.startswith('#') or ',' not in line:
                continue
            parts = line.rstrip('\n').split(',')
            if len(parts) < 10:
                continue
            rel = parts[1]
            cls = parts[9]
            if 'prior_coverage_' not in cls:
                continue
            kind = cls.split('prior_coverage_')[-1]
            m = None
            for p in parts:
                if 'src=' in p:
                    m = p.split('src=')[-1].strip()
                    break
            if not m or '#' not in m:
                continue
            cam, frame = m.split('#', 1)
            src_rel = by_cam_frame.get((cam, frame))
            if not src_rel or src_rel == rel:
                continue
            key = (rel, src_rel, kind)
            if key in seen:
                continue
            seen.add(key)
            claims.append(key)
    return claims


def analyze(site_folder, flight_logs, revisits_csv, water_method='auto',
            min_overlap=0.0, coverage_csv=None, xcam_offset_frac=0.9):
    """Cross-check every match the pipeline claimed against flight-log GPS.

    Checks both the revisit events (revisits.csv) and, when given, every
    claimed shared-ground region of the registration chain
    (prior_coverage.csv: sequential / cross_camera / revisit).

    A claim is false when the two frames' GPS-derived ground footprints do not
    overlap at all: the images assert they see the same ground, the flight log
    says they cannot.
    """
    geo, cams = load_site_geo(site_folder, flight_logs, xcam_offset_frac)
    if not geo:
        print('    no GPS fixes for this site; cannot cross-check matches')
        return [], {}

    # (image, source_image, kind, confirmed, overlap) claims from both sources.
    claims = []
    if os.path.isfile(revisits_csv):
        for r in csv.DictReader(open(revisits_csv)):
            claims.append((r['image'], r['source_image'], 'revisit',
                           str(r.get('confirmed', '')).strip().lower()
                           in ('yes', 'true', '1'),
                           r.get('overlap_frac', '')))
    for rel, src, kind in load_coverage_claims(coverage_csv, geo):
        claims.append((rel, src, kind, None, ''))
    if not claims:
        print('    no claimed matches found to check')
        return [], {}

    # Terrain only for frames actually involved in a claim.
    involved = sorted({c[0] for c in claims} | {c[1] for c in claims})
    involved = [r for r in involved if os.path.isfile(os.path.join(site_folder, r))]
    water = classify_terrain(site_folder, involved, water_method)

    site_tag = os.path.basename(os.path.normpath(site_folder))
    failures = []
    stats = {'claims': 0, 'false': 0, 'confirmed_false': 0, 'ok': 0, 'no_gps': 0}
    seen = set()

    for a, b, kind, confirmed, overlap in claims:
        if (a, b, kind) in seen:
            continue
        seen.add((a, b, kind))
        stats['claims'] += 1
        ga, gb = geo.get(a), geo.get(b)
        if not ga or not gb:
            stats['no_gps'] += 1
            continue
        # Compare where the CAMERAS actually looked (footprints), not where the
        # aircraft was: the PORT/STAR cant makes those two very different.
        fov = _footprint_overlap(ga['quad'], gb['quad'])
        d = float(np.hypot(ga['center'][0] - gb['center'][0],
                           ga['center'][1] - gb['center'][1]))
        la = (water.get(a) or {}).get('label', 'unknown')
        lb = (water.get(b) or {}).get('label', 'unknown')
        ta, tb = _terrain(la), _terrain(lb)

        if fov > min_overlap:
            stats['ok'] += 1
            continue

        # Footprints share no ground, yet the pipeline claimed they do -> false
        # match. `confirmed` means SIFT actively vouched for it (the worse case:
        # a genuine wrong feature match, not just grid bookkeeping).
        stats['false'] += 1
        if confirmed:
            stats['confirmed_false'] += 1
        mode = f'false_{kind}'
        if confirmed:
            mode += '_sift_confirmed'
        failures.append({
            'site': site_tag,
            'camera': (ga.get('cam') or ''),
            'frame': (ga.get('frame') if ga else ''),
            'image': a,
            'matched_against_image': b,
            'matched_against_frame': (gb.get('frame') if gb else ''),
            'match_kind': kind,
            'terrain': _pair_terrain(ta, tb),
            'class_image': la,
            'class_matched_against': lb,
            'footprint_gap_m': f'{d:.1f}',
            'overlap_frac_claimed': overlap,
            'sift_confirmed': ('yes' if confirmed else
                               ('no' if confirmed is False else 'n/a')),
            'failure_mode': mode,
            'note': (f'GPS footprints disjoint (centres {d:.0f} m apart, '
                     f'~{FOOTPRINT_M:.0f} m footprint): frames share no ground'),
        })
    failures.sort(key=lambda f: -float(f['footprint_gap_m']))
    return failures, stats


def write_failures_csv(path, failures):
    cols = ['site', 'camera', 'frame', 'image', 'matched_against_image',
            'matched_against_frame', 'match_kind', 'terrain', 'class_image',
            'class_matched_against', 'footprint_gap_m', 'overlap_frac_claimed',
            'sift_confirmed', 'failure_mode', 'note']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in failures:
            w.writerow(r)


def visualize_failures(site_folder, failures, out_dir, limit=12, scale=0.25):
    """Render side-by-side image pairs with SIFT matches drawn, so a human can
    see WHY the match was wrong (waves/glint on ocean, aliased grass texture)."""
    try:
        import cv2
    except ImportError:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for fr in failures[:limit]:
        pa = os.path.join(site_folder, fr['image'])
        pb = os.path.join(site_folder, fr['matched_against_image'])
        ia, ib = cv2.imread(pa), cv2.imread(pb)
        if ia is None or ib is None:
            continue
        ia = cv2.resize(ia, None, fx=scale, fy=scale)
        ib = cv2.resize(ib, None, fx=scale, fy=scale)
        ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=4000)
        ka, da = sift.detectAndCompute(ga, None)
        kb, db = sift.detectAndCompute(gb, None)
        if da is None or db is None:
            continue
        good = []
        for m_n in cv2.BFMatcher().knnMatch(da, db, k=2):
            if len(m_n) == 2 and m_n[0].distance < 0.8 * m_n[1].distance:
                good.append(m_n[0])
        vis = cv2.drawMatches(ia, ka, ib, kb, good[:60], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        hdr = (f"{fr['terrain']}  footprints {fr['footprint_gap_m']}m apart  "
               f"sift_confirmed={fr['sift_confirmed']}  ({len(good)} raw matches)")
        cv2.putText(vis, hdr, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{os.path.basename(fr['image'])}  vs  "
                         f"{os.path.basename(fr['matched_against_image'])}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)
        name = f"{fr['terrain']}_{os.path.splitext(os.path.basename(fr['image']))[0]}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), vis)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('site')
    ap.add_argument('--flight-logs', default=None)
    ap.add_argument('--revisits-csv', required=True)
    ap.add_argument('--coverage-csv', default=None,
                    help='prior_coverage.csv from the same run: also checks the '
                         'sequential / cross-camera chain matches, not just revisits')
    ap.add_argument('--output', required=True, help='Output directory')
    ap.add_argument('--water-method', choices=['auto', 'svm', 'sift'], default='auto')
    ap.add_argument('--min-overlap', type=float, default=0.0,
                    help='Claimed matches whose GPS footprints overlap by no '
                         'more than this fraction are false (default 0 = '
                         'footprints must be strictly disjoint to flag)')
    ap.add_argument('--xcam-offset-frac', type=float, default=0.9,
                    help='Across-track cant of PORT/STAR footprints as a '
                         'fraction of footprint width (match '
                         'detect_prior_coverage.py --xcam-offset-frac)')
    ap.add_argument('--visualize', action='store_true',
                    help='Render side-by-side SIFT-match images for flagged pairs')
    ap.add_argument('--vis-limit', type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    site_tag = os.path.basename(os.path.normpath(args.site))
    print(f'=== {site_tag}: match failure analysis ===')

    failures, stats = analyze(args.site, args.flight_logs, args.revisits_csv,
                              args.water_method, args.min_overlap,
                              coverage_csv=args.coverage_csv,
                              xcam_offset_frac=args.xcam_offset_frac)
    csv_path = os.path.join(args.output, 'match_failures.csv')
    write_failures_csv(csv_path, failures)

    print(f"  claimed matches checked : {stats.get('claims', 0)}")
    print(f"  footprint-consistent(OK): {stats.get('ok', 0)}")
    print(f"  FALSE matches           : {stats.get('false', 0)}"
          f"  (of which SIFT-confirmed: {stats.get('confirmed_false', 0)})")
    if stats.get('no_gps'):
        print(f"  skipped (no GPS)        : {stats['no_gps']}")

    if failures:
        from collections import Counter
        by_terrain = Counter(f['terrain'] for f in failures)
        by_kind = Counter(f['match_kind'] for f in failures)
        print('  false matches by terrain:')
        for t, c in by_terrain.most_common():
            print(f'    {t:20s} {c}')
        print('  false matches by match kind:')
        for k, c in by_kind.most_common():
            print(f'    {k:20s} {c}')
        worst = failures[0]
        print(f"  worst: {os.path.basename(worst['image'])} matched against "
              f"{os.path.basename(worst['matched_against_image'])} "
              f"(footprints {worst['footprint_gap_m']} m apart, "
              f"{worst['terrain']})")
        print(f'  -> {csv_path}')
        if args.visualize:
            vdir = os.path.join(args.output, 'failure_examples')
            n = visualize_failures(args.site, failures, vdir, args.vis_limit)
            print(f'  rendered {n} example match images -> {vdir}')
    else:
        print('  no GPS-inconsistent matches found')


if __name__ == '__main__':
    main()
