#!/usr/bin/env python3
"""survey_metadata.py - Unified per-image metadata for multi-camera aerial surveys.

Builds one record per image by fusing every metadata source available for the
SSL survey collections:

  1. FMC daily flight logs (SSL-FMCLOG_YYYY-MM-DD_*.csv). One row per camera
     trigger with GPS (NMEA), attitude (roll/pitch/yaw), altitude, UTC time,
     a global per-day frame counter, site name, and pass number. The frame
     counter matches the trailing number in the image filename
     (e.g. 20240708_CATON_SLC00186.JPG -> frame 186), which is how rows are
     linked to images. All three rig cameras (SLC/SLP/SLS = CENTER/PORT/STAR)
     share the trigger row of their frame number.
  2. EXIF embedded in the JPGs. The 2024 rig (Sony A7R III) carries precise
     timestamps (+ UTC offset) but NO GPS; timestamps are used to cross-check
     the flight-log linkage. The 2025 UAS imagery (Sony A7R IV) carries full
     GPS (lat/lon/alt/heading) and needs no flight log.

The output record is a plain dict:

  {camera, frame, time_utc, lat, lon, alt_agl, roll, pitch, yaw,
   site, pass, source}

`camera` is CENTER/PORT/STAR (or None for single-camera collections), `frame`
the trigger/frame number, `source` one of 'flight-log', 'exif-gps'.

Also provides local-ENU projection and metadata-derived ground footprints so
callers can reason about coverage in metres without any image registration.

Can be run standalone to dump a per-image metadata table:

  python survey_metadata.py <site_folder> --flight-logs <dir-or-csv> [--csv out.csv]
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

# Camera letter in SSL filenames -> rig camera subfolder name.
CAMERA_LETTERS = {'C': 'CENTER', 'P': 'PORT', 'S': 'STAR'}

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

# Full-frame sensor dimensions (mm) used with FocalLengthIn35mmFilm to derive
# the ground footprint of a (near-)nadir image.
SENSOR_W_MM = 36.0
SENSOR_H_MM = 24.0

EARTH_R = 6378137.0


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_SSL_NAME_RE = re.compile(
    r'^(?P<date>\d{8})_(?P<site>.+)_SL(?P<cam>[CPS])(?P<frame>\d+)\.\w+$',
    re.IGNORECASE)
_TRAILING_NUM_RE = re.compile(r'(?P<frame>\d+)\.\w+$')
_FOLDER_DATE_RE = re.compile(r'(?P<date>\d{8})')


def parse_image_filename(name):
    """Parse an SSL image filename into (date, site, camera, frame).

    Returns a dict with any fields that could be determined; falls back to
    just the trailing frame number for non-SSL naming.
    """
    base = os.path.basename(name)
    m = _SSL_NAME_RE.match(base)
    if m:
        return {'date': m.group('date'),
                'site': m.group('site'),
                'camera': CAMERA_LETTERS.get(m.group('cam').upper()),
                'frame': int(m.group('frame'))}
    m = _TRAILING_NUM_RE.search(base)
    return {'date': None, 'site': None, 'camera': None,
            'frame': int(m.group('frame')) if m else None}


def folder_date(folder):
    """Extract YYYYMMDD from a site folder name like '20240708_CATON'."""
    m = _FOLDER_DATE_RE.search(os.path.basename(os.path.normpath(folder)))
    return m.group('date') if m else None


def normalize_site(name):
    """Normalize a site name for comparison ('CLUBBING ROCKS_SOUTH' etc)."""
    return re.sub(r'[^a-z0-9]', '', (name or '').lower())


# ---------------------------------------------------------------------------
# Flight logs
# ---------------------------------------------------------------------------

def find_flight_log(logs, date):
    """Locate the flight-log CSV for YYYYMMDD `date`.

    `logs` may be a CSV path, or a directory searched for
    SSL-FMCLOG_YYYY-MM-DD_*.csv (an '_edited' variant is preferred when both
    exist since those contain manual corrections).

    An explicit CSV whose filename embeds a date is only accepted when that
    date matches: flight-log rows link to images by the per-day frame counter,
    so a wrong-day CSV would otherwise "match" by frame number and silently
    georeference the site with another flight's positions (it also outranks
    per-image EXIF GPS). A filename without a recognizable date is passed
    through unchanged.
    """
    if logs is None or date is None:
        return None
    if os.path.isfile(logs):
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', os.path.basename(logs))
        if m and ''.join(m.groups()) != date:
            print(f'    Ignoring flight log {os.path.basename(logs)}: its '
                  f'date does not match the survey folder date {date}')
            return None
        return logs
    iso = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
    matches = sorted(glob.glob(os.path.join(logs, f'*FMCLOG_{iso}*.csv')))
    if not matches:
        matches = sorted(glob.glob(os.path.join(logs, '**',
                                                f'*FMCLOG_{iso}*.csv'),
                         recursive=True))
    if not matches:
        return None
    edited = [m for m in matches if 'edited' in os.path.basename(m).lower()]
    return (edited or matches)[0]


def _nmea_to_dec(value, ref):
    v = float(value)
    deg = int(v // 100)
    dec = deg + (v - 100 * deg) / 60.0
    return -dec if str(ref).strip().upper() in ('S', 'W') else dec


def _parse_utc(s):
    """Parse '2024-07-08 20:37:46 +0000' (and close variants) to aware UTC."""
    s = (s or '').strip()
    for fmt in ('%Y-%m-%d %H:%M:%S %z', '%Y-%m-%d %H:%M:%S'):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def load_flight_log(csv_path):
    """Parse an FMCLOG CSV into {frame_count: record}.

    The frame counter is global across the day, so one map serves every site
    flown that day. Records carry lat/lon in decimal degrees, altitude in
    metres, attitude in degrees, UTC time, site name and pass number.
    """
    frames = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                fc = int(row.get('frame_count', '') or 0)
            except ValueError:
                continue
            if fc <= 0:
                continue
            try:
                lat = _nmea_to_dec(row['lat'], row.get('lat_ns', 'N'))
                lon = _nmea_to_dec(row['lon'], row.get('lon_ew', 'E'))
            except (ValueError, KeyError):
                continue

            def _f(key):
                try:
                    return float(row.get(key, '')) if row.get(key) else None
                except ValueError:
                    return None

            try:
                pass_no = int(row.get('pass', '') or 1)
            except ValueError:
                pass_no = 1
            frames[fc] = {
                'frame': fc,
                'time_utc': _parse_utc(row.get('utc_time')),
                'lat': lat, 'lon': lon,
                'alt_agl': _f('elevation_m'),
                'roll': _f('roll'), 'pitch': _f('pitch'), 'yaw': _f('yaw'),
                'site': (row.get('site_name') or '').strip() or None,
                'pass': pass_no,
                'source': 'flight-log',
            }
    return frames


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------

def load_exif_meta(path):
    """Read timestamp, GPS (when present) and lens info from one image's EXIF.

    Returns {} when the file has no usable EXIF. Timestamps are converted to
    aware UTC using the EXIF OffsetTime when available.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS
    except ImportError:
        return {}
    out = {}
    try:
        with Image.open(path) as im:
            out['width'], out['height'] = im.size
            ex = im.getexif()
            exif_ifd = ex.get_ifd(0x8769)
            gps = ex.get_ifd(0x8825)
    except Exception:
        return out

    dt = exif_ifd.get(0x9003) or ex.get(0x0132)   # DateTimeOriginal / DateTime
    off = exif_ifd.get(0x9011) or exif_ifd.get(0x9010)  # OffsetTimeOriginal
    if dt:
        try:
            naive = datetime.strptime(str(dt), '%Y:%m:%d %H:%M:%S')
            if off:
                m = re.match(r'([+-])(\d{2}):(\d{2})', str(off))
                if m:
                    sign = 1 if m.group(1) == '+' else -1
                    delta = timedelta(hours=int(m.group(2)),
                                      minutes=int(m.group(3)))
                    naive = naive - sign * delta
            out['time_utc'] = naive.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    out['focal_mm'] = float(exif_ifd[0x920A]) if 0x920A in exif_ifd else None
    out['focal35_mm'] = (float(exif_ifd[0xA405])
                         if 0xA405 in exif_ifd else None)

    if gps:
        g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
        try:
            def _dms(dms, ref):
                d, m_, s = [float(x) for x in dms]
                dec = d + m_ / 60.0 + s / 3600.0
                return -dec if str(ref).upper() in ('S', 'W') else dec
            out['lat'] = _dms(g['GPSLatitude'], g.get('GPSLatitudeRef', 'N'))
            out['lon'] = _dms(g['GPSLongitude'], g.get('GPSLongitudeRef', 'E'))
            if g.get('GPSAltitude') is not None:
                out['alt'] = float(g['GPSAltitude'])
            if g.get('GPSImgDirection') is not None:
                hdg = float(g['GPSImgDirection'])
                # 0xFFFFFFFF (~4.29e9) is the EXIF "no data" sentinel some UAS
                # cameras write for direction; reject it rather than inject a
                # garbage heading.
                if 0.0 <= hdg <= 360.0:
                    out['heading'] = hdg
        except (KeyError, ValueError, TypeError):
            pass
    return out


# ---------------------------------------------------------------------------
# 2025 UAS imagelog.json ingest
# ---------------------------------------------------------------------------

def load_imagelog(site_folder):
    """Load the 2025 single-camera UAS ``imagelog.json`` pose log(s) of a site.

    A site folder holds one or more ``imagelog*.json`` files (a flight split
    across battery swaps produces e.g. ``imagelog.json`` + ``imagelog (2).json``).
    Each holds an ``ImageLog`` list of per-trigger records with decimal lat/lon,
    absolute altitude ``alt`` (the imaging height AGL for these low UAS flights;
    ``alt_rel`` is unreliable here), and attitude ``yaw``/``pitch``/``roll`` in
    RADIANS (unlike the 2024 FMCLOG, which is degrees).

    Returns a list of pose dicts sorted by capture order (``trigger_index``),
    with yaw/pitch/roll converted to degrees, or ``[]`` if no log is present.
    The records are NOT yet linked to image files -- the on-disk images are
    renamed and fewer in number than the triggers, so :func:`link_imagelog`
    matches them by GPS position.
    """
    if site_folder and os.path.isfile(site_folder):
        logs = [site_folder]
    elif site_folder and os.path.isdir(site_folder):
        logs = sorted(glob.glob(os.path.join(site_folder, 'imagelog*.json')))
    else:
        logs = []
    recs = []
    for path in logs:
        try:
            data = json.load(open(path))
        except (ValueError, OSError):
            continue
        rows = data.get('ImageLog', []) if isinstance(data, dict) else data
        for r in rows:
            if r.get('lat') is None or r.get('lon') is None:
                continue
            alt = r.get('alt')
            recs.append({
                'lat': float(r['lat']), 'lon': float(r['lon']),
                'alt_agl': float(alt) if alt is not None else None,
                'yaw': (math.degrees(float(r['yaw'])) % 360.0
                        if r.get('yaw') is not None else None),
                'pitch': (math.degrees(float(r['pitch']))
                          if r.get('pitch') is not None else None),
                'roll': (math.degrees(float(r['roll']))
                         if r.get('roll') is not None else None),
                'trigger_index': r.get('trigger_index'),
                'time_utc': r.get('time_utc'),
            })
    recs.sort(key=lambda r: (r.get('trigger_index') is None,
                             r.get('trigger_index')))
    return recs


def link_imagelog(site_folder, image_list, log_recs, read_exif=True,
                  match_window=60, max_match_m=25.0):
    """Associate each image with its imagelog pose record.

    The on-disk files are renamed (``<date>_<SITE>_<seq>.jpg``) and there are
    fewer of them than log triggers, so a positional filename/index match is not
    possible. Instead each image's own EXIF GPS (same GPS unit as the log, so
    the true pair matches to sub-metre) is matched to the nearest log record,
    scanning FORWARD from the last match (monotonic capture order) so a revisit
    of the same ground pairs with the right pass rather than an earlier one.

    Falls back to plain order-pairing for images that lack EXIF GPS. Returns
    ``({rel_path: record}, stats)``.
    """
    if not log_recs:
        return {}, {'matched': 0, 'by_position': 0, 'by_order': 0}
    out = {}
    n = len(log_recs)
    j = 0                     # forward pointer into log_recs
    by_pos = by_order = 0
    resid = []
    for rel in image_list:
        exif = (load_exif_meta(os.path.join(site_folder, rel))
                if read_exif else {})
        rec = None
        if exif.get('lat') is not None and j < n:
            hi = min(n, j + match_window)
            best_k, best_d = None, None
            for k in range(j, hi):
                lr = log_recs[k]
                d = math.hypot((exif['lat'] - lr['lat']) * 111320.0,
                               (exif['lon'] - lr['lon']) * 111320.0
                               * math.cos(math.radians(exif['lat'])))
                if best_d is None or d < best_d:
                    best_k, best_d = k, d
            if best_k is not None and best_d <= max_match_m:
                rec = dict(log_recs[best_k])
                j = best_k + 1
                by_pos += 1
                resid.append(best_d)
        if rec is None:              # order fallback (no/failed EXIF match)
            if j < n:
                rec = dict(log_recs[j])
                j += 1
                by_order += 1
            else:
                rec = {}
        # Position is authoritative from the image's own EXIF GPS (exact); the
        # imagelog supplies the attitude (yaw) EXIF lacks. Only when the image
        # has no EXIF GPS do we fall back to the log record's position.
        if exif.get('lat') is not None:
            rec['lat'], rec['lon'] = exif['lat'], exif['lon']
            if exif.get('alt') is not None:
                rec['alt_agl'] = exif['alt']
        if not rec.get('lat'):
            continue
        out[rel] = rec
    stats = {'matched': len(out), 'by_position': by_pos, 'by_order': by_order,
             'median_resid_m': (sorted(resid)[len(resid) // 2] if resid else None)}
    return out, stats


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def list_site_images(site_folder):
    """List images of a site as {camera: [relative paths]}.

    Detects the rig layout (CENTER/PORT/STAR subfolders); single-camera
    collections come back under the key None. Paths are relative to
    `site_folder` and sorted by frame number.
    """
    def _frame_key(p):
        info = parse_image_filename(p)
        return (info['frame'] if info['frame'] is not None else 0, p)

    cams = {}
    subs = {d.upper(): d for d in os.listdir(site_folder)
            if os.path.isdir(os.path.join(site_folder, d))}
    rig = [c for c in ('CENTER', 'PORT', 'STAR') if c in subs]
    if rig:
        for cam in rig:
            d = os.path.join(site_folder, subs[cam])
            imgs = [os.path.join(subs[cam], f) for f in os.listdir(d)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            cams[cam] = sorted(imgs, key=_frame_key)
    else:
        imgs = [f for f in os.listdir(site_folder)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                and os.path.isfile(os.path.join(site_folder, f))]
        cams[None] = sorted(imgs, key=_frame_key)
    return cams


def build_image_records(site_folder, flight_logs=None, read_exif=True,
                        verbose=True):
    """Build unified metadata records for every image of a site.

    Returns ({rel_path: record}, {camera: [rel_paths]}). Records may be {}
    for images with no metadata from any source. Linking strategy:

      * flight log row matched by the image's frame number (global per-day
        trigger counter) - the primary source for the 2024 rig data;
      * EXIF GPS per image - the primary source for 2025 UAS data;
      * EXIF timestamps corroborate the flight-log linkage; a large median
        time offset is reported since it indicates a mis-link.
    """
    date = folder_date(site_folder)
    cams = list_site_images(site_folder)
    log_path = find_flight_log(flight_logs, date)
    log = load_flight_log(log_path) if log_path else {}
    if verbose and flight_logs and not log:
        print(f'    No flight log found for date {date}')

    # 2025 single-camera UAS: a co-located imagelog.json is the pose source.
    # Auto-detected (no --flight-logs needed); linked to the renamed image
    # files by GPS position. Only meaningful for a single-camera collection.
    imagelog = {}
    if len(cams) == 1 and None in cams and not log:
        il_recs = load_imagelog(site_folder)
        if il_recs:
            imagelog, il_stats = link_imagelog(
                site_folder, cams[None], il_recs, read_exif=read_exif)
            if verbose:
                mr = il_stats.get('median_resid_m')
                mrtxt = f', {mr:.1f} m median EXIF match' if mr is not None else ''
                print(f'    imagelog.json: linked {il_stats["matched"]}/'
                      f'{len(cams[None])} images '
                      f'({il_stats["by_position"]} by GPS, '
                      f'{il_stats["by_order"]} by order{mrtxt})')

    records = {}
    n_log = n_exif_gps = n_imagelog = 0
    time_deltas = []
    for cam, rel_paths in cams.items():
        for rel in rel_paths:
            info = parse_image_filename(rel)
            rec = {}
            il = imagelog.get(rel)
            if il is not None:
                rec = {'lat': il['lat'], 'lon': il['lon'],
                       'alt_agl': il.get('alt_agl'), 'yaw': il.get('yaw'),
                       'pitch': il.get('pitch'), 'roll': il.get('roll'),
                       'site': info['site'], 'pass': 1, 'frame': info['frame'],
                       'source': 'imagelog'}
                n_imagelog += 1
            row = (log.get(info['frame'])
                   if info['frame'] is not None and rec.get('lat') is None
                   else None)
            if row is not None:
                rec = dict(row)
                n_log += 1
            exif = (load_exif_meta(os.path.join(site_folder, rel))
                    if read_exif else {})
            if exif.get('lat') is not None and rec.get('lat') is None:
                rec.update({'lat': exif['lat'], 'lon': exif['lon'],
                            'alt_agl': exif.get('alt'),
                            'yaw': exif.get('heading'),
                            'roll': None, 'pitch': None,
                            'site': info['site'], 'pass': 1,
                            'frame': info['frame'],
                            'source': 'exif-gps'})
                n_exif_gps += 1
            if exif.get('time_utc'):
                if rec.get('time_utc'):
                    time_deltas.append(abs(
                        (exif['time_utc'] - rec['time_utc']).total_seconds()))
                else:
                    rec['time_utc'] = exif['time_utc']
            for k in ('focal_mm', 'focal35_mm', 'width', 'height'):
                if exif.get(k) is not None:
                    rec[k] = exif[k]
            rec['camera'] = cam
            if rec.get('frame') is None:
                rec['frame'] = info['frame']
            records[rel] = rec

    if verbose:
        total = sum(len(v) for v in cams.values())
        srcs = []
        if n_imagelog:
            srcs.append(f'{n_imagelog} imagelog')
        if n_log:
            srcs.append(f'{n_log} flight-log')
        if n_exif_gps:
            srcs.append(f'{n_exif_gps} EXIF-GPS')
        print(f'    Metadata: {" + ".join(srcs) if srcs else "none"}'
              f' of {total} images')
        if time_deltas:
            time_deltas.sort()
            med = time_deltas[len(time_deltas) // 2]
            note = '' if med <= 5.0 else '  ** LARGE - check linkage **'
            print(f'    EXIF vs flight-log time offset: median {med:.1f}s'
                  f' (n={len(time_deltas)}){note}')
    return records, cams


# ---------------------------------------------------------------------------
# Geometry: local ENU + metadata footprints
# ---------------------------------------------------------------------------

# Across-track cant of the rig cameras, as a fraction of the ground footprint
# width, matching ``detect_prior_coverage.py --xcam-offset-frac`` (default 0.9).
# The PORT/STAR optical axes point away from nadir, so their ground footprints
# sit ~0.9 footprint-widths either side of the aircraft track. Using the raw
# aircraft position for these cameras misplaces their footprint by ~95 m, which
# silently corrupts any footprint-overlap test.
#
# SIGN (verified empirically, 2024 survey): the camera in the PORT folder LOOKS
# to STARBOARD (+across-track) and the STAR camera looks to PORT. Measured by
# registering PORT/STAR frames against CENTER at the same trigger and mapping
# the offset to ground via the GPS-calibrated CENTER pixel scale: PORT boresight
# median +90 m (right of track), STAR -97 m, on CASTLE ROCK + PINNACLE ROCK
# (4 probes; see sea_lion_work/verify_rig_orientation.py). Folder names likely
# reflect the mounting side of a cross-aimed rig, not the look side.
CAM_LATERAL_FRAC = {'PORT': 0.9, 'CENTER': 0.0, 'STAR': -0.9}


def gps_degenerate(records, cams, min_unique_frac=0.5, max_single_share=0.5):
    """Detect a frozen / duplicated GPS track. Returns (bool, reason).

    An FMC log can report a valid fix (``gps_active_void='A'``, ``fix_type=3``)
    while the position itself is stale, so nothing downstream flags it. Seen in
    the 2024 survey: the GLACIER block of the (hand-edited) 2024-06-23 log holds
    242 trigger records spanning 6 seconds, 231 of them stamped with one
    identical lat/lon.

    Such a track is WORSE than no GPS: every frame dead-reckons onto the same
    spot, so footprints stack, every frame looks "already seen", and a
    footprint-overlap loop-closure score reads a meaningless 100%. Callers
    should drop the metadata and fall back to registration-only geo-referencing.
    """
    for cam, rels in cams.items():
        pts = [(round(r['lat'], 6), round(r['lon'], 6))
               for rel in rels
               if (r := records.get(rel)) and r.get('lat') is not None]
        n = len(pts)
        if n < 8:
            continue
        uniq = len(set(pts))
        top = max(collections.Counter(pts).values())
        if uniq / n < min_unique_frac or top / n > max_single_share:
            return True, (f'{cam}: {uniq} unique GPS positions for {n} frames '
                          f'({100.0 * uniq / n:.0f}% unique; one position covers '
                          f'{100.0 * top / n:.0f}% of frames) - track is frozen '
                          f'or duplicated')
    return False, ''


def build_footprints(site_folder, flight_logs=None, xcam_offset_frac=0.9,
                     read_exif=False, verbose=False):
    """Ground footprint of every image of a site, in one shared ENU frame.

    Returns {rel_path: {'quad': [(x,y) x4], 'center': (x,y), 'frame': n,
    'cam': c, 'pass': p}}. Headings come from each camera's own GPS track (the
    aircraft heading flips between out-and-back passes, and it - not the logged
    yaw - is what orients the footprint). The PORT/STAR across-track cant is
    applied, so footprints land where the camera actually looked rather than at
    the aircraft position.
    """
    records, cams = build_image_records(site_folder, flight_logs=flight_logs,
                                        read_exif=read_exif, verbose=verbose)
    # A frozen GPS track yields footprints that all stack on one spot, which
    # silently turns into "every frame overlaps every other" downstream. Refuse
    # it rather than emit meaningless geometry.
    bad, why = gps_degenerate(records, cams)
    if bad:
        print(f'    ** GPS track rejected: {why}')
        print('       -> no usable GPS for this site (use registration-only)')
        return {}, cams
    first = next((records[r] for rels in cams.values() for r in rels
                  if records.get(r, {}).get('lat') is not None), None)
    if first is None:
        return {}, cams
    to_enu = make_enu(first['lat'], first['lon'])

    out = {}
    for cam, rels in cams.items():
        fixed = [r for r in rels if records.get(r, {}).get('lat') is not None]
        if not fixed:
            continue
        # Order by frame so the track heading is computed along the flight path.
        fixed.sort(key=lambda r: (records[r].get('frame') is None,
                                  records[r].get('frame')))
        enu = [to_enu(records[r]['lat'], records[r]['lon']) for r in fixed]
        lat_frac = CAM_LATERAL_FRAC.get(str(cam).upper(), 0.0)
        if lat_frac:
            lat_frac = math.copysign(xcam_offset_frac, lat_frac)
        for i, r in enumerate(fixed):
            rec = records[r]
            heading = track_heading_deg(enu, i)
            quad = footprint_quad_enu(
                enu[i][0], enu[i][1], rec.get('alt_agl'), heading,
                rec.get('focal35_mm') or 85.0, lat_frac)
            cx = sum(p[0] for p in quad) / 4.0
            cy = sum(p[1] for p in quad) / 4.0
            out[r] = {'quad': quad, 'center': (cx, cy), 'enu': enu[i],
                      'frame': rec.get('frame'), 'cam': cam,
                      'pass': rec.get('pass'), 'heading': heading}
    return out, cams


def make_enu(lat0, lon0):
    """Return (to_enu(lat, lon) -> (x_east, y_north), origin) closures."""
    lat0r = math.radians(lat0)

    def to_enu(lat, lon):
        x = math.radians(lon - lon0) * EARTH_R * math.cos(lat0r)
        y = math.radians(lat - lat0) * EARTH_R
        return x, y
    return to_enu


def track_heading_deg(enu_pts, idx):
    """Heading (deg, 0=N, CW) of travel at index `idx` from the ENU track.

    Uses the neighbouring points; robust to the first/last frame. Logged
    aircraft yaw does NOT flip on out-and-back passes, so headings derived
    from the GPS track itself are preferred for footprint orientation.
    """
    n = len(enu_pts)
    if n < 2:
        return 0.0
    a = enu_pts[max(0, idx - 1)]
    b = enu_pts[min(n - 1, idx + 1)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, dy)) % 360.0


def footprint_quad_enu(x, y, alt_agl, heading_deg, focal35_mm=85.0,
                       lateral_offset_frac=0.0):
    """Metadata-only ground footprint (4x2 ENU quad) of a near-nadir image.

    Ground extent of a full-frame camera at altitude `alt_agl`:
    width = alt * 36 / f35 (across track), height = alt * 24 / f35 (along
    track). `lateral_offset_frac` shifts the footprint sideways by that
    fraction of the width - used for the PORT (-) / STAR (+) rig cameras
    whose optical axes are canted across-track relative to CENTER.

    Corners are returned in image order: (0,0), (W,0), (W,H), (0,H) i.e.
    top-left first with the top edge on the leading (flight-direction) side,
    so callers can map polygon vertices back to pixel space consistently.
    """
    f35 = focal35_mm or 85.0
    alt = alt_agl or 250.0
    w = alt * SENSOR_W_MM / f35
    h = alt * SENSOR_H_MM / f35
    # Local camera frame: u = across-track (right of travel), v = along-track.
    off = lateral_offset_frac * w
    corners_uv = [(-w / 2 + off, h / 2), (w / 2 + off, h / 2),
                  (w / 2 + off, -h / 2), (-w / 2 + off, -h / 2)]
    th = math.radians(heading_deg)
    sin_t, cos_t = math.sin(th), math.cos(th)
    quad = []
    for u, v in corners_uv:
        # Rotate camera frame into ENU: heading 0 => v axis points north.
        e = u * cos_t + v * sin_t
        n = -u * sin_t + v * cos_t
        quad.append((x + e, y + n))
    return quad


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Dump unified per-image survey metadata for a site folder')
    ap.add_argument('site_folder')
    ap.add_argument('--flight-logs', default=None,
                    help='Flight-log CSV or a directory of daily FMCLOG CSVs')
    ap.add_argument('--csv', default=None, help='Write records to CSV')
    ap.add_argument('--no-exif', action='store_true',
                    help='Skip per-image EXIF reads (faster)')
    args = ap.parse_args()

    records, cams = build_image_records(
        args.site_folder, flight_logs=args.flight_logs,
        read_exif=not args.no_exif)

    cols = ['image', 'camera', 'frame', 'time_utc', 'lat', 'lon', 'alt_agl',
            'roll', 'pitch', 'yaw', 'site', 'pass', 'source',
            'focal35_mm', 'width', 'height']
    rows = []
    for cam in cams:
        for rel in cams[cam]:
            r = records.get(rel, {})
            rows.append([rel] + [r.get(c) for c in cols[1:]])

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(f'Wrote {len(rows)} records -> {args.csv}')
    else:
        w = csv.writer(sys.stdout)
        w.writerow(cols)
        for row in rows[:20]:
            w.writerow(row)
        if len(rows) > 20:
            print(f'... ({len(rows) - 20} more rows; use --csv for all)')


if __name__ == '__main__':
    main()
