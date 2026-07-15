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
import csv
import glob
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

    `logs` may be a CSV path (returned as-is), or a directory searched for
    SSL-FMCLOG_YYYY-MM-DD_*.csv (an '_edited' variant is preferred when both
    exist since those contain manual corrections).
    """
    if logs is None or date is None:
        return None
    if os.path.isfile(logs):
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
                out['heading'] = float(g['GPSImgDirection'])
        except (KeyError, ValueError, TypeError):
            pass
    return out


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

    records = {}
    n_log = n_exif_gps = 0
    time_deltas = []
    for cam, rel_paths in cams.items():
        for rel in rel_paths:
            info = parse_image_filename(rel)
            rec = {}
            row = log.get(info['frame']) if info['frame'] is not None else None
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
