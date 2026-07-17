#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Add SAM2 segmentation polygons to an existing box-level annotation set.

The segmentation itself is done by a KWIVER pipeline (utility_add_segmentations_sam2).
This tool is everything around it: finding the units to process, working out the
frame rate each one has to run at, launching them, merging the computed polygons
back into the original annotations without disturbing the boxes, validating the
result, and re-segmenting anything that came out wrong.

Sub-commands (run `add_segmentations.py <cmd> --help` for the details):

  scan         inventory a data directory -> manifest.json (+ a printed report)
  gen-scripts  emit slurm / bash run scripts for the manifest
  run-unit     process ONE unit end to end (what the slurm tasks call)
  merge        merge computed polygons into an original CSV (standalone)
  keypoints    add head/tail keypoints to a polygon CSV (standalone)
  validate     check one unit's output against its input
  audit        validate every unit, print a summary table
  finalize     copy validated outputs over the source CSVs (backing them up)
  reseg        re-segment selected detections directly with SAM2

Typical whole-dataset run:

  add_segmentations.py scan      -i /data/FishTrack23 -r ./run
  add_segmentations.py gen-scripts -r ./run --scheduler slurm
  sbatch ./run/run_array.sbatch                 # one unit per array task
  add_segmentations.py audit     -r ./run
  add_segmentations.py finalize  -r ./run       # optional: update sources in place
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mpg', '.mpeg', '.mkv', '.wmv', '.m4v')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

DEFAULT_PIPELINE = 'utility_add_segmentations_sam2.pipe'
DEFAULT_FRAME_RATE = 5.0            # process_video's own default, used as last resort

# ocv_windowed chipping parameters, as shipped in the sam2 pipeline. Only used to
# predict whether an image is large enough to be chipped (see predict_chipping).
CHIP_ADAPTIVE_THRESH = 25000000
CHIP_SIZE = 2000
CHIP_STEP = 1000

# A polygon covering less of its box than this is treated as a failed mask rather
# than a thin object. Fish and sea lions both fill well over 10% of their box.
WEAK_FILL = 0.05

# Within one detection, a polygon smaller than this fraction of the biggest one is
# a stray speck of mask rather than a real second piece of the object.
SPECK_FRACTION = 0.05

POLY_RE = re.compile(r'\(poly\)((?:\s+-?\d+(?:\.\d+)?)+)')
FPS_RE = re.compile(r'fps["\']?\s*[:=]\s*["\']?\s*([\d.]+)')

# A trailing CSV field that opens a "(kwd) ..." token rather than a species pair.
TOKEN_RE = re.compile(r'^\([a-zA-Z_]+\)')


# -----------------------------------------------------------------------------
# VIAME CSV handling
#
# Columns: 0 id, 1 video/image identifier, 2 frame, 3-6 box (tlx tly brx bry),
#          7 confidence, 8 length, 9+ species/confidence pairs then (kwd) tokens.
# -----------------------------------------------------------------------------

def csv_rows(path):
    """Yield (line_no, raw_line, fields) for every non-comment row."""
    with open(path, errors='replace') as fin:
        for i, line in enumerate(fin):
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split(',')
            if len(fields) < 7:
                continue
            yield i, line, fields


def csv_header(path):
    """The leading comment lines of a CSV, verbatim."""
    head = []
    with open(path, errors='replace') as fin:
        for line in fin:
            if not line.startswith('#'):
                break
            head.append(line)
    return head


def row_box(fields):
    """Rounded (tlx, tly, brx, bry) -- the join key against the computed CSV."""
    return (round(float(fields[3])), round(float(fields[4])),
            round(float(fields[5])), round(float(fields[6])))


def row_frame(fields):
    return int(float(fields[2]))


def row_polys(line):
    """Every (poly) token on a row, as flat [x1, y1, x2, y2, ...] lists.

    SAM2 routinely emits a main contour plus a stray one-or-two pixel speck as a
    separate token; keeping only the first token (the speck) is what made the
    sea-lion run look like it had a 10% failure rate when it did not. Keep every
    token with at least 3 points and let the consumer treat it as a multipolygon.
    """
    out = []
    for m in POLY_RE.finditer(line):
        nums = [float(v) for v in m.group(1).split()]
        if len(nums) >= 6:
            out.append([int(round(v)) for v in nums])
    return out


def split_tokens(fields):
    """(columns, tokens) -- the id/box/species columns, and the (kwd) tokens after.

    A row can carry attributes, keypoints and notes as well as polygons. Only the
    (poly) tokens are ours to rewrite; everything else has to come through the
    merge untouched.
    """
    for i, field in enumerate(fields):
        if TOKEN_RE.match(field.strip()):
            return fields[:i], fields[i:]
    return list(fields), []


def replace_poly_tokens(fields, polys):
    """The row with its (poly) tokens swapped out, every other token preserved."""
    columns, tokens = split_tokens(fields)
    others = [t for t in tokens if not t.strip().startswith('(poly)')]
    return columns + others + [poly_token(p) for p in polys]


def poly_token(poly):
    return '(poly) ' + ' '.join(str(int(round(v))) for v in poly)


def box_poly(box):
    """The bbox itself as a rectangle polygon -- the fallback when SAM2 returns
    an empty mask, so that every detection ends up carrying some segmentation."""
    x0, y0, x1, y1 = box
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def poly_area(flat):
    xs, ys = flat[0::2], flat[1::2]
    n = len(xs)
    a = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    return abs(a) / 2.0


def drop_specks(polys, min_fraction=SPECK_FRACTION):
    """Throw away the stray fragments SAM2 leaves around the object.

    A mask routinely comes back as one real contour plus a handful of one-or-two
    pixel specks, each written as its own (poly) token -- 23 of them on one frame
    of the test clip. Keep the largest contour and anything within a reasonable
    fraction of it (a genuinely two-part object, say a fish behind an occluder,
    survives), and drop the rest.
    """
    if len(polys) < 2:
        return polys
    areas = [poly_area(p) for p in polys]
    largest = max(areas)
    if largest <= 0:
        return polys
    return [p for p, a in zip(polys, areas) if a >= min_fraction * largest]


# -----------------------------------------------------------------------------
# Frame rate resolution
#
# This is the one thing that will silently corrupt a run. Annotation frame ids
# index the stream *as the annotator saw it*, which for these video datasets is a
# downsampled one (e.g. 5 fps GT over a 29.97 fps video). The pipeline attaches GT
# frame i to the i'th frame the downsampler emits, so the downsampler has to run at
# the GT's rate or every polygon lands on the wrong picture.
# -----------------------------------------------------------------------------

def declared_fps(csv_path):
    """The fps recorded in the CSV header.

    DIVE writes both `fps: 5` and `#meta fps=5` depending on its vintage.
    process_video's rate_from_gt only matches the first form and only looks at the
    first two lines, so it misses the second; we accept either, anywhere in the
    header. (Where it misses, process_video falls back to its -frate default,
    which is why we always pass -frate explicitly.)
    """
    for line in csv_header(csv_path):
        m = FPS_RE.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def parse_timecode(text):
    """'0:01:02.5' -> 62.5 seconds. None if the field isn't a timecode."""
    if ':' not in text:
        return None
    total = 0.0
    for part in text.split(':'):
        try:
            total = total * 60 + float(part)
        except ValueError:
            return None
    return total


def derived_fps(csv_path):
    """fps implied by the CSV's own timecode column, if it has one.

    Uses deltas between consecutive frames of a track, not absolute time, because
    some clips are trimmed segments whose timecodes start partway into the source.
    """
    per_track = collections.defaultdict(list)
    for _, _, f in csv_rows(csv_path):
        t = parse_timecode(f[1])
        if t is not None:
            per_track[f[0]].append((row_frame(f), t))
    rates = []
    for points in per_track.values():
        points.sort()
        for (f0, t0), (f1, t1) in zip(points, points[1:]):
            if f1 > f0 and t1 > t0:
                rates.append((f1 - f0) / (t1 - t0))
    if not rates:
        return None
    rates.sort()
    return round(rates[len(rates) // 2], 2)


def resolve_fps(csv_path):
    """(fps_to_run_at, declared, derived). Declared wins; derived is a cross-check."""
    dec = declared_fps(csv_path)
    der = derived_fps(csv_path)
    return (dec if dec else (der if der else DEFAULT_FRAME_RATE)), dec, der


# -----------------------------------------------------------------------------
# MP4 probing
#
# VIAME's bundled OpenCV is built without FFMPEG, so cv2 cannot open these videos
# at all (it reports 0x0, 0 frames). The pipeline reads them fine via vidl_ffmpeg,
# but anything we want to know here -- resolution, frame count, native rate -- has
# to come from parsing the container ourselves.
# -----------------------------------------------------------------------------

def _atoms(fh, start, end):
    fh.seek(start)
    while fh.tell() < end - 8:
        off = fh.tell()
        hdr = fh.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack('>I4s', hdr)
        hsize = 8
        if size == 1:
            size = struct.unpack('>Q', fh.read(8))[0]
            hsize = 16
        elif size == 0:
            size = end - off
        if size < hsize:
            return
        yield typ.decode('latin1'), off + hsize, off + size
        fh.seek(off + size)


def _find(fh, start, end, path):
    for typ, dstart, dend in _atoms(fh, start, end):
        if typ == path[0]:
            if len(path) == 1:
                return dstart, dend
            found = _find(fh, dstart, dend, path[1:])
            if found:
                return found
    return None


def probe_video(path):
    """{width, height, frames, fps} from the mp4 moov atom, or None."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            moov = _find(fh, 0, size, ['moov'])
            if not moov:
                return None
            for typ, dstart, dend in _atoms(fh, *moov):
                if typ != 'trak':
                    continue
                hdlr = _find(fh, dstart, dend, ['mdia', 'hdlr'])
                mdhd = _find(fh, dstart, dend, ['mdia', 'mdhd'])
                tkhd = _find(fh, dstart, dend, ['tkhd'])
                if not (hdlr and mdhd and tkhd):
                    continue
                fh.seek(hdlr[0])
                if fh.read(12)[8:12] != b'vide':
                    continue

                fh.seek(mdhd[0])
                buf = fh.read(36)
                if buf[0] == 1:
                    timescale = struct.unpack('>I', buf[20:24])[0]
                    duration = struct.unpack('>Q', buf[24:32])[0]
                else:
                    timescale = struct.unpack('>I', buf[12:16])[0]
                    duration = struct.unpack('>I', buf[16:20])[0]

                fh.seek(tkhd[0])
                buf = fh.read(96)
                off = 76 if buf[0] == 0 else 88
                width, height = struct.unpack('>II', buf[off:off + 8])
                width >>= 16
                height >>= 16

                frames = 0
                stts = _find(fh, dstart, dend, ['mdia', 'minf', 'stbl', 'stts'])
                if stts:
                    fh.seek(stts[0])
                    count = struct.unpack('>I', fh.read(8)[4:8])[0]
                    data = fh.read(8 * count)
                    for i in range(count):
                        frames += struct.unpack('>I', data[i * 8:i * 8 + 4])[0]

                fps = (frames * timescale / duration) if duration else 0.0
                return {'width': width, 'height': height, 'frames': frames,
                        'fps': round(fps, 3)}
    except Exception:
        return None
    return None


def find_ffmpeg():
    """VIAME's own ffmpeg, else a system one. None if there isn't one.

    Deliberately does not fall back to the static ffmpeg bundled under dive/ --
    that one ships with the GUI and is not the build's video stack.
    """
    install = find_viame_install()
    if install:
        candidate = os.path.join(install, 'bin', 'ffmpeg')
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which('ffmpeg')


def extract_frame(video, frame, fps, dest, ffmpeg):
    """Pull a single frame out of a video by its annotation frame id.

    Frame ids index the stream at the annotation rate, so frame i is at i/fps
    seconds. -ss ahead of -i seeks (accurately, by default) instead of decoding
    the whole clip, which is what makes this usable one frame at a time.
    """
    timestamp = frame / float(fps)
    command = [ffmpeg, '-nostdin', '-loglevel', 'error', '-accurate_seek',
               '-ss', '%.6f' % timestamp, '-i', video,
               '-frames:v', '1', '-y', dest]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dest):
        return False
    return True


def probe_image(path):
    """(width, height) without a full decode, or None if the file is unreadable.

    Corrupt source images are not hypothetical: three of them segfaulted the
    sea-lion run, because VXL hands the refiner an empty image and the pipeline
    dies downstream. Cheaper to find them here.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Unit discovery
# -----------------------------------------------------------------------------

def find_units(data_dir):
    """Every processable unit under data_dir.

    A unit is either a video with a sibling <stem>.csv, or a directory of images
    with a CSV inside it. Both layouts appear in the same dataset.
    """
    units = []
    for entry in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, entry)
        stem, ext = os.path.splitext(entry)

        if os.path.isfile(path) and ext.lower() in VIDEO_EXTS:
            gt = os.path.join(data_dir, stem + '.csv')
            if os.path.exists(gt):
                units.append({'name': stem, 'kind': 'video',
                              'input': path, 'gt': gt})
            continue

        if os.path.isdir(path):
            images = [f for f in os.listdir(path)
                      if f.lower().endswith(IMAGE_EXTS)]
            if not images:
                continue
            gt = os.path.join(path, entry + '.csv')
            if not os.path.exists(gt):
                found = sorted(glob.glob(os.path.join(path, '*.csv')))
                if not found:
                    continue
                gt = found[0]
            units.append({'name': entry, 'kind': 'image_dir',
                          'input': path, 'gt': gt})
    return units


# A unit carrying any of these is left alone: there is either nothing to do, or
# nothing we can do without producing something wrong.
SKIP_WARNINGS = ('empty_gt', 'unreadable_video', 'fully_annotated',
                 'frame_ids_misaligned')


def is_runnable(rec):
    return not any(w in rec['warnings'] for w in SKIP_WARNINGS)


def runnable_units(manifest):
    return [r for r in manifest['units'] if is_runnable(r)]


def predict_chipping(width, height):
    """Whether ocv_windowed's adaptive mode will tile an image this size.

    Matters because when it does tile, each detection is segmented in a single
    tile and a box straddling that tile's far edge comes back with its mask cut
    off on the gridline. Recent VIAME fixes this with prefer_containing_tile
    (on by default); on an older build, run `reseg --clipped` afterwards.
    """
    return bool(width and height and width * height >= CHIP_ADAPTIVE_THRESH)


# -----------------------------------------------------------------------------
# scan
# -----------------------------------------------------------------------------

def summarize_gt(gt_path):
    n_det = 0
    max_frame = -1
    tracks = set()
    n_existing_poly = 0
    for _, line, f in csv_rows(gt_path):
        n_det += 1
        max_frame = max(max_frame, row_frame(f))
        tracks.add(f[0])
        if '(poly)' in line:
            n_existing_poly += 1
    return {'detections': n_det, 'tracks': len(tracks),
            'max_frame': max_frame, 'existing_polys': n_existing_poly}


def check_frame_alignment(gt_path, images):
    """Does each row's frame id really point at the image the row names?

    The pipeline hands GT frame i to the i'th image in sorted order, so if a CSV's
    frame ids are (say) 1-based while the directory is 0-based, every polygon lands
    one frame off -- and nothing downstream would notice, because the boxes still
    match. Only checkable when column 2 holds a filename; some exports put a stream
    id like 'Bait_1.data@7' there instead, in which case there is nothing to check.

    Returns (offset, consistent, n_checked); offset None when not checkable.
    """
    index = {name: i for i, name in enumerate(images)}
    offsets = collections.Counter()
    for _, _, f in csv_rows(gt_path):
        name = f[1].strip()
        if name and name in index:
            offsets[row_frame(f) - index[name]] += 1
    if not offsets:
        return None, True, 0
    offset, _ = offsets.most_common(1)[0]
    return offset, len(offsets) == 1, sum(offsets.values())


def cmd_scan(args):
    data_dir = os.path.abspath(args.input)
    units = find_units(data_dir)
    if not units:
        sys.exit('No units found under ' + data_dir)

    print('Scanning %d units under %s' % (len(units), data_dir))
    records = []
    for unit in units:
        rec = dict(unit)
        rec['gt'] = os.path.relpath(unit['gt'], data_dir)
        rec['input'] = os.path.relpath(unit['input'], data_dir)
        rec.update(summarize_gt(unit['gt']))

        fps, declared, derived = resolve_fps(unit['gt'])
        rec['fps'] = fps
        rec['fps_declared'] = declared
        rec['fps_derived'] = derived

        warnings = []
        if rec['detections'] == 0:
            warnings.append('empty_gt')
        elif rec['existing_polys'] == rec['detections']:
            # Every detection already has a polygon -- almost certainly hand drawn.
            # There is nothing for SAM2 to add, and the surest way not to damage
            # manual work is not to touch it.
            warnings.append('fully_annotated')

        if unit['kind'] == 'video':
            info = probe_video(unit['input'])
            if info is None:
                warnings.append('unreadable_video')
                rec['native_fps'] = rec['width'] = rec['height'] = None
                rec['frames'] = None
            else:
                rec['native_fps'] = info['fps']
                rec['frames'] = info['frames']
                rec['width'] = info['width']
                rec['height'] = info['height']
                if declared is None:
                    warnings.append('no_declared_fps')
                # The downsampler emits ~frames * fps / native_fps images; GT
                # frames past that are annotations on frames the pipeline will
                # never see, and end up relying on the box fallback.
                if info['fps']:
                    emitted = info['frames'] * fps / info['fps']
                    overrun = rec['max_frame'] + 1 - emitted
                    rec['emitted_frames'] = round(emitted, 1)
                    rec['frame_overrun'] = round(overrun, 1)
                    if overrun > 1.5:
                        warnings.append('gt_frames_exceed_video')
        else:
            images = sorted(f for f in os.listdir(unit['input'])
                            if f.lower().endswith(IMAGE_EXTS))
            rec['frames'] = len(images)
            rec['native_fps'] = None
            size = probe_image(os.path.join(unit['input'], images[0])) if images else None
            rec['width'], rec['height'] = size if size else (None, None)
            if rec['max_frame'] >= len(images):
                warnings.append('gt_frames_exceed_images')

            offset, consistent, n_checked = check_frame_alignment(unit['gt'], images)
            rec['frame_offset'] = offset
            rec['frame_offset_checked'] = n_checked
            if n_checked and (offset != 0 or not consistent):
                # Boxes would still line up perfectly in the output, so nothing
                # downstream catches this -- the polygons would just be drawn from
                # the wrong picture. Refuse to run rather than produce it.
                warnings.append('frame_ids_misaligned')

            if args.check_images:
                corrupt = [f for f in images
                           if probe_image(os.path.join(unit['input'], f)) is None]
                rec['corrupt_images'] = corrupt
                if corrupt:
                    warnings.append('corrupt_images')

        rec['chipped'] = predict_chipping(rec.get('width'), rec.get('height'))
        if rec['chipped']:
            warnings.append('will_chip')
        rec['warnings'] = warnings
        records.append(rec)

    os.makedirs(args.run_dir, exist_ok=True)
    manifest = {'data_dir': data_dir, 'units': records}
    manifest_path = os.path.join(args.run_dir, 'manifest.json')
    with open(manifest_path, 'w') as fout:
        json.dump(manifest, fout, indent=2)

    runnable = runnable_units(manifest)
    counts = collections.Counter(w for r in records for w in r['warnings'])
    rates = collections.Counter(r['fps'] for r in runnable)

    print()
    print('  units            %d  (%d to segment, %d skipped)'
          % (len(records), len(runnable), len(records) - len(runnable)))
    print('  detections       %d' % sum(r['detections'] for r in runnable))
    print('  frame rates      %s'
          % ', '.join('%g fps x%d' % (k, v) for k, v in sorted(rates.items())))
    if counts:
        print()
        print('  notes (units carrying a skip reason are left untouched):')
        for warn, n in counts.most_common():
            skip = ' [SKIPPED]' if warn in SKIP_WARNINGS else ''
            print('    %-24s %4d%s' % (warn, n, skip))
            if args.verbose:
                for r in records:
                    if warn in r['warnings']:
                        print('        ' + r['name'])
    print()
    print('Wrote ' + manifest_path)
    return 0


# -----------------------------------------------------------------------------
# merge -- fold computed polygons back into the original annotations
# -----------------------------------------------------------------------------

def load_computed_polys(computed_csv):
    """(frame, box) -> [poly, ...] from the pipeline's output CSV.

    The pipeline renumbers track ids and rewrites the identifier column, so the
    box is the only thing that survives untouched and can serve as a join key.
    Same-frame duplicate boxes are kept in a list and consumed in order.
    """
    table = collections.defaultdict(list)
    for _, line, f in csv_rows(computed_csv):
        polys = row_polys(line)
        if polys:
            table[(row_frame(f), row_box(f))].append(polys)
    return table


def merge_polys(original_csv, computed_csv, out_csv, box_fallback=True,
                clean_specks=True):
    """Write original_csv with polygons appended, boxes and rows untouched."""
    table = load_computed_polys(computed_csv)
    taken = collections.Counter()

    n_rows = n_added = n_kept = n_fallback = n_missing = n_specks = 0
    lines = list(csv_header(original_csv))

    for _, line, fields in csv_rows(original_csv):
        n_rows += 1

        # Never overwrite an annotation that already carries a polygon: those are
        # manual, and the pipeline is configured (overwrite_existing=false) not to
        # touch them either.
        if row_polys(line):
            n_kept += 1
            lines.append(line)
            continue

        key = (row_frame(fields), row_box(fields))
        candidates = table.get(key)
        idx = taken[key]
        if candidates and idx < len(candidates):
            polys = candidates[idx]
            taken[key] += 1
            n_added += 1
            if clean_specks:
                cleaned = drop_specks(polys)
                n_specks += len(polys) - len(cleaned)
                polys = cleaned
        elif box_fallback:
            polys = [box_poly(row_box(fields))]
            n_fallback += 1
        else:
            n_missing += 1
            lines.append(line)
            continue

        lines.append(','.join(replace_poly_tokens(fields, polys)) + '\n')

    with open(out_csv, 'w') as fout:
        fout.writelines(lines)

    return {'rows': n_rows, 'sam2_polys': n_added, 'kept_existing': n_kept,
            'box_fallback': n_fallback, 'no_polygon': n_missing,
            'specks_dropped': n_specks}


def cmd_merge(args):
    stats = merge_polys(args.original, args.computed, args.output,
                        box_fallback=not args.no_box_fallback,
                        clean_specks=not args.keep_specks)
    print('rows=%(rows)d sam2=%(sam2_polys)d kept=%(kept_existing)d '
          'box_fallback=%(box_fallback)d none=%(no_polygon)d '
          'specks_dropped=%(specks_dropped)d' % stats)
    print('Wrote ' + args.output)
    return 0


# -----------------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------------

def fill_ratio(polys, box):
    """How much of the bounding box the polygons actually cover."""
    x0, y0, x1, y1 = box
    box_area = max(1.0, float((x1 - x0) * (y1 - y0)))
    return sum(poly_area(p) for p in polys) / box_area


def is_weak(polys, box, threshold=WEAK_FILL):
    """A mask that covers almost none of its box.

    SAM2 sometimes returns nothing but one or two stray specks instead of the
    object -- shaped like a valid polygon, so no structural check catches it, but
    useless. These are what `reseg --weak` exists to redo.
    """
    return bool(polys) and fill_ratio(polys, box) < threshold


def validate_unit(original_csv, output_csv):
    """Confirm the output is the input plus polygons, and nothing else.

    `errors` are structural -- the annotations themselves changed -- and fail the
    unit. `warnings` are about mask quality, which is worth knowing about but is
    not a reason to reject an otherwise faithful output.
    """
    result = {'ok': True, 'errors': [], 'warnings': [], 'stats': {}}

    orig = [(row_frame(f), row_box(f), f[0]) for _, _, f in csv_rows(original_csv)]
    out = []
    n_poly = n_degenerate = n_outside = n_weak = 0
    area_ratios = []

    for _, line, f in csv_rows(output_csv):
        box = row_box(f)
        out.append((row_frame(f), box, f[0]))
        polys = row_polys(line)
        if not polys:
            continue
        n_poly += 1
        if any(len(p) < 6 for p in polys):
            n_degenerate += 1
        if is_weak(polys, box):
            n_weak += 1

        x0, y0, x1, y1 = box
        pad = 2
        for p in polys:
            xs, ys = p[0::2], p[1::2]
            if (min(xs) < x0 - pad or max(xs) > x1 + pad or
                    min(ys) < y0 - pad or max(ys) > y1 + pad):
                n_outside += 1
                break

        area_ratios.append(fill_ratio(polys, box))

    if len(orig) != len(out):
        result['ok'] = False
        result['errors'].append('detection count changed: %d -> %d'
                                % (len(orig), len(out)))
    else:
        for i, (a, b) in enumerate(zip(orig, out)):
            if a[0] != b[0] or a[1] != b[1]:
                result['ok'] = False
                result['errors'].append(
                    'row %d changed: frame/box %s -> %s' % (i, a[:2], b[:2]))
                break
        if [a[2] for a in orig] != [b[2] for b in out]:
            result['ok'] = False
            result['errors'].append('track ids changed')

    if n_outside:
        result['warnings'].append(
            '%d detections have polygon points outside the box' % n_outside)
    if n_weak:
        result['warnings'].append(
            '%d detections have a mask covering under %g of their box'
            % (n_weak, WEAK_FILL))

    area_ratios.sort()
    median = area_ratios[len(area_ratios) // 2] if area_ratios else 0.0

    result['stats'] = {
        'detections': len(out),
        'with_polygon': n_poly,
        'without_polygon': len(out) - n_poly,
        'degenerate': n_degenerate,
        'outside_box': n_outside,
        'weak': n_weak,
        # A real object's mask tends to fill 0.3-0.8 of its box. A median far below
        # that says the masks are not tracking the objects at all -- the signature
        # of SAM2 having been fed the wrong frames.
        'median_fill': round(median, 3),
    }
    return result


def cmd_validate(args):
    result = validate_unit(args.original, args.output)
    stats = result['stats']
    print('detections=%(detections)d with_poly=%(with_polygon)d '
          'without=%(without_polygon)d degenerate=%(degenerate)d weak=%(weak)d '
          'outside_box=%(outside_box)d median_fill=%(median_fill)s' % stats)
    for warn in result['warnings']:
        print('  warning: ' + warn)
    for err in result['errors']:
        print('  ERROR: ' + err)
    print('OK' if result['ok'] else 'FAILED')
    return 0 if result['ok'] else 1


# -----------------------------------------------------------------------------
# run-unit -- the whole per-clip flow, which is what the batch scripts call
# -----------------------------------------------------------------------------

def find_process_video():
    """process_video.py: beside this tool (source or install), else the install."""
    here = os.path.dirname(os.path.abspath(__file__))
    install = find_viame_install()
    for candidate in (os.path.join(here, 'process_video.py'),
                      os.path.join(install, 'configs', 'process_video.py')
                      if install else '',
                      shutil.which('process_video.py') or ''):
        if candidate and os.path.exists(candidate):
            return candidate
    sys.exit('Cannot find process_video.py beside this tool, in the VIAME '
             'install, or on PATH.')


def find_viame_install(explicit=None):
    """The VIAME install tree, wherever this tool itself is being run from.

    Pipelines, SAM2 weights and ffmpeg all live in the install, so the install has
    to be located; but the tool may be run from a source checkout or a build tree,
    so it is never assumed to be the directory we happen to sit in. Order:
    --viame-install, then $VIAME_INSTALL, then an ancestor of this file that is
    itself an install (which is what makes the installed copy work with no env).

    Deliberately does NOT go hunting for nearby directories called 'install'. A
    checkout commonly has several -- an old one, a beta, the current build -- and
    picking the wrong one silently gives you a build with no SAM2 in it.
    """
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get('VIAME_INSTALL', '')
    if env:
        return os.path.abspath(env)

    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if is_viame_install(here):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return ''
        here = parent


def is_viame_install(path):
    return bool(path) and os.path.exists(os.path.join(path, 'setup_viame.sh'))


def pipeline_candidates(pipeline, install=None):
    """Everywhere a pipeline name might resolve to, in preference order.

    Only ever an *installed* pipeline. The copies in the source tree look tempting
    but are not runnable: they locate the SAM2 weights with `relativepath`, which
    resolves against the pipe's own directory, and the models only sit beside the
    installed one.
    """
    install = find_viame_install(install)
    here = os.path.dirname(os.path.abspath(__file__))
    places = [pipeline]
    if install:
        places.append(os.path.join(install, 'configs', 'pipelines', pipeline))
    # The installed tool sits in $VIAME_INSTALL/configs/, next to pipelines/.
    places.append(os.path.join(here, 'pipelines', pipeline))
    return places


def resolve_pipeline(pipeline, install=None):
    """process_video wants a path that exists, not a pipeline name."""
    tried = pipeline_candidates(pipeline, install)
    for candidate in tried:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    install = find_viame_install(install)
    message = ['Cannot find pipeline: ' + pipeline, '', 'Looked in:']
    message += ['    ' + os.path.normpath(t) for t in tried]
    message.append('')
    if not install:
        message.append('No VIAME install found. Source setup_viame.sh, set '
                       'VIAME_INSTALL, or pass --viame-install.')
    else:
        message.append('VIAME_INSTALL = ' + install)
        pipe_dir = os.path.join(install, 'configs', 'pipelines')
        sam2 = sorted(glob.glob(os.path.join(pipe_dir, '*sam2*')))
        if sam2:
            message.append('That install does have other sam2 pipelines:')
            message += ['    ' + os.path.basename(p) for p in sam2]
            message.append('Pass one of those with -p, or a full path.')
        else:
            message.append(
                'That install has NO sam2 pipelines at all, so it was almost '
                'certainly built without the SAM2 add-on (VIAME_ENABLE_PYTORCH '
                'plus the sam2 add-on). SAM2 has to be enabled in the build for '
                'this tool to have anything to run.')
    sys.exit('\n'.join(message))


def unit_by_name(manifest, name):
    for rec in manifest['units']:
        if rec['name'] == name:
            return rec
    sys.exit('Unit not in manifest: ' + name)


def cmd_run_unit(args):
    with open(os.path.join(args.run_dir, 'manifest.json')) as fin:
        manifest = json.load(fin)
    rec = unit_by_name(manifest, args.unit)
    data_dir = manifest['data_dir']

    if not is_runnable(rec):
        print('%s: skipped (%s)' % (rec['name'], ', '.join(
            w for w in rec['warnings'] if w in SKIP_WARNINGS)))
        return 0

    input_path = os.path.join(data_dir, rec['input'])
    gt_path = os.path.join(data_dir, rec['gt'])

    work_dir = os.path.join(args.run_dir, 'work', rec['name'])
    out_dir = os.path.join(args.run_dir, 'output')
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    command = [
        sys.executable, find_process_video(),
        '-i', input_path,
        '-o', work_dir,
        '-p', resolve_pipeline(args.pipeline),
        '-gt-file', gt_path,
        '-frate', str(rec['fps']),
        # Without this, process_video stops to ask whether to wipe the output
        # folder, which under a batch scheduler means an EOF and a dead task.
        '--no-reset-prompt',
        # Keep existing polygons, and read them as polygons rather than masks so a
        # pre-annotated shape survives the round trip unchanged.
        '-s', 'detection_refiner:ocv_windowed:refiner:sam2:overwrite_existing=false',
        '-s', 'track_reader:reader:viame_csv:poly_to_mask=false',
    ]
    if rec['kind'] == 'image_dir':
        # Hold the input rate equal to the output rate so the downsampler is a
        # pass-through and frame i of the GT stays frame i of the stream.
        command += ['-ifrate', str(rec['fps'])]
    for extra in args.setting or []:
        command += ['-s', extra]

    print('$ ' + ' '.join(command), flush=True)
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print('%s: process_video FAILED (exit %d)' % (rec['name'], proc.returncode))
        return proc.returncode

    computed = os.path.join(work_dir, rec['name'] + '_tracks.csv')
    if not os.path.exists(computed):
        # For a directory input, process_video derives the output name from the
        # input path relative to itself, which collapses to '.' -- so the tracks
        # land in '._tracks.csv'. Note glob() would skip that, being a dotfile.
        found = sorted(f for f in os.listdir(work_dir)
                       if f.endswith('_tracks.csv'))
        if not found:
            print('%s: pipeline produced no _tracks.csv in %s'
                  % (rec['name'], work_dir))
            return 1
        computed = os.path.join(work_dir, found[0])

    out_csv = os.path.join(out_dir, rec['name'] + '.csv')
    stats = merge_polys(gt_path, computed, out_csv,
                        box_fallback=not args.no_box_fallback,
                        clean_specks=not args.keep_specks)
    print('%s: merged rows=%d sam2=%d kept=%d box_fallback=%d none=%d '
          'specks_dropped=%d'
          % (rec['name'], stats['rows'], stats['sam2_polys'],
             stats['kept_existing'], stats['box_fallback'], stats['no_polygon'],
             stats['specks_dropped']), flush=True)

    result = validate_unit(gt_path, out_csv)
    for warn in result['warnings']:
        print('  warning: ' + warn)
    for err in result['errors']:
        print('  ERROR: ' + err)
    print('%s: %s' % (rec['name'], 'OK' if result['ok'] else 'FAILED'))

    with open(os.path.join(out_dir, rec['name'] + '.status.json'), 'w') as fout:
        json.dump({'unit': rec['name'], 'ok': result['ok'],
                   'errors': result['errors'], 'warnings': result['warnings'],
                   'stats': result['stats'], 'merge': stats}, fout, indent=2)

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0 if result['ok'] else 1


# -----------------------------------------------------------------------------
# audit / finalize
# -----------------------------------------------------------------------------

def cmd_audit(args):
    with open(os.path.join(args.run_dir, 'manifest.json')) as fin:
        manifest = json.load(fin)
    out_dir = os.path.join(args.run_dir, 'output')

    print('%-42s %8s %8s %7s %6s %8s %6s' %
          ('unit', 'dets', 'w/ poly', 'no poly', 'weak', 'fallback', 'status'))

    todo, totals = [], collections.Counter()
    for rec in runnable_units(manifest):
        name = rec['name']
        out_csv = os.path.join(out_dir, name + '.csv')
        if not os.path.exists(out_csv):
            print('%-42s %8d %8s %7s %6s %8s %6s'
                  % (name[:42], rec['detections'], '-', '-', '-', '-', 'MISSING'))
            todo.append(name)
            continue

        result = validate_unit(os.path.join(manifest['data_dir'], rec['gt']), out_csv)
        stats = result['stats']
        fallback = 0
        status_path = os.path.join(out_dir, name + '.status.json')
        if os.path.exists(status_path):
            with open(status_path) as fin:
                fallback = json.load(fin).get('merge', {}).get('box_fallback', 0)

        status = 'OK' if result['ok'] else 'FAILED'
        if not result['ok']:
            todo.append(name)
        print('%-42s %8d %8d %7d %6d %8d %6s'
              % (name[:42], stats['detections'], stats['with_polygon'],
                 stats['without_polygon'], stats['weak'], fallback, status))

        totals['detections'] += stats['detections']
        totals['with_polygon'] += stats['with_polygon']
        totals['weak'] += stats['weak']
        totals['box_fallback'] += fallback
        totals['units'] += 1

    print()
    print('%d units validated, %d detections, %d with polygons (%.2f%%), '
          '%d box fallbacks, %d weak masks'
          % (totals['units'], totals['detections'], totals['with_polygon'],
             100.0 * totals['with_polygon'] / max(1, totals['detections']),
             totals['box_fallback'], totals['weak']))
    if totals['weak']:
        print('  weak masks cover under %g of their box; redo them per unit with:'
              % WEAK_FILL)
        print('    add_segmentations.py reseg -r %s -u <unit> --weak'
              % os.path.relpath(args.run_dir))

    if todo:
        run_dir = os.path.abspath(args.run_dir)
        todo_path = os.path.join(run_dir, 'rerun.txt')
        with open(todo_path, 'w') as fout:
            fout.write('\n'.join(todo) + '\n')
        print()
        print('%d units still need a run. Re-submit just those with:' % len(todo))
        print('  sbatch --array=0-%d%%3 --export=ALL,UNITS_FILE=%s %s/run_array.sbatch'
              % (len(todo) - 1, todo_path, run_dir))
        return 1
    return 0


def cmd_finalize(args):
    """Copy validated outputs over the source CSVs, backing the originals up."""
    with open(os.path.join(args.run_dir, 'manifest.json')) as fin:
        manifest = json.load(fin)
    out_dir = os.path.join(args.run_dir, 'output')
    data_dir = manifest['data_dir']

    pending = []
    for rec in runnable_units(manifest):
        out_csv = os.path.join(out_dir, rec['name'] + '.csv')
        gt_csv = os.path.join(data_dir, rec['gt'])
        if not os.path.exists(out_csv):
            print('SKIP %s: no output' % rec['name'])
            continue
        result = validate_unit(gt_csv, out_csv)
        if not result['ok']:
            print('SKIP %s: validation failed' % rec['name'])
            continue
        pending.append((rec['name'], out_csv, gt_csv))

    print('%d units validated and ready to write in place' % len(pending))
    if args.dry_run:
        print('(dry run -- nothing written)')
        return 0
    if not pending:
        return 1

    for name, out_csv, gt_csv in pending:
        backup = gt_csv + '.orig'
        if not os.path.exists(backup):
            shutil.copy2(gt_csv, backup)
        shutil.copy2(out_csv, gt_csv)
    print('Wrote %d CSVs in place; originals saved alongside as *.csv.orig'
          % len(pending))
    return 0


# -----------------------------------------------------------------------------
# reseg -- re-run SAM2 on selected detections, in a window that contains them
# -----------------------------------------------------------------------------

def chip_grid(width, height):
    """The tiles ocv_windowed will cut, in its own iteration order."""
    if not predict_chipping(width, height):
        return []
    regions = []
    left = 0
    while left < width - CHIP_SIZE + CHIP_STEP:
        right = min(left + CHIP_SIZE, width)
        top = 0
        while top < height - CHIP_SIZE + CHIP_STEP:
            bottom = min(top + CHIP_SIZE, height)
            regions.append((left, top, right, bottom))
            top += CHIP_STEP
        left += CHIP_STEP
    return regions


def clipped_sides(box, width, height):
    """Which tile edge, if any, truncated this detection's mask.

    Each detection is refined in the first tile it overlaps, so a box running past
    that tile's far edge comes back cut off along the gridline. Only meaningful on
    a VIAME build without prefer_containing_tile.
    """
    regions = chip_grid(width, height)
    if not regions:
        return []
    x0, y0, x1, y1 = box
    for (left, top, right, bottom) in regions:
        if min(x1, right) > max(x0, left) and min(y1, bottom) > max(y0, top):
            sides = []
            if x1 > right and right < width:
                sides.append('right')
            if y1 > bottom and bottom < height:
                sides.append('bottom')
            if x0 < left and left > 0:
                sides.append('left')
            if y0 < top and top > 0:
                sides.append('top')
            return sides
    return []


def choose_window(box, width, height):
    """A CHIP_SIZE window that fully contains the box, snapped to the tile grid.

    Snapping means detections sitting near each other share a window, and so share
    one SAM2 image embedding -- the difference between a re-seg that takes minutes
    and one that takes hours.
    """
    def origin(lo, hi, extent):
        if hi - lo > CHIP_SIZE or extent <= CHIP_SIZE:
            return int(min(max(0, (lo + hi) / 2 - CHIP_SIZE / 2),
                           max(0, extent - CHIP_SIZE)))
        omin = max(0, hi - CHIP_SIZE)
        omax = min(lo, extent - CHIP_SIZE)
        if omax < omin:
            return int(omin)
        first = ((int(omin) + CHIP_STEP - 1) // CHIP_STEP) * CHIP_STEP
        candidates = list(range(first, int(omax) + 1, CHIP_STEP)) or [int((omin + omax) / 2)]
        center = (lo + hi) / 2.0
        candidates.sort(key=lambda o: abs(o + CHIP_SIZE / 2.0 - center))
        return candidates[0]

    x0, y0, x1, y1 = box
    left = origin(x0, x1, width)
    top = origin(y0, y1, height)
    return (left, top, min(CHIP_SIZE, width - left), min(CHIP_SIZE, height - top))


def masks_to_polys(mask, offset_x, offset_y, box=None):
    """Contour a mask into polygons in full-image coordinates.

    SAM2 will happily bleed a little outside the box it was prompted with; the
    pipeline crops the mask to the detection box before contouring, so we do too,
    or a re-segmented detection would end up shaped differently from every one
    around it.
    """
    import cv2
    import numpy as np
    binary = (mask > 0).astype(np.uint8)
    if box is not None:
        keep = np.zeros_like(binary)
        x0 = max(0, box[0] - offset_x)
        y0 = max(0, box[1] - offset_y)
        x1 = min(binary.shape[1], box[2] - offset_x)
        y1 = min(binary.shape[0], box[3] - offset_y)
        if x1 > x0 and y1 > y0:
            keep[y0:y1, x0:x1] = 1
        binary = binary * keep
    if binary.sum() == 0:
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        eps = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        flat = []
        for px, py in approx:
            flat += [int(px) + offset_x, int(py) + offset_y]
        polys.append(flat)
    polys.sort(key=poly_area, reverse=True)
    return polys


def cmd_reseg(args):
    with open(os.path.join(args.run_dir, 'manifest.json')) as fin:
        manifest = json.load(fin)
    rec = unit_by_name(manifest, args.unit)

    data_dir = manifest['data_dir']
    input_path = os.path.join(data_dir, rec['input'])
    csv_path = os.path.join(args.run_dir, 'output', rec['name'] + '.csv')
    if not os.path.exists(csv_path):
        sys.exit('No output to repair: ' + csv_path)

    width, height = rec.get('width'), rec.get('height')

    # Frames come either straight off disk, or one at a time out of the video.
    # VIAME's OpenCV is built without FFMPEG and cannot open these videos itself,
    # so for a video unit we seek the frame out with ffmpeg and read the still.
    if rec['kind'] == 'image_dir':
        images = sorted(f for f in os.listdir(input_path)
                        if f.lower().endswith(IMAGE_EXTS))

        def load_frame(frame):
            import cv2
            if frame >= len(images):
                return None
            return cv2.imread(os.path.join(input_path, images[frame]))
    else:
        ffmpeg = args.ffmpeg or find_ffmpeg()
        if not ffmpeg:
            sys.exit('reseg on a video unit needs ffmpeg, and there is none in '
                     '$VIAME_INSTALL/bin or on PATH. Rebuild VIAME with '
                     'VIAME_ENABLE_FFMPEG, pass --ffmpeg, or re-run the whole clip '
                     'with run-unit.')
        scratch = os.path.join(args.run_dir, 'work', rec['name'] + '_frames')
        os.makedirs(scratch, exist_ok=True)

        def load_frame(frame):
            import cv2
            still = os.path.join(scratch, 'frame_%08d.png' % frame)
            if not os.path.exists(still):
                if not extract_frame(input_path, frame, rec['fps'], still, ffmpeg):
                    return None
            return cv2.imread(still)

    # Work out which rows to redo.
    targets = []           # (line_index, frame, box)
    lines = []
    for idx, (_, line, fields) in enumerate(csv_rows(csv_path)):
        lines.append((line, fields))
        box = row_box(fields)
        polys = row_polys(line)
        select = False
        if args.missing and (not polys or polys == [box_poly(box)]):
            select = True
        if args.weak and is_weak(polys, box, args.weak_threshold):
            select = True
        if args.clipped and clipped_sides(box, width, height):
            select = True
        if args.frames:
            lo, hi = args.frames
            if lo <= row_frame(fields) <= hi:
                select = True
        if select:
            targets.append((idx, row_frame(fields), box))

    if not targets:
        print('%s: nothing selected to re-segment' % rec['name'])
        return 0
    print('%s: re-segmenting %d detections' % (rec['name'], len(targets)))
    if args.dry_run:
        return 0

    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # The config is resolved by SAM2's own hydra search path, but the checkpoint is
    # a plain file we have to point at ourselves.
    checkpoint = args.sam2_checkpoint
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(find_viame_install(), checkpoint)
    if not os.path.exists(checkpoint):
        sys.exit('Cannot find the SAM2 checkpoint (%s). Source setup_viame.sh, set '
                 'VIAME_INSTALL, or pass --sam2-checkpoint.' % args.sam2_checkpoint)

    model = build_sam2(config_file=args.sam2_config, ckpt_path=checkpoint,
                       device='cuda', mode='eval', apply_postprocessing=True)
    predictor = SAM2ImagePredictor(model)

    # Group by (frame, window) so co-located boxes share one embedding.
    groups = collections.defaultdict(list)
    for idx, frame, box in targets:
        groups[(frame, choose_window(box, width, height))].append((idx, box))

    new_polys = {}
    n_ok = n_empty = 0
    for (frame, window), items in groups.items():
        image = load_frame(frame)
        if image is None:
            n_empty += len(items)
            continue
        image = image[:, :, ::-1]

        left, top, win_w, win_h = window
        win_w = min(win_w, image.shape[1] - left)
        win_h = min(win_h, image.shape[0] - top)
        if win_w < 2 or win_h < 2:
            n_empty += len(items)
            continue
        chip = np.ascontiguousarray(image[top:top + win_h, left:left + win_w])

        boxes = np.array([[min(max(b[0] - left, 0), win_w - 1),
                           min(max(b[1] - top, 0), win_h - 1),
                           min(max(b[2] - left, 0), win_w - 1),
                           min(max(b[3] - top, 0), win_h - 1)]
                          for _, b in items], dtype=np.float32)
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                predictor.set_image(chip)
                chunks = []
                for start in range(0, len(boxes), args.batch):
                    batch = boxes[start:start + args.batch]
                    masks, _, _ = predictor.predict(box=batch, multimask_output=False)
                    masks = np.asarray(masks)
                    if masks.ndim == 3:
                        masks = masks[:, None] if len(batch) > 1 else masks[None]
                    chunks.append(masks)
            masks = np.concatenate(chunks, axis=0)
        except Exception as exc:
            print('  window %s frame %d failed: %s' % (window, frame, exc))
            n_empty += len(items)
            torch.cuda.empty_cache()
            continue

        for k, (idx, box) in enumerate(items):
            polys = masks_to_polys(masks[k, 0], left, top, box=box)
            if not args.keep_specks:
                polys = drop_specks(polys)
            if polys:
                new_polys[idx] = polys
                n_ok += 1
            else:
                n_empty += 1
        torch.cuda.empty_cache()

    out_lines = list(csv_header(csv_path))
    for idx, (line, fields) in enumerate(lines):
        if idx in new_polys:
            out_lines.append(
                ','.join(replace_poly_tokens(fields, new_polys[idx])) + '\n')
        else:
            out_lines.append(line)
    with open(csv_path, 'w') as fout:
        fout.writelines(out_lines)

    print('%s: re-segmented %d, empty %d -> %s'
          % (rec['name'], n_ok, n_empty, csv_path))
    return 0


# -----------------------------------------------------------------------------
# keypoints -- derive head/tail keypoints from existing polygons
#
# Adds head/tail keypoints to a CSV that already carries segmentation polygons,
# in the same manner the interactive segmentation service adds them: each
# detection's polygon is rasterized to a mask and handed to the vital
# add_keypoints_from_mask algorithm, configured as the measurement / keypoint
# pipelines and the interactive service are -- 'hull_extremes' method (fit a
# min-area rectangle to the mask's convex hull and take the midpoints of its
# short edges) with clip_to_mask -- and the result is written back as (kp) head
# / (kp) tail tokens. The rasterize+algorithm step mirrors
# interactive_stereo._polygon_to_keypoints so the endpoints match a click-drawn
# one exactly. The algorithm itself only labels head/tail geometrically (larger
# x = head), so on top of it we orient head/tail per track by direction of
# travel: the head is the endpoint on the leading side of the box's motion.
# Needs the VIAME environment sourced (kwiver.vital), as reseg does.
# -----------------------------------------------------------------------------

def make_keypoint_algo(method='hull_extremes', clip_to_mask=True):
    """The add_keypoints_from_mask vital algorithm, configured to match the
    measurement / keypoint pipelines and the interactive segmentation service
    (hull_extremes method, clip_to_mask) rather than the algorithm's bare
    default."""
    try:
        from kwiver.vital.algo import RefineDetections
        from kwiver.vital.modules import modules
    except ImportError:
        sys.exit('Cannot import kwiver.vital; source setup_viame.sh so the '
                 'VIAME plugins are on the path, then re-run.')
    # A bare Python process has not scanned the plugin path the way a running
    # pipeline has, so the algorithm factories are not registered yet.
    modules.load_known_modules()
    algo = RefineDetections.create('add_keypoints_from_mask')
    if algo is None:
        sys.exit('add_keypoints_from_mask algorithm is not registered in this '
                 'VIAME install (it ships with the OpenCV plugins).')
    cfg = algo.get_configuration()
    cfg.set_value('method', method)
    cfg.set_value('clip_to_mask', 'true' if clip_to_mask else 'false')
    algo.set_configuration(cfg)
    return algo


def polygons_to_head_tail(polys, algo):
    """(head_xy, tail_xy) for a detection's polygon(s), or None.

    Rasterizes every (poly) on the detection into one mask spanning their
    combined extent and runs the vital algorithm, exactly as the interactive
    service does for a single click-drawn polygon.
    """
    import cv2
    import numpy as np
    from kwiver.vital.types import (
        DetectedObject, DetectedObjectSet, BoundingBoxD, ImageContainer, Image)

    contours = []
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 3:
            contours.append(pts)
    if not contours:
        return None

    allpts = np.concatenate(contours, axis=0)
    x0, y0 = np.floor(allpts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(allpts.max(axis=0)).astype(int)
    w, h = int(x1 - x0), int(y1 - y0)
    if w <= 0 or h <= 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [(c - [x0, y0]).astype(np.int32) for c in contours], 255)
    det = DetectedObject(
        BoundingBoxD(float(x0), float(y0), float(x1), float(y1)),
        1.0, None, ImageContainer(Image(mask)))
    dummy = ImageContainer(Image(np.zeros((h, w, 3), dtype=np.uint8)))

    refined = algo.refine(dummy, DetectedObjectSet([det]))
    dets = list(refined)
    if not dets:
        return None
    kps = dets[0].keypoints
    if 'head' not in kps or 'tail' not in kps:
        return None
    head = kps['head'].value
    tail = kps['tail'].value
    return ([float(head[0]), float(head[1])], [float(tail[0]), float(tail[1])])


def replace_keypoint_tokens(fields, head, tail):
    """Row with head/tail (kp) tokens set; polygons and other tokens preserved.

    Any pre-existing head/tail keypoint is dropped first so the command is
    idempotent; other keypoints, notes and attributes carry through untouched.
    """
    columns, tokens = split_tokens(fields)
    kept = []
    for token in tokens:
        parts = token.strip().split()
        if len(parts) >= 2 and parts[0] == '(kp)' and parts[1] in ('head', 'tail'):
            continue
        kept.append(token)
    kept.append('(kp) head %d %d' % (round(head[0]), round(head[1])))
    kept.append('(kp) tail %d %d' % (round(tail[0]), round(tail[1])))
    return columns + kept


def _row_center(fields):
    """Bounding-box center (cx, cy) from a VIAME CSV row's fields."""
    x0, y0, x1, y1 = (float(fields[3]), float(fields[4]),
                      float(fields[5]), float(fields[6]))
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def add_keypoints(input_csv, output_csv, method='hull_extremes',
                  clip_to_mask=True, orient_by_motion=True,
                  motion_window=3, min_speed=2.0, algo=None):
    """Write input_csv with head/tail keypoints added to every polygon row.

    add_keypoints_from_mask gives the two endpoints of each detection; which one
    is the head is then decided per track by direction of travel -- the head is
    the endpoint on the leading side of the box's motion (a box moving right
    gets the right-hand point as its head). Direction is estimated from the box
    center over a window of +/- motion_window track states. Tracks too short or
    too still to give a reliable direction fall back to the algorithm's
    geometric labelling (head = the endpoint with the larger x).

    Pass a pre-built algo to reuse it across many files (avoids reloading the
    kwiver plugins per call).
    """
    if algo is None:
        algo = make_keypoint_algo(method, clip_to_mask)

    # Pass 1: endpoints per row, plus the track/frame/center used for motion.
    header = list(csv_header(input_csv))
    entries = []
    n_rows = n_no_polygon = n_failed = 0
    for _, line, fields in csv_rows(input_csv):
        n_rows += 1
        polys = row_polys(line)
        if not polys:
            n_no_polygon += 1
            entries.append({'line': line})
            continue
        ends = polygons_to_head_tail(polys, algo)
        if ends is None:
            n_failed += 1
            entries.append({'line': line})
            continue
        entry = {'fields': fields, 'ends': ends}
        try:
            entry['track'] = fields[0]
            entry['frame'] = int(float(fields[2]))
            entry['center'] = _row_center(fields)
        except (ValueError, IndexError):
            pass  # no usable motion info; falls back to geometric labelling
        entries.append(entry)

    # Per-track direction of travel: windowed displacement of the box center.
    if orient_by_motion:
        by_track = collections.defaultdict(list)
        for i, entry in enumerate(entries):
            if 'center' in entry:
                by_track[entry['track']].append(i)
        for idxs in by_track.values():
            idxs.sort(key=lambda i: entries[i]['frame'])
            centers = [entries[i]['center'] for i in idxs]
            n = len(centers)
            for j, i in enumerate(idxs):
                lo = centers[max(0, j - motion_window)]
                hi = centers[min(n - 1, j + motion_window)]
                entries[i]['vel'] = (hi[0] - lo[0], hi[1] - lo[1])

    # Pass 2: assign head/tail (motion when reliable, else geometric) and write.
    n_added = n_by_motion = 0
    out = list(header)
    for entry in entries:
        if 'ends' not in entry:
            out.append(entry['line'])
            continue
        head, tail = entry['ends']  # geometric default: head = larger-x endpoint
        vel = entry.get('vel')
        if orient_by_motion and vel is not None \
                and math.hypot(vel[0], vel[1]) >= min_speed:
            # Head is the endpoint on the leading (direction-of-travel) side.
            if (head[0] - tail[0]) * vel[0] + (head[1] - tail[1]) * vel[1] < 0:
                head, tail = tail, head
            n_by_motion += 1
        out.append(','.join(replace_keypoint_tokens(entry['fields'], head, tail)) + '\n')
        n_added += 1

    with open(output_csv, 'w') as fout:
        fout.writelines(out)

    return {'rows': n_rows, 'keypoints_added': n_added,
            'oriented_by_motion': n_by_motion,
            'no_polygon': n_no_polygon, 'failed': n_failed}


def cmd_keypoints(args):
    stats = add_keypoints(args.input, args.output, method=args.method,
                          clip_to_mask=args.clip_to_mask,
                          orient_by_motion=args.orient_by_motion,
                          motion_window=args.motion_window)
    print('rows=%(rows)d keypoints_added=%(keypoints_added)d '
          'oriented_by_motion=%(oriented_by_motion)d '
          'no_polygon=%(no_polygon)d failed=%(failed)d' % stats)
    print('Wrote ' + args.output)
    return 0


# -----------------------------------------------------------------------------
# gen-scripts
# -----------------------------------------------------------------------------

WORKER = """#!/usr/bin/env bash
# Process one unit: SAM2 pipeline, then merge polygons into a copy of its CSV.
# Usage: run_unit.sh <unit-name>
set -eo pipefail

# Exported, not just assigned: setup_viame.sh normally exports this itself, but
# the tool needs it in the environment to find models and pipelines either way.
export VIAME_INSTALL="{viame}"
RUN_DIR="{run_dir}"

# Prefer the installed copy of the tool; fall back to the source tree it was
# generated from, so this works against a build that has not been reinstalled.
TOOL="$VIAME_INSTALL/configs/add_segmentations.py"
[ -f "$TOOL" ] || TOOL="{tool}"

set +u
source "$VIAME_INSTALL/setup_viame.sh"
set -u
export VIAME_INSTALL="{viame}"

# setup_viame.sh exports PYTHONPATH but does not necessarily put a matching
# interpreter on PATH, so a stray system python3 would pick up VIAME's packages
# and fail to import them. Default to the interpreter that generated this script
# (it is the one that could import VIAME), and let VIAME_PYTHON override.
PYTHON="${{VIAME_PYTHON:-{python}}}"
if [ ! -x "$PYTHON" ]; then
  PYTHON=$(command -v python3 || command -v python)
fi
[ -n "$PYTHON" ] || {{ echo "No usable python; set VIAME_PYTHON"; exit 1; }}

"$PYTHON" "$TOOL" run-unit -r "$RUN_DIR" -u "$1" -p "{pipeline}"
"""

SLURM = """#!/usr/bin/env bash
#SBATCH --job-name=add_seg
#SBATCH --output={run_dir}/logs/%A_%a.log
#SBATCH --error={run_dir}/logs/%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --array=0-{last}%{concurrent}
{extra_sbatch}
# One unit per array task, and only a few at a time. Both matter: on the sea-lion
# run, several clips sharing one GPU OOM-killed each other, and handing
# process_video a whole directory made it double-count the units and truncate
# outputs it had already finished.
#
# To re-run only the units that failed, after `audit` has written rerun.txt:
#   sbatch --array=0-$(($(wc -l < {run_dir}/rerun.txt) - 1))%{concurrent} \\
#          --export=ALL,UNITS_FILE={run_dir}/rerun.txt {run_dir}/run_array.sbatch

RUN_DIR="{run_dir}"
UNITS_FILE="${{UNITS_FILE:-$RUN_DIR/units.txt}}"
UNIT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$UNITS_FILE")

if [ -z "$UNIT" ]; then
  echo "No unit at index $SLURM_ARRAY_TASK_ID of $UNITS_FILE"
  exit 0
fi

echo "=== $UNIT on $(hostname) gpu=$CUDA_VISIBLE_DEVICES"
exec "$RUN_DIR/run_unit.sh" "$UNIT"
"""

BASH_DRIVER = """#!/usr/bin/env bash
# Fallback driver for a machine without slurm: run units across local GPUs.
# Usage: run_local.sh [n_gpus]   (default {gpus})
set -uo pipefail

RUN_DIR="{run_dir}"
NGPU="${{1:-{gpus}}}"

mkdir -p "$RUN_DIR/logs"
i=0
while read -r UNIT; do
  [ -z "$UNIT" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 2; done
  GPU=$(( i % NGPU ))
  echo "[launch] $UNIT on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$RUN_DIR/run_unit.sh" "$UNIT" \\
      > "$RUN_DIR/logs/$UNIT.log" 2>&1 &
  i=$(( i + 1 ))
  sleep 3   # stagger the model loads
done < "$RUN_DIR/units.txt"
wait
echo "ALL UNITS DONE"
"""

README = """SAM2 segmentation run
=====================

Generated by add_segmentations.py for:
    data:    {data_dir}
    units:   {n_units} runnable ({n_skipped} skipped: empty or unreadable)
    dets:    {n_dets}

Everything here is driven by add_segmentations.py, which lives in VIAME's
tools directory. Nothing else is needed.

1. Segment
--------------------------------------------------------------------------
    sbatch {run_dir}/run_array.sbatch

   (or, without slurm:  {run_dir}/run_local.sh 4)

   One unit per task. Each task runs the SAM2 pipeline over the clip and then
   merges the polygons into a copy of that clip's CSV, under {run_dir}/output/.
   The source data is not touched.

2. Check
--------------------------------------------------------------------------
    add_segmentations.py audit -r {run_dir}

   Prints per-unit detection counts, polygon coverage and pass/fail. A unit
   passes only if its output has exactly the same detections, in the same order,
   with the same boxes and track ids as its input; the only difference a pass
   permits is added (poly) tokens.

   Anything that failed or never ran is listed in {run_dir}/rerun.txt. Re-submit
   just those:

       sbatch --array=0-$(($(wc -l < {run_dir}/rerun.txt) - 1))%3 \\
              --export=ALL,UNITS_FILE={run_dir}/rerun.txt \\
              {run_dir}/run_array.sbatch

   Detections whose mask came back as specks rather than the object are counted
   as "weak". To redo just those, without re-running the whole clip:

       add_segmentations.py reseg -r {run_dir} -u <unit> --weak

   reseg re-runs SAM2 on selected detections only, in a window that fully contains
   each box, and rewrites just those polygons. Selectors combine: --weak, --missing
   (no polygon, or only the box fallback), --frames LO HI. For a video unit it
   needs ffmpeg to pull the frames out ($VIAME_INSTALL/bin/ffmpeg, or --ffmpeg);
   for an image sequence it reads them straight off disk.

3. Publish
--------------------------------------------------------------------------
    add_segmentations.py finalize -r {run_dir} --dry-run   # see what it would do
    add_segmentations.py finalize -r {run_dir}

   Copies each validated output over its source CSV, saving the original next to
   it as <name>.csv.orig. Only units that pass validation are written.

Notes
--------------------------------------------------------------------------
* Frame rate is the thing to get right. Annotation frame ids index the stream as
  the annotator saw it, which here is a downsampled one ({rates}), while the
  videos themselves are ~30 fps. Each unit's rate is recorded in manifest.json
  and passed to the pipeline explicitly, so the frame the GT means is the frame
  SAM2 sees. Do not override -frate by hand.

* Detections that SAM2 cannot segment (an empty mask, or an annotation on a frame
  past the end of the clip) fall back to the bounding box as a rectangle polygon,
  so every detection carries some segmentation. audit reports how many.

* Existing polygons are never overwritten, in the pipeline or in the merge. Units
  that are already fully polygon-annotated are skipped outright.

* SAM2 leaves stray specks of mask beside the object, each its own (poly) token.
  These are dropped on merge (--keep-specks to retain them). Note that a
  detection's polygons are a multipolygon: read every (poly) token on a row, not
  just the first, or you will pick up a speck instead of the object.
"""


def cmd_gen_scripts(args):
    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, 'manifest.json')) as fin:
        manifest = json.load(fin)

    runnable = runnable_units(manifest)
    if not runnable:
        sys.exit('No runnable units in the manifest')

    # Largest first: with a capped number of concurrent tasks, starting the long
    # clips early is the difference between a tidy tail and a very long one.
    runnable.sort(key=lambda r: -r['detections'])

    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    with open(os.path.join(run_dir, 'units.txt'), 'w') as fout:
        fout.write('\n'.join(r['name'] for r in runnable) + '\n')

    viame = find_viame_install(args.viame_install)
    if not viame:
        sys.exit('No VIAME install found. Source setup_viame.sh, set VIAME_INSTALL, '
                 'or pass --viame-install.')
    if not is_viame_install(viame):
        sys.exit('No setup_viame.sh under %s -- that is not a VIAME install.' % viame)
    tool = os.path.abspath(__file__)

    # Resolve the pipeline now, against the install these scripts will use, and
    # bake the full path in. Finding out it is missing here costs a second;
    # finding out inside the array costs every task in it.
    pipeline = resolve_pipeline(args.pipeline, install=viame)

    def emit(name, text, mode=0o755):
        path = os.path.join(run_dir, name)
        with open(path, 'w') as fout:
            fout.write(text)
        os.chmod(path, mode)
        return path

    emit('run_unit.sh', WORKER.format(viame=viame, run_dir=run_dir, tool=tool,
                                      pipeline=pipeline,
                                      python=sys.executable))

    written = ['units.txt        %d units, largest first' % len(runnable),
               'run_unit.sh      per-unit worker']

    if args.scheduler in ('slurm', 'both'):
        extra = []
        if args.partition:
            extra.append('#SBATCH --partition=' + args.partition)
        if args.account:
            extra.append('#SBATCH --account=' + args.account)
        emit('run_array.sbatch', SLURM.format(
            run_dir=run_dir, last=len(runnable) - 1, concurrent=args.concurrent,
            mem=args.mem, cpus=args.cpus, extra_sbatch='\n'.join(extra)))
        written.append('run_array.sbatch slurm array, %d tasks, %d at a time'
                       % (len(runnable), args.concurrent))

    if args.scheduler in ('bash', 'both'):
        emit('run_local.sh', BASH_DRIVER.format(run_dir=run_dir, gpus=args.gpus))
        written.append('run_local.sh     bash driver across %d GPUs' % args.gpus)

    rates = collections.Counter(r['fps'] for r in runnable)
    emit('README.txt', README.format(
        data_dir=manifest['data_dir'], run_dir=run_dir,
        n_units=len(runnable),
        n_skipped=len(manifest['units']) - len(runnable),
        n_dets=sum(r['detections'] for r in runnable),
        rates=', '.join('%g fps x%d' % (k, v) for k, v in sorted(rates.items()))),
        mode=0o644)
    written.append('README.txt       how to run it')

    print('Wrote to %s:' % run_dir)
    for line in written:
        print('  ' + line)
    print()
    if args.scheduler in ('slurm', 'both'):
        print('Next:  sbatch %s/run_array.sbatch' % run_dir)
    else:
        print('Next:  %s/run_local.sh' % run_dir)
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command')
    subs.required = True

    def add_run_dir(sub):
        sub.add_argument('-r', '--run-dir', default='./run',
                         help='directory holding manifest.json and the outputs')

    p = subs.add_parser('scan', help='inventory a data directory')
    p.add_argument('-i', '--input', required=True, help='dataset directory')
    add_run_dir(p)
    p.add_argument('--check-images', action='store_true',
                   help='open every image to find corrupt ones (slow, but they '
                        'segfault the pipeline)')
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(func=cmd_scan)

    p = subs.add_parser('gen-scripts', help='emit run scripts for a manifest')
    add_run_dir(p)
    p.add_argument('--viame-install', default='',
                   help='VIAME install directory (the one with setup_viame.sh). '
                        'Defaults to $VIAME_INSTALL, else an install found near '
                        'this tool.')
    p.add_argument('-p', '--pipeline', default=DEFAULT_PIPELINE)
    p.add_argument('--scheduler', choices=('slurm', 'bash', 'both'), default='both')
    p.add_argument('--concurrent', type=int, default=3,
                   help='max array tasks at once')
    p.add_argument('--gpus', type=int, default=4, help='GPUs for the bash driver')
    p.add_argument('--mem', default='35000')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--partition', default='')
    p.add_argument('--account', default='')
    p.set_defaults(func=cmd_gen_scripts)

    p = subs.add_parser('run-unit', help='process one unit end to end')
    add_run_dir(p)
    p.add_argument('-u', '--unit', required=True)
    p.add_argument('-p', '--pipeline', default=DEFAULT_PIPELINE)
    p.add_argument('-s', '--setting', action='append',
                   help='extra process_video -s override')
    p.add_argument('--no-box-fallback', action='store_true',
                   help='leave detections without a polygon rather than falling '
                        'back to the bounding box')
    p.add_argument('--keep-specks', action='store_true',
                   help='keep the stray mask fragments SAM2 leaves beside the '
                        'object instead of dropping them')
    p.add_argument('--keep-work', action='store_true')
    p.set_defaults(func=cmd_run_unit)

    p = subs.add_parser('merge', help='merge computed polygons into a CSV')
    p.add_argument('original')
    p.add_argument('computed')
    p.add_argument('output')
    p.add_argument('--no-box-fallback', action='store_true')
    p.add_argument('--keep-specks', action='store_true')
    p.set_defaults(func=cmd_merge)

    p = subs.add_parser('keypoints',
                        help='add head/tail keypoints to a polygon CSV')
    p.add_argument('input', help='CSV that already carries segmentation polygons')
    p.add_argument('output', help='CSV to write, with (kp) head/tail added')
    p.add_argument('--method', default='hull_extremes',
                   help='add_keypoints_from_mask method (default: hull_extremes, '
                        'as the measurement pipelines and interactive service '
                        'use); e.g. oriented_bbox, pca, skeleton')
    p.add_argument('--no-clip-to-mask', dest='clip_to_mask',
                   action='store_false',
                   help='do not clip keypoints to the mask (default: clip)')
    p.add_argument('--no-orient-by-motion', dest='orient_by_motion',
                   action='store_false',
                   help='do not use per-track direction of travel to pick the '
                        'head; keep the geometric default (head = larger-x end)')
    p.add_argument('--motion-window', type=int, default=3,
                   help='track states each side used to estimate direction of '
                        'travel (default 3)')
    p.set_defaults(func=cmd_keypoints, clip_to_mask=True, orient_by_motion=True)

    p = subs.add_parser('validate', help='check one output against its input')
    p.add_argument('original')
    p.add_argument('output')
    p.set_defaults(func=cmd_validate)

    p = subs.add_parser('audit', help='validate every unit in a run')
    add_run_dir(p)
    p.set_defaults(func=cmd_audit)

    p = subs.add_parser('finalize', help='write validated outputs over the sources')
    add_run_dir(p)
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_finalize)

    p = subs.add_parser('reseg', help='re-segment selected detections with SAM2')
    add_run_dir(p)
    p.add_argument('-u', '--unit', required=True)
    p.add_argument('--missing', action='store_true',
                   help='detections with no polygon or only the box fallback')
    p.add_argument('--weak', action='store_true',
                   help='detections whose mask barely covers their box (SAM2 '
                        'returned specks instead of the object)')
    p.add_argument('--weak-threshold', type=float, default=WEAK_FILL,
                   help='fraction of the box a mask must cover (default %g)'
                        % WEAK_FILL)
    p.add_argument('--clipped', action='store_true',
                   help='detections whose mask was cut off on a tile boundary '
                        '(only possible on a build without prefer_containing_tile)')
    p.add_argument('--frames', nargs=2, type=int, metavar=('LO', 'HI'),
                   help='every detection in this inclusive frame range')
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--keep-specks', action='store_true')
    p.add_argument('--ffmpeg', default='',
                   help='ffmpeg to pull video frames with (default: '
                        '$VIAME_INSTALL/bin/ffmpeg, else PATH)')
    p.add_argument('--sam2-config', default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    p.add_argument('--sam2-checkpoint',
                   default='configs/pipelines/models/sam2_hbp.pt')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_reseg)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
