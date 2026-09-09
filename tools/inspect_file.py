#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Identify a file, check it is intact, and say how VIAME can use it.

Usage:
  viame inspect PATH [PATH ...] [--json]

Handles imagery and video, image lists and folders, pipelines and training
configurations, model files, VIAME CSV, DIVE and COCO JSON, and archives.
Each report says what the file is, whether its format is standard, whether it
is corrupt, how it relates to VIAME, and the command that runs it if any.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from viame.core import model_wrap

VIDEO_EXTS = (
    "3qp;3g2;amv;asf;avi;drc;gif;gifv;f4v;f4p;f4a;f4bflv;m4v;mkv;mp4;m4p;"
    "mpg;mpg2;mp2;mpeg;mpe;mpv;mng;mts;m2ts;mov;mxf;nsv;ogg;ogv;qt;roq;rm;"
    "rmvb;svi;webm;wmv;vob;yuv").split(';')
IMAGE_EXTS = (
    "bmp;dds;gif;heic;jpg;jpeg;png;psd;psp;pspimage;tga;thm;tif;tiff;"
    "yuv").split(';')
MODEL_EXTS = ('pt', 'pth', 'ckpt', 'weights', 'onnx', 'zip')

# Magic numbers of the still image formats VIAME's readers accept
IMAGE_MAGIC = [
    (b'\xff\xd8\xff', 'JPEG', ('jpg', 'jpeg', 'thm')),
    (b'\x89PNG\r\n\x1a\n', 'PNG', ('png',)),
    (b'GIF87a', 'GIF', ('gif',)),
    (b'GIF89a', 'GIF', ('gif',)),
    (b'BM', 'BMP', ('bmp',)),
    (b'II*\x00', 'TIFF', ('tif', 'tiff')),
    (b'MM\x00*', 'TIFF', ('tif', 'tiff')),
    (b'DDS ', 'DDS', ('dds',)),
    (b'8BPS', 'Photoshop', ('psd',)),
]


class Report:
    def __init__(self, path):
        self.path = path
        self.category = 'unknown'
        self.detail = ''
        self.format = 'unknown'
        self.integrity = 'unknown'
        self.relation = 'not something VIAME reads'
        self.runnable = False
        self.command = ''
        self.notes = []

    def corrupt(self, why):
        self.integrity = 'corrupt: ' + why
        self.runnable = False
        self.command = ''

    def as_dict(self):
        return {
            'path': self.path,
            'category': self.category,
            'detail': self.detail,
            'format': self.format,
            'integrity': self.integrity,
            'viame': self.relation,
            'runnable': self.runnable,
            'command': self.command,
            'notes': self.notes,
        }

    def text(self):
        lines = [self.path]
        rows = [
            ('type', self.category + (' (' + self.detail + ')' if self.detail else '')),
            ('format', self.format),
            ('integrity', self.integrity),
            ('viame', self.relation),
            ('runnable', ('yes: ' + self.command) if self.runnable else 'no'),
        ]
        for key, value in rows:
            lines.append('  %-10s %s' % (key + ':', value))
        for note in self.notes:
            lines.append('  %-10s %s' % ('note:', note))
        return '\n'.join(lines)


# -----------------------------------------------------------------------------
def ext_of(path):
    return os.path.splitext(path)[1].lower().lstrip('.')


def read_head(path, n=16):
    try:
        with open(path, 'rb') as f:
            return f.read(n)
    except OSError:
        return b''


def sniff_image_magic(head):
    for magic, name, exts in IMAGE_MAGIC:
        if head.startswith(magic):
            return name, exts
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'WebP', ('webp',)
    return None, ()


def sniff_video(head):
    if head[4:8] in (b'ftyp', b'moov', b'mdat'):
        return 'MP4/QuickTime'
    if head.startswith(b'\x1a\x45\xdf\xa3'):
        return 'Matroska/WebM'
    if head.startswith(b'RIFF') and head[8:12] == b'AVI ':
        return 'AVI'
    if head.startswith(b'\x30\x26\xb2\x75'):
        return 'ASF/WMV'
    if head.startswith(b'\x00\x00\x01\xba') or head.startswith(b'\x00\x00\x01\xb3'):
        return 'MPEG program stream'
    if head.startswith(b'FLV'):
        return 'FLV'
    if head.startswith(b'OggS'):
        return 'Ogg'
    return None


# -----------------------------------------------------------------------------
def inspect_image(path, report):
    report.category = 'image'
    report.relation = 'input imagery for detection, tracking and training'
    head = read_head(path)
    magic_name, magic_exts = sniff_image_magic(head)
    ext = ext_of(path)

    if magic_name is None and ext not in ('yuv', 'heic', 'tga', 'psp', 'pspimage'):
        report.format = 'no recognizable image header'
        report.corrupt('does not start like a %s file' % ext.upper())
        return

    if magic_name and ext not in magic_exts:
        report.format = 'non-standard: .%s extension holds %s data' % (ext, magic_name)
    elif magic_name:
        report.format = 'standard ' + magic_name
    else:
        report.format = ext.upper() + ' (no header to verify)'

    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
            frames = getattr(im, 'n_frames', 1)
            im.load()
    except ImportError:
        report.integrity = 'header ok (PIL unavailable for a full decode)'
        report.detail = magic_name or ext.upper()
    except Exception as e:
        report.corrupt('decode failed: %s' % str(e).strip() or type(e).__name__)
        return
    else:
        report.integrity = 'ok, decodes fully'
        bits = '16-bit' if mode.startswith('I;16') or mode == 'I' else '8-bit'
        report.detail = '%dx%d, %s %s' % (width, height, bits, mode)
        if frames > 1:
            report.notes.append('%d frames; VIAME reads the first' % frames)
        if bits == '16-bit':
            report.notes.append('16-bit imagery: use the *_16bit pipeline templates')

    report.runnable = True
    report.command = 'viame run <pipeline> %s' % path


def ffprobe(path):
    if shutil.which('ffprobe') is None:
        return None
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-print_format', 'json',
             '-show_format', '-show_streams', path],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return {'error': out.stderr.strip()}
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


def decodes_first_frame(path):
    """True/False when a decoder settled it, None when none was available."""
    if shutil.which('ffmpeg') is not None:
        try:
            out = subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', path, '-frames:v', '1',
                 '-f', 'null', '-'],
                capture_output=True, text=True, timeout=120)
            return out.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        ok, _frame = cap.read()
        cap.release()
        # An OpenCV without a video backend opens nothing, so only success counts
        return True if ok else None
    except Exception:
        return None


def inspect_video(path, report):
    report.category = 'video'
    report.relation = 'input video for detection, tracking and training'
    container = sniff_video(read_head(path))
    ext = ext_of(path)

    probe = ffprobe(path)
    if probe is not None and 'error' in probe:
        report.format = container or ext.upper()
        report.corrupt('ffprobe cannot parse it: ' + probe['error'].splitlines()[-1])
        return

    if probe:
        fmt = probe.get('format', {})
        streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
        if not streams:
            report.format = fmt.get('format_long_name', container or ext.upper())
            report.corrupt('no video stream inside')
            return
        v = streams[0]
        rate = v.get('avg_frame_rate', '0/1')
        try:
            num, den = rate.split('/')
            fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            fps = 0.0
        report.detail = '%s, %sx%s, %.3g fps, %s s' % (
            v.get('codec_name', '?'), v.get('width', '?'), v.get('height', '?'),
            fps, fmt.get('duration', '?'))
        report.format = 'standard ' + fmt.get('format_long_name', fmt.get('format_name', ''))
        if container and fmt.get('format_name', '') and \
                container.split('/')[0].lower() not in fmt.get('format_name', '').lower():
            report.notes.append('extension .%s does not match the %s container'
                                % (ext, fmt.get('format_name')))
    else:
        report.format = container or (ext.upper() + ' (unverified)')

    decoded = decodes_first_frame(path)
    if decoded is False:
        report.corrupt('first frame does not decode')
        return
    report.integrity = 'ok, first frame decodes' if decoded else \
        ('ok per ffprobe, no decoder available to check frames' if probe
         else 'unverified, no ffprobe or decoder available')
    report.runnable = True
    report.command = 'viame run <pipeline> %s' % path


def inspect_image_list(path, report, lines):
    report.category = 'image list'
    entries = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
    base = os.path.dirname(os.path.abspath(path))
    missing = [e for e in entries
               if not os.path.isfile(e) and not os.path.isfile(os.path.join(base, e))]
    report.detail = '%d entries' % len(entries)
    report.format = 'one image path per line'
    report.relation = 'input image list for detection, tracking and training'
    if not entries:
        report.corrupt('no entries')
        return
    if missing:
        report.corrupt('%d of %d listed images are missing, e.g. %s'
                       % (len(missing), len(entries), missing[0]))
        return
    report.integrity = 'ok, every listed image exists'
    report.runnable = True
    report.command = 'viame run <pipeline> %s' % path


def inspect_folder(path, report):
    report.category = 'folder'
    images = videos = subfolders = 0
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            subfolders += 1
        elif ext_of(name) in IMAGE_EXTS:
            images += 1
        elif ext_of(name) in VIDEO_EXTS:
            videos += 1
    report.detail = '%d images, %d videos, %d subfolders' % (images, videos, subfolders)
    report.format = 'directory'
    if images or videos or subfolders:
        report.integrity = 'ok'
        report.relation = 'input folder; images process as one sequence, videos one by one'
        report.runnable = True
        report.command = 'viame run <pipeline> %s' % path
    else:
        report.integrity = 'ok, but empty of imagery'
        report.relation = 'nothing for VIAME to process'


# -----------------------------------------------------------------------------
def inspect_pipeline(path, report):
    report.category = 'pipeline'
    report.format = 'kwiver pipe file'
    with open(path, errors='replace') as f:
        text = f.read()

    staged = any(l.strip().startswith('pipeline stage ') for l in text.splitlines())
    embedded = model_wrap.is_embedded_pipeline(text)

    try:
        import pipeline as pipe_tool
        doc = pipe_tool.load(path)
        errors, warnings = pipe_tool.check_document(doc)
    except Exception as e:
        report.corrupt('does not parse: %s' % e)
        return

    report.detail = '%d processes, %d connections' % (len(doc.processes), len(doc.connections))
    if staged:
        report.detail += ', staged'
    for w in warnings:
        report.notes.append('warning: ' + w)
    if errors:
        report.corrupt('; '.join(errors[:3]))
        return
    report.integrity = 'ok, includes and referenced files resolve'

    if embedded:
        report.relation = ('embedded pipeline: no reader or writer of its own, ' +
                           'as written by training for DIVE')
        report.notes.append('zip it with its model files and viame run will wrap it')
        return
    report.relation = 'complete pipeline'
    report.runnable = True
    report.command = 'viame run %s <video|image|image-list|folder>' % path


def inspect_conf(path, report):
    with open(path, errors='replace') as f:
        text = f.read()
    report.category = 'configuration'
    report.format = 'kwiver config block'
    report.integrity = 'ok, readable text'
    lowered = text.lower()
    if 'trainer:type' in lowered or 'pipeline_template' in lowered:
        report.category = 'training configuration'
        report.relation = 'parameters for viame train'
        report.runnable = True
        report.command = 'viame train -c %s <training data>' % path
    else:
        report.relation = 'settings block applied with viame run ... -c'


# -----------------------------------------------------------------------------
def zip_is_intact(path):
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        return None if bad is None else 'member %s fails its checksum' % bad
    except zipfile.BadZipFile as e:
        return str(e)


def inspect_model(path, report):
    ext = ext_of(path)
    report.category = 'model'
    if ext == 'zip':
        problem = zip_is_intact(path)
        if problem:
            report.format = 'zip archive'
            report.corrupt(problem)
            return
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith('/')]
        if any(n.replace('\\', '/').startswith('configs/pipelines/') for n in names):
            report.category = 'add-on model pack'
            report.format = 'zip laid out like a VIAME install'
            report.integrity = 'ok, archive verifies'
            report.relation = 'installs pipelines and models into configs/pipelines'
            report.runnable = True
            report.command = 'viame add-ons install <name> --from-file %s' % path
            pipes = [n for n in names if n.endswith('.pipe')]
            report.notes.append('%d pipelines inside; viame run %s asks which to run'
                                % (len(pipes), path))
            return
    elif ext in ('pt', 'pth', 'ckpt'):
        if zipfile.is_zipfile(path):
            problem = zip_is_intact(path)
            if problem:
                report.format = 'PyTorch zip checkpoint'
                report.corrupt(problem)
                return
        elif not read_head(path, 2).startswith(b'\x80'):
            report.format = 'not a PyTorch checkpoint'
            report.corrupt('neither a zip archive nor a pickle stream')
            return
    elif ext == 'onnx':
        try:
            import onnx
            onnx.checker.check_model(onnx.load(path))
            report.notes.append('onnx.checker accepts the graph')
        except ImportError:
            pass
        except Exception as e:
            report.format = 'ONNX'
            report.corrupt('onnx rejects it: %s' % str(e).splitlines()[0])
            return

    work_dir = tempfile.mkdtemp(prefix='viame_inspect_')
    try:
        info = model_wrap.identify(path, work_dir)
    finally:
        shutil.rmtree(work_dir, True)

    report.detail = info.kind
    if info.kind == 'pipeline_zip':
        report.category = 'packaged pipeline'
        report.detail = '%d pipeline file(s): %s' % (len(info.pipes), ', '.join(info.pipes))
        report.format = 'zip with .pipe files'
        report.integrity = 'ok, archive verifies'
        report.relation = 'pipeline plus its model files, as written by viame train'
        report.runnable = True
        report.command = 'viame run %s <video|image|image-list|folder>' % path
        return

    report.format = {
        'pt': 'PyTorch checkpoint', 'pth': 'PyTorch checkpoint',
        'ckpt': 'Lightning checkpoint', 'weights': 'Darknet weights',
        'onnx': 'ONNX', 'zip': 'zip archive'}.get(ext, ext)
    if report.integrity == 'unknown':
        report.integrity = 'ok, readable'
    if not info.runnable:
        report.relation = info.reason
        report.notes.append('recognized as ' + info.kind + ' but not runnable as is')
        return
    report.relation = 'runs with the %s %s' % (
        info.impl, 'full-frame classifier' if info.classifier else 'detector')
    report.runnable = True
    report.command = 'viame run %s <video|image|image-list|folder>' % path


# -----------------------------------------------------------------------------
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def inspect_csv(path, report):
    report.category = 'CSV'
    with open(path, newline='', errors='replace') as f:
        raw = f.read()
    lines = raw.splitlines()
    header = any(l.startswith('# 1: Detection or Track-id') for l in lines[:3])
    rows = []
    bad = 0
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        cols = next(csv.reader([line]))
        if len(cols) >= 9 and all(is_number(c) for c in cols[2:9]):
            rows.append(cols)
        else:
            bad += 1

    if not rows and not header:
        report.format = 'generic CSV, %d rows' % bad
        report.integrity = 'ok, readable text'
        report.relation = 'not in the VIAME CSV layout'
        return

    report.category = 'VIAME CSV annotations'
    report.format = 'VIAME CSV' + (' with standard header' if header else ' without its header comment')
    tracks = {r[0] for r in rows}
    frames = {r[2] for r in rows}
    classes = set()
    pairs_bad = 0
    for r in rows:
        i = 9
        while i + 1 < len(r) and r[i] and not r[i].startswith('('):
            if is_number(r[i + 1]):
                classes.add(r[i])
            else:
                pairs_bad += 1
            i += 2
    report.detail = '%d rows, %d tracks, %d frames, %d classes' % (
        len(rows), len(tracks), len(frames), len(classes))
    report.relation = 'detections or tracks; read by DIVE, viame csv, scoring and training'
    if bad:
        report.corrupt('%d row(s) do not have the 9 leading columns' % bad)
        return
    if pairs_bad:
        report.corrupt('%d species entries lack a numeric confidence' % pairs_bad)
        return
    report.integrity = 'ok, every row parses'
    report.notes.append('viame csv --help lists the edits available')


def inspect_json(path, report):
    report.category = 'JSON'
    try:
        with open(path, errors='replace') as f:
            doc = json.load(f)
    except ValueError as e:
        report.format = 'JSON'
        report.corrupt('invalid JSON: %s' % e)
        return
    report.integrity = 'ok, parses'
    if isinstance(doc, dict) and isinstance(doc.get('tracks'), dict):
        report.category = 'DIVE annotations'
        report.format = 'DIVE JSON version %s' % doc.get('version', '?')
        report.detail = '%d tracks, %d groups' % (
            len(doc['tracks']), len(doc.get('groups', {}) or {}))
        report.relation = 'annotations; read by DIVE, viame json, scoring and training'
        report.notes.append('viame json --help lists the edits available')
    elif isinstance(doc, dict) and 'images' in doc and 'annotations' in doc:
        report.category = 'COCO annotations'
        report.format = 'MS-COCO JSON'
        report.detail = '%d images, %d annotations, %d categories' % (
            len(doc.get('images', [])), len(doc.get('annotations', [])),
            len(doc.get('categories', [])))
        report.relation = 'annotations; read by viame json, training and DIVE import'
        report.notes.append('viame json --help lists the edits available')
    elif isinstance(doc, dict) and 'cameras' in doc or (
            isinstance(doc, dict) and any(k in doc for k in ('camera_left', 'camera_right'))):
        report.category = 'camera calibration'
        report.format = 'VIAME camera JSON'
        report.relation = 'stereo calibration for measurement pipelines'
    else:
        report.format = 'generic JSON (%s)' % type(doc).__name__
        report.relation = 'not a VIAME annotation layout'


# -----------------------------------------------------------------------------
def inspect_path(path):
    report = Report(path)
    if not os.path.exists(path):
        report.corrupt('does not exist')
        report.integrity = 'missing'
        return report
    if os.path.isdir(path):
        inspect_folder(path, report)
        return report
    if os.path.getsize(path) == 0:
        report.corrupt('empty file')
        return report

    ext = ext_of(path)
    if ext == 'pipe':
        inspect_pipeline(path, report)
    elif ext == 'conf':
        inspect_conf(path, report)
    elif ext == 'csv':
        inspect_csv(path, report)
    elif ext == 'json':
        inspect_json(path, report)
    elif ext in MODEL_EXTS:
        inspect_model(path, report)
    elif ext in IMAGE_EXTS or sniff_image_magic(read_head(path))[0]:
        inspect_image(path, report)
    elif ext in VIDEO_EXTS or sniff_video(read_head(path)):
        inspect_video(path, report)
    elif ext == 'txt':
        with open(path, errors='replace') as f:
            inspect_image_list(path, report, f.readlines())
    else:
        head = read_head(path)
        report.format = 'binary' if any(b < 9 for b in head[:8]) else 'text'
        report.integrity = 'ok, readable'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='viame inspect',
        description='Identify a file, check it is intact, and say how VIAME can use it.')
    parser.add_argument('paths', nargs='+', metavar='PATH')
    parser.add_argument('--json', action='store_true',
                        help='Print one JSON object per path instead of text')
    args = parser.parse_args(argv)

    reports = [inspect_path(p) for p in args.paths]
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    else:
        print('\n\n'.join(r.text() for r in reports))
    return 1 if any(r.integrity.startswith(('corrupt', 'missing')) for r in reports) else 0


if __name__ == '__main__':
    sys.exit(main())
