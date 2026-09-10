#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Generate, inspect, validate and modify VIAME/KWIVER .pipe files.

Usage:
  viame pipeline info    <pipe> [--settings] [--json]
  viame pipeline get     <pipe> <key> [<key> ...]
  viame pipeline set     <pipe> -s <key>=<value> ... [-o <out>] [--dry-run]
  viame pipeline flatten <pipe> [-o <out>]
  viame pipeline check   <pipe> ...
  viame pipeline multicam --cams N --mode tracker|suppressor --detector NAME
                          [-o <out>] [...]

Keys use the same form as 'viame run -s', e.g. detector:netharn:deployed.
"""

import argparse
import json
import os
import re
import sys
import tempfile

from collections import OrderedDict
from pathlib import Path

RE_PROCESS = re.compile(r'^\s*process\s+(\S+)(?:\s*::\s*(\S+))?')
RE_TYPE = re.compile(r'^\s*::\s*(\S+)')
RE_CONFIG = re.compile(r'^\s*config\s+(\S+)')
RE_CLUSTER = re.compile(r'^\s*cluster\s+(\S+)')
RE_BLOCK = re.compile(r'^\s*block\s+(\S+)')
RE_ENDBLOCK = re.compile(r'^\s*endblock\b')
RE_INCLUDE = re.compile(r'^\s*include\s+(\S+)')
RE_RELPATH = re.compile(r'^(\s*relativepath\s+)(\S+?)(\s*=\s*)(.*)$')
RE_CONNECT = re.compile(r'^\s*connect\s+from\s+(\S+)(?:\s+to\s+(\S+))?')
RE_TO = re.compile(r'^\s*to\s+(\S+)')
RE_SETTING = re.compile(
    r'^(\s*)(:?)([A-Za-z0-9_.\-]+(?::[A-Za-z0-9_.\-]+)*)(\[[^\]]*\])?(\s*=\s*|\s+)(.*)$')
RE_COMMENT = re.compile(r'(^|\s)#')


def split_comment(line):
    m = RE_COMMENT.search(line)
    if m is None:
        return line.rstrip('\n'), ''
    cut = m.start() + (0 if m.group(1) == '' else 1)
    return line[:cut].rstrip('\n'), line[cut:].rstrip('\n')


class Entry:
    __slots__ = ('raw', 'kind', 'key', 'value', 'scope', 'lineno', 'span')

    def __init__(self, raw, kind, key=None, value=None, scope=None, lineno=0,
                 span=None):
        self.raw = raw
        self.kind = kind
        self.key = key
        self.value = value
        self.scope = scope
        self.lineno = lineno
        self.span = span


def parse(text):
    """Classify every line, tagging settings with their fully qualified key."""
    entries = []
    scope = None
    blocks = []
    pending_process = None
    pending_connect = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        code, _ = split_comment(raw)
        stripped = code.strip()
        e = Entry(raw, 'other', lineno=lineno)
        if not stripped:
            e.kind = 'comment' if raw.strip() else 'blank'
        elif (m := RE_PROCESS.match(code)):
            scope, blocks = m.group(1), []
            e.kind, e.key, e.value = 'process', m.group(1), m.group(2)
            pending_process = None if m.group(2) else e
        elif (m := RE_TYPE.match(code)) and pending_process is not None:
            pending_process.value = m.group(1)
            e.kind, e.value = 'type', m.group(1)
            pending_process = None
        elif (m := RE_CONFIG.match(code)):
            scope, blocks = m.group(1), []
            e.kind, e.key = 'config', m.group(1)
        elif (m := RE_CLUSTER.match(code)):
            scope, blocks = m.group(1), []
            e.kind, e.key = 'cluster', m.group(1)
        elif (m := RE_BLOCK.match(code)):
            blocks.append(m.group(1))
            e.kind, e.key = 'block', m.group(1)
        elif RE_ENDBLOCK.match(code):
            if blocks:
                blocks.pop()
            e.kind = 'endblock'
        elif (m := RE_INCLUDE.match(code)):
            e.kind, e.value = 'include', m.group(1)
        elif (m := RE_RELPATH.match(code)):
            e.kind = 'relativepath'
            e.key = ':'.join(filter(None, [scope] + blocks + [m.group(2)]))
            e.value = m.group(4).strip()
            e.span = (m.start(4), m.end(4))
            e.scope = scope
        elif (m := RE_CONNECT.match(code)):
            e.kind, e.key, e.value = 'connect', m.group(1), m.group(2)
            pending_connect = None if m.group(2) else e
        elif (m := RE_TO.match(code)) and pending_connect is not None:
            pending_connect.value = m.group(1)
            e.kind, e.value = 'to', m.group(1)
            pending_connect = None
        elif scope is not None and (m := RE_SETTING.match(code)):
            e.kind = 'setting'
            e.key = ':'.join([scope] + blocks + [m.group(3)])
            e.value = m.group(6).strip()
            e.span = (m.start(6), m.start(6) + len(m.group(6).rstrip()))
            e.scope = scope
        entries.append(e)
    return entries


def default_pipeline_dirs():
    here = Path(__file__).resolve().parent
    dirs = [here / 'pipelines', here.parent / 'configs' / 'pipelines']
    install = os.environ.get('VIAME_INSTALL')
    if install:
        dirs.append(Path(install) / 'configs' / 'pipelines')
    env = os.environ.get('SPROKIT_PIPE_INCLUDE_PATH', '')
    dirs = [Path(p) for p in env.split(os.pathsep) if p] + dirs
    return [d for d in dirs if d.is_dir()]


def resolve_include(name, from_file):
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for d in [Path(from_file).resolve().parent] + default_pipeline_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


class Document:
    def __init__(self):
        self.files = []
        self.processes = OrderedDict()
        self.connections = []
        self.settings = OrderedDict()
        self.relative_paths = []
        self.includes = []
        self.errors = []
        self.warnings = []
        self.entries = {}


def load(path, doc=None, seen=None, depth=0):
    doc = doc or Document()
    seen = seen or set()
    path = Path(path).resolve()
    if path in seen:
        doc.errors.append(f'{path}: include cycle')
        return doc
    seen = seen | {path}
    doc.files.append(path)
    try:
        entries = parse(path.read_text())
    except OSError as ex:
        doc.errors.append(str(ex))
        return doc
    doc.entries[path] = entries
    for e in entries:
        where = f'{path}:{e.lineno}'
        if e.kind == 'process':
            if e.key in doc.processes:
                doc.errors.append(
                    f'{where}: process "{e.key}" already defined at '
                    f'{doc.processes[e.key][1]}')
            doc.processes[e.key] = (e.value, where)
        elif e.kind == 'connect':
            doc.connections.append((e.key, e.value, where))
        elif e.kind == 'setting':
            doc.settings[e.key] = (e.value, where)
        elif e.kind == 'relativepath':
            doc.settings[e.key] = (e.value, where)
            doc.relative_paths.append((e.key, e.value, path.parent, where))
        elif e.kind == 'include':
            target = resolve_include(e.value, path)
            doc.includes.append((e.value, target, depth, where))
            if target is None:
                doc.errors.append(f'{where}: cannot find include "{e.value}"')
            else:
                load(target, doc, seen, depth + 1)
    return doc


def endpoint_process(endpoint):
    return endpoint.rsplit('.', 1)[0] if '.' in endpoint else endpoint


def write_output(text, out):
    if out in (None, '-'):
        sys.stdout.write(text)
    else:
        target = Path(out)
        fd, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', dir=target.resolve().parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(text)
            if target.exists():
                os.chmod(temporary, target.stat().st_mode)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


# ---------------------------------------------------------------------------
def cmd_info(args):
    doc = load(args.pipe)
    if args.json:
        print(json.dumps({
            'files': [str(f) for f in doc.files],
            'includes': [{'name': n, 'path': str(t) if t else None, 'depth': d}
                         for n, t, d, _ in doc.includes],
            'processes': [{'name': n, 'type': t, 'where': w}
                          for n, (t, w) in doc.processes.items()],
            'connections': [{'from': a, 'to': b, 'where': w}
                            for a, b, w in doc.connections],
            'settings': {k: {'value': v, 'where': w}
                         for k, (v, w) in doc.settings.items()},
            'errors': doc.errors,
        }, indent=2))
        return 1 if doc.errors else 0
    if doc.includes:
        print('Includes:')
        for name, target, depth, _ in doc.includes:
            state = '' if target else '  (NOT FOUND)'
            print('  ' * (depth + 1) + name + state)
        print()
    print('Processes:')
    for name, (ptype, where) in doc.processes.items():
        origin = ''
        if Path(where.rsplit(':', 1)[0]) != doc.files[0]:
            origin = '  [' + Path(where.rsplit(':', 1)[0]).name + ']'
        print(f'  {name} :: {ptype or "?"}{origin}')
    print()
    print('Connections:')
    for a, b, _ in doc.connections:
        print(f'  {a} -> {b or "?"}')
    if args.settings:
        print()
        print('Settings:')
        for k, (v, _) in doc.settings.items():
            print(f'  {k} = {v}')
    for err in doc.errors:
        print('error: ' + err, file=sys.stderr)
    return 1 if doc.errors else 0


def cmd_get(args):
    doc = load(args.pipe)
    missing = 0
    for key in args.keys:
        if key in doc.settings:
            value, where = doc.settings[key]
            suffix = f'    # {where}' if args.verbose else ''
            print(f'{key} = {value}{suffix}')
        else:
            print(f'{key}: not set', file=sys.stderr)
            missing += 1
    return 1 if missing else 0


def parse_assignment(text):
    if '=' not in text:
        raise SystemExit(f'error: setting "{text}" is not of the form key=value')
    key, value = text.split('=', 1)
    key = key.strip()
    if ':' not in key:
        raise SystemExit(f'error: key "{key}" must be at least scope:key')
    return key, value.strip()


def replace_value(entry, value):
    start, end = entry.span
    return entry.raw[:start] + value + entry.raw[end:]


def cmd_set(args):
    path = Path(args.pipe)
    doc = load(path)
    if doc.errors:
        raise SystemExit("error: " + "; ".join(doc.errors))
    entries = doc.entries[path.resolve()]
    by_key = {}
    for e in entries:
        if e.kind in ('setting', 'relativepath'):
            by_key[e.key] = e
    scopes = set(doc.processes)
    for es in doc.entries.values():
        scopes.update(e.key for e in es if e.kind in ('config', 'cluster'))
        scopes.update(e.scope for e in es if e.scope)
    overrides = OrderedDict()
    assignments = OrderedDict(parse_assignment(text) for text in args.settings)
    for key, value in assignments.items():
        if key in by_key:
            by_key[key].raw = replace_value(by_key[key], value)
            continue
        scope, rest = key.split(':', 1)
        if key in doc.settings:
            _, where = doc.settings[key]
            print(f'note: {key} is set in {where}; overriding it at the end '
                  f'of {path.name}', file=sys.stderr)
        elif scope not in scopes:
            print(f'warning: no process or config named "{scope}" in '
                  f'{path.name} or its includes', file=sys.stderr)
        overrides.setdefault(scope, []).append((rest, value))
    lines = [e.raw for e in entries]
    for scope, values in overrides.items():
        lines.append('')
        lines.append(f'config {scope}')
        for rest, value in values:
            lines.append(f'  {rest} = {value}')
    text = '\n'.join(lines) + '\n'
    if args.dry_run:
        sys.stdout.write(text)
        return 0
    write_output(text, args.output or str(path))
    return 0


def flatten_file(path, out_dir, seen, depth=0):
    path = Path(path).resolve()
    if path in seen:
        raise SystemExit(f'error: include cycle at {path}')
    seen = seen | {path}
    lines = []
    for e in parse(path.read_text()):
        if e.kind == 'include':
            target = resolve_include(e.value, path)
            if target is None:
                raise SystemExit(f'error: {path}:{e.lineno}: cannot find '
                                 f'include "{e.value}"')
            lines.append(f'# ---- begin include {e.value}')
            lines.extend(flatten_file(target, out_dir, seen, depth + 1))
            lines.append(f'# ---- end include {e.value}')
        elif e.kind == 'relativepath':
            code, comment = split_comment(e.raw)
            m = RE_RELPATH.match(code)
            target = (path.parent / e.value).resolve()
            shown = str(target)
            if out_dir is not None:
                try:
                    rel = os.path.relpath(target, out_dir)
                    if not rel.startswith('..' + os.sep + '..'):
                        shown = rel
                except ValueError:
                    pass
            if os.path.isabs(shown):
                head = m.group(1).replace('relativepath ', '')
            else:
                head = m.group(1)
            lines.append(head + m.group(2) + m.group(3) + shown + comment)
        else:
            lines.append(e.raw)
    return lines


def cmd_flatten(args):
    out_dir = None
    if args.output and args.output != '-':
        out_dir = Path(args.output).resolve().parent
    lines = flatten_file(args.pipe, out_dir, set())
    write_output('\n'.join(lines).rstrip('\n') + '\n', args.output)
    return 0


def check_document(doc, missing_files_fatal=True):
    errors = list(doc.errors)
    warnings = list(doc.warnings)
    for key, value, base, where in doc.relative_paths:
        if not (base / value).exists():
            dest = errors if missing_files_fatal else warnings
            dest.append(f'{where}: {key} refers to missing file "{value}"')
    referenced = set()
    for a, b, where in doc.connections:
        if b is None:
            errors.append(f'{where}: connect from {a} has no "to"')
            continue
        for endpoint in (a, b):
            proc = endpoint_process(endpoint)
            referenced.add(proc)
            if proc not in doc.processes:
                errors.append(f'{where}: connection refers to unknown '
                              f'process "{proc}"')
            if '.' not in endpoint:
                errors.append(f'{where}: "{endpoint}" is not of the form '
                              f'process.port')
    for name, (ptype, where) in doc.processes.items():
        if not ptype:
            errors.append(f'{where}: process "{name}" has no type')
        elif name not in referenced and len(doc.processes) > 1:
            warnings.append(f'{where}: process "{name}" is not connected')
    for name, target, depth, where in doc.includes:
        if target is not None and target.suffix != '.pipe':
            warnings.append(f'{where}: include "{name}" is not a .pipe file')
    return errors, warnings


def cmd_check(args):
    failed = 0
    for pipe in args.pipes:
        doc = load(pipe)
        errors, warnings = check_document(doc, not args.ignore_missing_files)
        for w in warnings:
            print(f'warning: {w}')
        for e in errors:
            print(f'error: {e}')
        summary = f'{len(errors)} error(s), {len(warnings)} warning(s)'
        print(f'{pipe}: {"FAIL" if errors else "ok"}, {summary}')
        failed += bool(errors)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
INPUT_INCLUDES = {
    1: 'common_default_input_with_downsampler.pipe',
    2: 'common_two_camera_input_with_downsamplers.pipe',
    3: 'common_three_camera_input_with_downsamplers.pipe',
}

SUPPRESSOR_CONFIG = """\
  suppression_poly_class = Suppressed
  remove_suppressed = false
  past_frames = all  # DIVE_PARAM["Past frames used (all or prev_neighbors)", string]
  full_homogs_file = registration_full_homogs.npz
  max_overlap_suppr_regions = 5  # DIVE_PARAM["Merge overlapping suppression regions when more than", integer]
  min_suppr_region_area_frac = 0.001  # DIVE_PARAM["Drop suppression regions smaller than this fraction of the frame", double]
  min_suppr_region_fill_ratio = 0.15  # DIVE_PARAM["Drop suppression regions filling less than this fraction of their box (warped slivers)", double]
  boundary_threshold = 1.0
"""

MULTICAM_HEADER_TAIL = """\
# Metadata File: stabilizer:flight_log
#   DIVE binds an optional per-dataset metadata file (e.g. an FMCLOG
#   flight-log CSV) to the registration stabilizer's flight_log input.
# Image List Keys: stabilizer:image_list{cam}
#   DIVE binds each camera's input image list (one line-separated file per
#   camera) to the stabilizer's per-camera image_list<i> input
#   (image_list1, image_list2, ...). Camera 1's list also locates the survey folder.
"""


def wrap_comment(text, width=77):
    import textwrap
    out = []
    for para in text.split('\n'):
        out.extend(textwrap.wrap(para, width - 2) or [''])
    return ''.join(f'# {l}'.rstrip() + '\n' for l in out)


def connect(src, dst, pad=''):
    return f'connect from {src}\n        to {pad}{dst}\n'


def generate_multicam(args):
    n = args.cams
    if n not in INPUT_INCLUDES and not args.input:
        raise SystemExit(f'error: no default input include for {n} cameras; '
                         'pass --input')
    tracker = args.mode == 'tracker'
    suffix = (lambda i: '' if n == 1 else str(i))
    down = (lambda i: 'downsampler' + suffix(i))
    det_in = args.detector_input or 'detector_input_cam{cam}'
    det_out = args.detector_output or 'detector_fuser_cam{cam}.detected_object_set'
    det_inc = args.detector_include or 'common_{detector}_cam{cam}.pipe'
    reg_inc = args.registration or 'common_sea_lion_registration_{cams}cam.pipe'
    input_inc = args.input or INPUT_INCLUDES[n]
    fmt = (lambda s, i=0: s.format(cam=i, cams=n, detector=args.detector))

    kind = 'Tracker' if tracker else 'Suppressor'
    title = args.title or f'{kind} - {args.detector.replace("_", " ").title()}'
    if n > 1:
        title += f' {n}-cam'
    if args.description:
        description = args.description
    elif tracker:
        description = ('Runs object tracking on detected objects across video '
                       'frames, linking detections of the same object from '
                       'frame to frame so repeat sightings are tied to one '
                       'individual.')
    else:
        description = ('Marks regions already observed in previous overlapping '
                       'frames with boxes so the same objects are not counted '
                       'more than once.')

    out = []
    out.append('# ' + '=' * 77 + '\n')
    out.append(f'# Title: {title}\n#\n')
    out.append(wrap_comment('Description: ' + description))
    if tracker:
        out.append('# Input: IMAGE\n# Output: TRACK\n')
    out.append('#\n')
    out.append(MULTICAM_HEADER_TAIL)
    out.append('# ' + '=' * 77 + '\n\n')

    out.append('config _scheduler\n  type = pythread_per_process\n\n')
    out.append('config _pipeline:_edge\n  capacity = 5\n\n')
    out.append('config global\n  :scale                                       '
               '1.0  # DIVE_PARAM["Detection Scale", strictly_positive_float]\n\n')
    out.append(f'include {input_inc}\n\n')
    out.append(f'include {fmt(reg_inc)}\n\n')
    for i in range(1, n + 1):
        out.append(connect(f'{down(i)}.output_1', f'stabilizer.image{i}'))
        out.append(connect(f'{down(i)}.output_2', f'stabilizer.file_name{i}'))
    out.append('\n')
    for i in range(1, n + 1):
        out.append(f'include {fmt(det_inc, i)}\n\n')
        out.append(connect(f'{down(i)}.output_1', f'{fmt(det_in, i)}.image'))
        out.append('\n')

    main = 'tracker' if tracker else 'suppressor'
    ptype = 'multicam_homog_tracker' if tracker else 'multicam_homog_det_suppressor'
    out.append(f'process {main} :: {ptype}\n  n_input = {n}\n')
    if not tracker:
        out.append(SUPPRESSOR_CONFIG)
    out.append('\n')
    for i in range(1, n + 1):
        out.append(connect(f'stabilizer.homog{i}', f'{main}.homog{i}'))
    out.append('\n')
    for i in range(1, n + 1):
        out.append(connect(fmt(det_out, i), f'{main}.det_objs_{i}'))
    out.append('\n')
    if tracker:
        out.append(connect(f'{down(1)}.timestamp', 'tracker.timestamp'))
    else:
        for i in range(1, n + 1):
            out.append(connect(f'{down(i)}.output_1', f'suppressor.image{i}'))
            out.append(connect(f'{down(i)}.output_2', f'suppressor.file_name{i}'))
    out.append('\n')

    for i in range(1, n + 1):
        s = suffix(i)
        if tracker:
            out.append(f'process track_classifier{s} :: refine_tracks\n'
                       '  refiner:type = average_tot\n'
                       '  refiner:average_tot:tot_option = weighted_average\n\n')
            out.append(f'process track_writer{s} :: write_object_track\n'
                       f'  file_name = tracks{s}.csv\n'
                       f'  frame_list_output = track_images{"_" + s if s else ""}.txt\n'
                       '  writer:type = viame_csv\n\n')
            out.append(connect(f'tracker.obj_tracks_{i}',
                               f'track_classifier{s}.object_track_set'))
            out.append(connect(f'track_classifier{s}.object_track_set',
                               f'track_writer{s}.object_track_set'))
            out.append(connect(f'{down(i)}.timestamp',
                               f'track_writer{s}.timestamp'))
            out.append(connect(f'{down(i)}.output_2',
                               f'track_writer{s}.image_file_name'))
        else:
            out.append(f'process detector_writer{s} :: detected_object_output\n'
                       f'  file_name = detections{s}.csv\n'
                       f'  frame_list_output = det_images{"_" + s if s else ""}.txt\n'
                       '  writer:type = viame_csv\n\n')
            out.append(connect(f'suppressor.det_objs_{i}',
                               f'detector_writer{s}.detected_object_set'))
            out.append(connect(f'{down(i)}.output_2',
                               f'detector_writer{s}.image_file_name'))
        out.append('\n')

    if not args.no_homographies:
        for i in range(1, n + 1):
            s = suffix(i)
            out.append(f'process homog_writer{s}\n  :: kw_write_homography\n'
                       f'  output = homogs{s}.txt\n\n')
            out.append(connect(f'stabilizer.homog{i}',
                               f'homog_writer{s}.homography'))
            out.append('\n')
    return ''.join(out).rstrip('\n') + '\n'


def cmd_multicam(args):
    text = generate_multicam(args)
    out = args.output
    if out is None and args.output_dir:
        kind = 'tracker' if args.mode == 'tracker' else 'suppressor'
        cams = f'_{args.cams}-cam' if args.cams > 1 else ''
        out = str(Path(args.output_dir) / f'{kind}_{args.detector}{cams}.pipe')
    write_output(text, out)
    if out and out != '-':
        print(f'wrote {out}', file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog='viame pipeline',
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n', 2)[2])
    sub = p.add_subparsers(dest='command', metavar='<command>')
    sub.required = True

    s = sub.add_parser('info', help='List includes, processes, connections')
    s.add_argument('pipe')
    s.add_argument('--settings', action='store_true',
                   help='Also list every configuration key and value')
    s.add_argument('--json', action='store_true')
    s.set_defaults(func=cmd_info)

    s = sub.add_parser('get', help='Print the value of configuration keys')
    s.add_argument('pipe')
    s.add_argument('keys', nargs='+', metavar='key')
    s.add_argument('-v', '--verbose', action='store_true',
                   help='Show which file and line sets each key')
    s.set_defaults(func=cmd_get)

    s = sub.add_parser('set', help='Change configuration values')
    s.add_argument('pipe')
    s.add_argument('-s', '--setting', dest='settings', action='append',
                   required=True, metavar='key=value')
    s.add_argument('-o', '--output',
                   help='Write here instead of editing the file in place')
    s.add_argument('--dry-run', action='store_true',
                   help='Print the result instead of writing it')
    s.set_defaults(func=cmd_set)

    s = sub.add_parser('flatten',
                       help='Inline every include into one self-contained file')
    s.add_argument('pipe')
    s.add_argument('-o', '--output', help='Output file (default: stdout)')
    s.set_defaults(func=cmd_flatten)

    s = sub.add_parser('check', help='Validate includes, paths and connections')
    s.add_argument('pipes', nargs='+', metavar='pipe')
    s.add_argument('--ignore-missing-files', action='store_true',
                   help='Report missing relativepath targets (e.g. models '
                        'not yet downloaded) as warnings instead of errors')
    s.set_defaults(func=cmd_check)

    s = sub.add_parser('multicam',
                       help='Generate a stabilized multi-camera tracker or '
                            'suppressor pipeline')
    s.add_argument('--cams', type=int, required=True)
    s.add_argument('--mode', choices=['tracker', 'suppressor'], required=True)
    s.add_argument('--detector', required=True,
                   help='Detector name, e.g. sea_lion_fusion_two_class')
    s.add_argument('--detector-include', metavar='TEMPLATE',
                   help='Per-camera detector include '
                        '(default: common_{detector}_cam{cam}.pipe)')
    s.add_argument('--detector-input', metavar='TEMPLATE',
                   help='Process receiving each camera image '
                        '(default: detector_input_cam{cam})')
    s.add_argument('--detector-output', metavar='TEMPLATE',
                   help='Port producing each camera\'s detections '
                        '(default: detector_fuser_cam{cam}.detected_object_set)')
    s.add_argument('--registration', metavar='TEMPLATE',
                   help='Stabilizer include '
                        '(default: common_sea_lion_registration_{cams}cam.pipe)')
    s.add_argument('--input', metavar='FILE',
                   help='Camera input include (default chosen by --cams)')
    s.add_argument('--title')
    s.add_argument('--description')
    s.add_argument('--no-homographies', action='store_true',
                   help='Do not write per-camera homography files')
    s.add_argument('-o', '--output', help='Output file (default: stdout)')
    s.add_argument('--output-dir',
                   help='Write to <mode>_<detector>[_<N>-cam].pipe in this dir')
    s.set_defaults(func=cmd_multicam)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
