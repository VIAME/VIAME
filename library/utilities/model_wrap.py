#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Wrap a bare model file in a runnable detector pipeline.

`viame run` accepts a .pt/.pth/.ckpt checkpoint or a .zip in place of a
pipeline. This module works out which detector implementation can load the
file, without importing any deep learning framework, and renders the default
detector template around it.
"""
import io
import json
import os
import pickletools
import re
import tempfile
import zipfile

MODEL_EXTS = ('.pt', '.pth', '.ckpt', '.weights', '.onnx', '.zip')

_TORCH_EXTS = ('.pt', '.pth', '.ckpt')
_DARKNET_EXTS = ('.weights', '.wt')
_CLASS_LIST_EXTS = ('.lbl', '.names', '.txt')

_MAX_PICKLE_BYTES = 64 * 1024 * 1024


class ModelInfo:
    def __init__(self, path, kind, impl='', keys=None, runnable=True,
                 reason='', mode='disabled', pipes=None, windowed=True):
        self.path = os.path.abspath(path)
        self.kind = kind
        self.impl = impl
        self.keys = keys or {}
        self.runnable = runnable
        self.reason = reason
        self.mode = mode
        # Whether to nest the detector in ocv_windowed for resizing
        self.windowed = windowed
        # A whole-frame classifier rather than an object detector
        self.classifier = False
        # Pipe files inside a zip; the caller narrows this to one before
        # building when several are present
        self.pipes = pipes or []
        # The archive a bundled model was extracted from, if any
        self.source = ''

    @property
    def pipe_in_zip(self):
        return self.pipes[0] if len(self.pipes) == 1 else ''

    def describe(self):
        base = os.path.basename(self.source or self.path)
        if self.source:
            base += ' (' + os.path.basename(self.path) + ')'
        if not self.runnable:
            return base + ': ' + self.kind + ', not runnable: ' + self.reason
        if self.kind == 'pipeline_zip':
            if len(self.pipes) == 1:
                return base + ': packaged pipeline ' + self.pipes[0]
            return base + ': zip holding ' + str(len(self.pipes)) + \
                ' pipelines: ' + ', '.join(self.pipes)
        return base + ': ' + self.kind + ', runs with the ' + self.impl + \
            (' full-frame classifier' if self.classifier else ' detector')


# -----------------------------------------------------------------------------
def is_model_file(path):
    return path.lower().endswith(MODEL_EXTS) and os.path.isfile(path)


# -----------------------------------------------------------------------------
def _pickle_strings(path):
    """Every string constant and global reference in a torch checkpoint.

    Tensor data lives outside the pickle, so this stays cheap on multi-GB
    files and never imports the framework that wrote them.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith('data.pkl')]
            if not members:
                return []
            data = zf.read(members[0])
    else:
        with open(path, 'rb') as f:
            data = f.read(_MAX_PICKLE_BYTES)

    strings = []
    try:
        for _op, arg, _pos in pickletools.genops(io.BytesIO(data)):
            if isinstance(arg, str):
                strings.append(arg)
    except Exception:
        pass
    return strings


def _companion(files, exts, stem=None, basename=None):
    """A file among `files` with one of `exts`: the one sharing `stem`, or
    the only candidate. `basename` asks for an exact file name instead."""
    if basename is not None:
        for f in files:
            if os.path.basename(f) == basename:
                return f
        return None
    candidates = [f for f in files if f.lower().endswith(exts)]
    if stem is not None:
        for f in candidates:
            if os.path.splitext(os.path.basename(f))[0] == stem:
                return f
    if len(candidates) == 1:
        return candidates[0]
    return None


def _files_beside(path):
    folder = os.path.dirname(os.path.abspath(path))
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))]


def _identify_darknet(path, files):
    stem = os.path.splitext(os.path.basename(path))[0]
    config = _companion(files, ('.cfg',), stem)
    names = _companion(files, _CLASS_LIST_EXTS, stem)
    if not (config and names):
        return ModelInfo(path, 'Darknet weights', runnable=False,
                         reason='needs a .cfg network definition and a '
                                '.lbl class list beside it')
    return ModelInfo(path, 'Darknet weights', 'darknet',
                     {'net_config': config, 'weight_file': path,
                      'class_names': names}, windowed=False)


def _identify_torch(path, files):
    strings = _pickle_strings(path)
    if not strings:
        return ModelInfo(path, 'unrecognized checkpoint', runnable=False,
                         reason='not a readable PyTorch checkpoint')

    seen = set(strings)

    def any_prefix(prefix):
        return any(s.startswith(prefix) for s in strings)

    def any_contains(sub):
        return any(sub in s for s in strings)

    if any_prefix('ultralytics'):
        return ModelInfo(path, 'Ultralytics YOLO checkpoint', 'ultralytics',
                         {'weight': path}, mode='original_and_resized')

    if any_prefix('lightning_hydra_detection'):
        return ModelInfo(path, 'LitDet checkpoint', 'litdet',
                         {'checkpoint': path})

    if any_contains('transformer.decoder') and any_contains('class_embed'):
        return ModelInfo(path, 'RF-DETR checkpoint', 'rf_detr',
                         {'weight': path})

    if any_prefix('yolo.'):
        train_config = _companion(files, (), basename='train_config.yaml')
        if train_config is None:
            return ModelInfo(path, 'MIT YOLO checkpoint', runnable=False,
                             reason='needs train_config.yaml beside it')
        return ModelInfo(path, 'MIT YOLO checkpoint', 'mit_yolo',
                         {'weight': path,
                          'model': _mit_yolo_model_name(train_config)},
                         mode='original_and_resized')

    if 'model_state_dict' in seen:
        return ModelInfo(path, 'netharn training snapshot', runnable=False,
                         reason='needs the deployed .zip written at the end '
                                'of training, which also carries the model '
                                'topology')

    stem = os.path.splitext(os.path.basename(path))[0]

    if 'meta' in seen and 'state_dict' in seen:
        config = _companion(files, ('.py',), stem)
        labels = _companion(files, _CLASS_LIST_EXTS, stem)
        if not (config and labels):
            return ModelInfo(path, 'MMDetection checkpoint', runnable=False,
                             reason='needs a .py network config and .lbl '
                                    'class list beside it')
        return ModelInfo(path, 'MMDetection checkpoint', 'mmdet',
                         {'net_config': config, 'weight_file': path,
                          'class_names': labels})

    if 'model' in seen and 'iteration' in seen:
        config = _companion(files, ('.yaml', '.yml'), stem)
        if config is None:
            return ModelInfo(path, 'Detectron2 checkpoint', runnable=False,
                             reason='needs its .yaml config beside it, '
                                    'which the checkpoint does not carry')
        return ModelInfo(path, 'Detectron2 checkpoint', 'detectron2',
                         {'checkpoint_fpath': path, 'cfg': config})

    return ModelInfo(path, 'unrecognized checkpoint', runnable=False,
                     reason='no detector implementation recognizes its '
                            'contents')


def _mit_yolo_model_name(train_config):
    try:
        import yaml
        with open(train_config) as f:
            doc = yaml.safe_load(f) or {}
        model = doc.get('model')
        if isinstance(model, dict):
            return str(model.get('name', ''))
        if isinstance(model, str):
            return model
    except Exception:
        pass
    return ''


def _identify_weights(path, files):
    if path.lower().endswith(_DARKNET_EXTS):
        return _identify_darknet(path, files)
    return _identify_torch(path, files)


# -----------------------------------------------------------------------------
def _onnx_modelspec(path):
    """The modelspec beside a bare .onnx file, or inside a package zip."""
    try:
        if path.lower().endswith('.zip'):
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith('.modelspec.json'):
                        return json.loads(zf.read(name).decode('utf-8'))
            return {}
        sidecar = os.path.splitext(path)[0] + '.modelspec.json'
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _identify_onnx(path):
    spec = _onnx_modelspec(path)
    post = spec.get('postprocess', {}) if isinstance(spec, dict) else {}
    meta = spec.get('meta', {}) if isinstance(spec, dict) else {}
    if post.get('decoder') == 'classifier' or meta.get('task') == 'classification':
        info = ModelInfo(path, 'ONNX classifier package', 'onnx_classifier',
                         {'model': path})
        info.classifier = True
        return info
    return ModelInfo(path, 'ONNX detector package', 'onnx', {'model': path})


def _netharn_is_classifier(zip_path, names):
    """Deployed classifiers wrap ClfModel; detectors wrap a detection
    network. train_info.json records the class, the topology file is named
    after it."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in names:
                if name.endswith('train_info.json'):
                    info = json.loads(zf.read(name).decode('utf-8'))
                    model = info.get('hyper', {}).get('model')
                    if isinstance(model, (list, tuple)) and model:
                        return 'clf' in str(model[0]).lower()
    except Exception:
        pass
    return any(os.path.basename(n).lower().startswith('clfmodel')
               for n in names if n.endswith('.py'))


def _identify_netharn_zip(path, names):
    if _netharn_is_classifier(path, names):
        info = ModelInfo(path, 'netharn deployed classifier',
                         'netharn_classifier', {'deployed': path})
        info.classifier = True
        return info
    return ModelInfo(path, 'netharn deployed model', 'netharn',
                     {'deployed': path}, mode='original_and_resized')


# -----------------------------------------------------------------------------
def _identify_bundle(path, work_dir):
    """An arbitrary zip of weights and their companions: extract it and
    classify what is inside."""
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='viame_model_')
    extract_dir = os.path.join(
        work_dir, os.path.splitext(os.path.basename(path))[0])
    with zipfile.ZipFile(path) as zf:
        zf.extractall(extract_dir)

    files = []
    for root, _dirs, names in os.walk(extract_dir):
        files.extend(os.path.join(root, n) for n in sorted(names))

    weights = [f for f in files if f.lower().endswith(_TORCH_EXTS + _DARKNET_EXTS)]
    if not weights:
        return ModelInfo(path, 'unrecognized archive', runnable=False,
                         reason='holds no pipeline, ONNX package, deployed '
                                'model or weights')

    first = None
    for weight in weights:
        info = _identify_weights(weight, files)
        info.kind = 'zip of ' + info.kind
        info.source = os.path.abspath(path)
        if info.runnable:
            return info
        first = first or info
    return first


# -----------------------------------------------------------------------------
def _identify_zip(path, work_dir):
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith('/')]
    except zipfile.BadZipFile:
        return ModelInfo(path, 'unreadable archive', runnable=False,
                         reason='not a valid zip file')

    pipes = sorted((n.replace('\\', '/') for n in names if n.endswith('.pipe')),
                   key=lambda n: (n.count('/'), n))
    if pipes:
        return ModelInfo(path, 'pipeline_zip', pipes=pipes)

    exts = {os.path.splitext(n)[1].lower() for n in names}

    if '.onnx' in exts:
        return _identify_onnx(path)

    if '.pt' in exts and '.py' in exts and '.json' in exts:
        return _identify_netharn_zip(path, names)

    return _identify_bundle(path, work_dir)


# -----------------------------------------------------------------------------
def identify(path, work_dir=None):
    """Classify a model file, returning a ModelInfo.

    A zip bundling weights with their companion files is extracted into
    work_dir (a fresh temporary folder when not given) so its contents can be
    inspected; the returned keys then point into that folder.
    """
    if path.lower().endswith('.zip'):
        return _identify_zip(path, work_dir)
    if path.lower().endswith('.onnx'):
        return _identify_onnx(path)
    if path.lower().endswith(_TORCH_EXTS + _DARKNET_EXTS):
        return _identify_weights(path, _files_beside(path))
    return ModelInfo(path, 'unrecognized file', runnable=False,
                     reason='unsupported extension')


# -----------------------------------------------------------------------------
def _detector_block(info, indent):
    """The [-DETECTOR-IMPL-] replacement, nested in the windowed detector
    the way trained pipelines are."""
    lines = []

    def entry(level, key, value):
        prefix = indent + '  ' * level + ':' + key
        lines.append(prefix.ljust(44) + ' ' + str(value))

    if not info.windowed:
        entry(0, 'detector:type', info.impl)
        lines.append('')
        lines.append(indent + 'block detector:' + info.impl)
        for key, value in info.keys.items():
            entry(1, key, value)
        lines.append(indent + 'endblock')
        return '\n'.join(lines)

    entry(0, 'detector:type', 'ocv_windowed')
    lines.append('')
    lines.append(indent + 'block detector:ocv_windowed')
    entry(1, 'detector:type', info.impl)
    lines.append('')
    entry(1, 'mode', info.mode)
    if info.mode == 'original_and_resized':
        entry(1, 'chip_width', 640)
        entry(1, 'chip_height', 640)
        entry(1, 'chip_adaptive_thresh', 1600000)
    lines.append('')
    lines.append(indent + '  block detector:' + info.impl)
    for key, value in info.keys.items():
        if value != '':
            entry(2, key, value)
    lines.append(indent + '  endblock')
    lines.append(indent + 'endblock')

    return '\n'.join(lines)


def _marker_indent(text, marker):
    """Leading whitespace of a marker that sits alone on its line, else ''."""
    pos = text.find(marker)
    if pos < 0:
        return ''
    start = text.rfind('\n', 0, pos) + 1
    prefix = text[start:pos]
    return prefix if prefix.strip() == '' else ''


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)
    return path


def _render_template(template, replacements, out_path):
    with open(template) as f:
        text = f.read()
    for marker, value in replacements.items():
        indent = _marker_indent(text, marker)
        text = text.replace(indent + marker, value)
    return _write(out_path, text)


def is_embedded_pipeline(text):
    """Pipelines written by training carry no reader; a host supplies one."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('include') and 'input' in stripped:
            return False
        if stripped.startswith('::') and stripped[2:].strip() in (
                'video_input', 'image_list_reader', 'frame_list_input'):
            return False
    return True


def _wrap_packaged_pipeline(info, work_dir, template):
    """Extract the zip and run its chosen pipe. One packaged by training has
    no input or output of its own and is spliced between the default
    template's reader and writer; a complete pipeline runs as it is."""
    if len(info.pipes) != 1:
        raise ValueError('choose one of the pipelines in ' + info.path)

    extract_dir = os.path.join(
        work_dir, os.path.splitext(os.path.basename(info.path))[0])
    with zipfile.ZipFile(info.path) as zf:
        zf.extractall(extract_dir)
    inner = os.path.join(extract_dir, *info.pipes[0].split('/'))

    with open(inner) as f:
        if not is_embedded_pipeline(f.read()):
            return inner

    with open(template) as f:
        text = f.read()
    head_end = text.find('process detector_input')
    tail_start = text.find('process detector_writer')
    if head_end < 0 or tail_start < 0:
        raise RuntimeError('Unexpected layout in ' + template)

    middle = (
        'include ' + inner + '\n\n'
        'connect from downsampler.output_1\n'
        '        to   detector_input.image\n\n'
        '# ' + '=' * 77 + '\n\n'
    )
    return _write(os.path.join(work_dir, 'detector.pipe'),
                  text[:head_end] + middle + text[tail_start:])


def _render_classifier(template, info, out_path):
    """The frame classifier template around another whole-frame classifier:
    swap its netharn_classifier block for one holding `info`."""
    with open(template) as f:
        text = f.read()

    text = re.sub(r'(?m)^([ \t]*:detector:type)[ \t]+netharn_classifier[ \t]*$',
                  lambda m: m.group(1).ljust(44) + ' ' + info.impl, text)

    block = re.compile(r'(?P<indent>[ \t]*)block detector:netharn_classifier\n'
                       r'.*?\n(?P=indent)endblock', re.S)
    match = block.search(text)
    if match is None:
        raise RuntimeError('Unexpected layout in ' + template)
    indent = match.group('indent')
    lines = [indent + 'block detector:' + info.impl]
    for key, value in info.keys.items():
        lines.append((indent + '  :' + key).ljust(44) + ' ' + str(value))
    lines.append(indent + 'endblock')
    text = text[:match.start()] + '\n'.join(lines) + text[match.end():]
    return _write(out_path, text)


def build_pipeline(info, work_dir, pipelines_dir):
    """Write a runnable pipeline for `info` into work_dir and return its path.

    pipelines_dir is the install's configs/pipelines folder, which holds the
    templates.
    """
    if not info.runnable:
        raise ValueError(info.describe())

    templates = os.path.join(pipelines_dir, 'templates')

    if info.kind == 'pipeline_zip':
        return _wrap_packaged_pipeline(
            info, work_dir, os.path.join(templates, 'detector_default.pipe'))

    if info.classifier:
        return _render_classifier(
            os.path.join(templates, 'detector_netharn_clfr.pipe'), info,
            os.path.join(work_dir, 'classifier.pipe'))

    if info.impl == 'onnx':
        return _render_template(
            os.path.join(templates, 'detector_onnx.pipe'),
            {'[-MODEL-]': info.path},
            os.path.join(work_dir, 'detector.pipe'))

    template = os.path.join(templates, 'detector_default.pipe')
    with open(template) as f:
        indent = _marker_indent(f.read(), '[-DETECTOR-IMPL-]')
    return _render_template(
        template,
        {'[-DETECTOR-IMPL-]': _detector_block(info, indent)},
        os.path.join(work_dir, 'detector.pipe'))
