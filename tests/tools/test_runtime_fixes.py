"""Regression coverage for calibration export and wheel configuration closure."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_export_preserves_existing_file_on_failure(tmp_path):
    # Load the writer without requiring calibration GUI/model dependencies.
    tree = ast.parse((ROOT / 'tools/calibrate.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'write_calibration_json')
    namespace = dict(json=json, os=os, tempfile=tempfile)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'calibrate.py', 'exec'), namespace)
    write = namespace['write_calibration_json']
    output = tmp_path / 'calibration.json'
    output.write_text('{"previous": true}')
    with pytest.raises(TypeError):
        write(output, {'width': np.int64(640)})
    assert json.loads(output.read_text()) == {'previous': True}
    write(output, {'width': 640})
    assert json.loads(output.read_text()) == {'width': 640}
    assert list(tmp_path.iterdir()) == [output]


def test_default_training_templates_and_missing_dependencies(tmp_path):
    selector = load_module(ROOT / 'cmake/wheel/select_default_configs.py')
    configs = tmp_path / 'configs/pipelines'
    configs.mkdir(parents=True)
    (configs / 'templates').mkdir()
    (configs / 'train_detector_default.conf').write_text(
        'include common_train.conf\n'
        'relativepath pipeline_template = templates/embedded.pipe\n'
        'relativepath seed_model = models/optional.pth\n')
    (configs / 'common_train.conf').write_text('trainer:type = adaptive\n')
    (configs / 'templates/embedded.pipe').write_text('include common_tracker.pipe\n')
    (configs / 'common_tracker.pipe').write_text('process tracker\n  :: track_objects\n')
    (configs / 'unfinished.pipe').write_text('TODO: Make me\n')
    (configs / 'broken.pipe').write_text('include missing.pipe\n')
    output = tmp_path / 'contents.txt'
    assert selector.main(['--prefix', str(tmp_path), '--output', str(output)]) == 0
    text = output.read_text()
    assert 'train_detector_default.conf' in text
    assert '-> {data}/configs/pipelines/templates/embedded.pipe' in text
    assert 'common_tracker.pipe' in text
    assert 'unfinished.pipe' not in text
    assert 'broken.pipe' not in text
    assert 'optional.pth' not in text


def test_sparse_dependencies_do_not_import_open3d(monkeypatch):
    # The dependency importer is independent of the reconstruction routines.
    location = ROOT / 'library/measurement/colmap/reconstruction.py'
    if not location.exists():
        location = ROOT / 'plugins/colmap/reconstruction.py'
    module = load_module(location)
    import builtins
    import types
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == 'open3d':
            raise AssertionError('sparse reconstruction imported Open3D')
        if name == 'pycolmap':
            return types.SimpleNamespace()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    module.import_dependencies(dense=False)


def test_query_outputs_and_shutdown(tmp_path):
    import subprocess
    import sys
    pipeline = tmp_path / 'query.pipe'
    pipeline.write_text('''process input
 :: input_adapter
process query
 :: perform_query
 :external_handler false
process output
 :: output_adapter
connect from input.query
 to query.database_query
connect from query.query_result
 to output.results
connect from query.feedback_request
 to output.feedback
connect from query.iqr_model
 to output.model
''')
    code = '''
import sys
from viame.modules import load_known_modules
from viame.adapters.embedded_pipeline import EmbeddedPipeline
from viame.adapters.adapter_data_set import AdapterDataSet
from viame.types import QueryResult
result = QueryResult()
assert result.preference_score == 0.0
result.preference_score = 0.25
assert result.relevancy_score == 0.0
load_known_modules()
ep = EmbeddedPipeline()
ep.build_pipeline(sys.argv[1])
ep.start()
ds = AdapterDataSet.create()
ds.add_nullptr('query', 'database_query')
ep.send(ds)
out = ep.receive()
assert out['results'] == []
assert out['feedback'] == []
assert out['model'] is None
ep.send_end_of_input()
assert ep.receive().is_end_of_data()
ep.wait()
'''
    result = subprocess.run([sys.executable, '-c', code, str(pipeline)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_depth_writes_rgb_vertex_colors(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    depth = load_module(ROOT / 'tools/depth.py')
    pixels = np.array([[[135, 35, 11], [135, 35, 11]]], dtype=np.uint8)
    monkeypatch.setattr(depth.imageops, 'read_image', lambda _: pixels)
    monkeypatch.setattr(depth.opencv_yaml, 'read', lambda *args: {'Q': np.eye(4)})
    monkeypatch.setattr(depth, 'scaled_disparity', lambda *args: np.ones((1, 1)))
    monkeypatch.setattr(depth.projection, 'reproject_to_3d',
                        lambda *args: np.ones((1, 1, 3)))
    colors = {}
    monkeypatch.setattr(depth, 'write_ply_file',
                        lambda points, path, color, keys: colors.update(color))
    monkeypatch.setattr(sys, 'argv', ['depth', 'stereo.png', 'extrinsics.yml'])
    assert depth.main() == 0
    assert [colors[c][0] for c in ['red', 'green', 'blue']] == [135, 35, 11]
