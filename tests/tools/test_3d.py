import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip('viame.opencv.registration_utils')

spec = importlib.util.spec_from_file_location("review_3d", Path(__file__).resolve().parents[2] / "tools" / "3d.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import sys


def test_all_has_distinct_outputs_and_reports_failure(tmp_path):
    for name in ('a', 'b'):
        (tmp_path / name).mkdir()
    outputs = []
    def process(folder, output, **kwargs):
        outputs.append(Path(output))
        if Path(folder).name == 'a':
            raise RuntimeError('reconstruction failed')
        return True
    with patch.object(sys, 'argv', ['3d', '--all', '--base-dir', str(tmp_path), '--output', str(tmp_path / 'out')]), patch.object(m, 'import_dependencies'), patch.object(m, 'get_image_files', return_value=['one', 'two']), patch.object(m, 'process_folder', side_effect=process):
        assert m.main() == 1
    assert outputs == [tmp_path / 'out/a', tmp_path / 'out/b']
