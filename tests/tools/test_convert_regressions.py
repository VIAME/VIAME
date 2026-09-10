import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_convert", Path(__file__).resolve().parents[2] / "tools" / "convert.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_camcal_lone_positional_is_output():
    with patch.object(m.sys, 'argv', ['convert', '--left-cal', 'left.CamCAL', '-o', 'json', 'output.json']), patch.object(m, '_import_calibration_deps'), patch.object(m, 'convert') as convert:
        assert m.main() == 0
    assert convert.call_args.kwargs['input_path'] is None
    assert convert.call_args.kwargs['output_path'] == 'output.json'

from .tool_regression_helpers import tool_env, run_viame


def test_negative_frame_offset_is_annotation_option(tool_env, tmp_path):
    p, out = tmp_path / 'in.csv', tmp_path / 'out.csv'
    p.write_text('7,a.png,1,0,0,10,10,1,-1,fish,1\n')
    r = run_viame(tool_env, 'convert', str(p), str(out), '--frame-offset', '-1', '--no-images')
    assert r.returncode == 0, r.stderr
    rows = [line.split(',') for line in out.read_text().splitlines() if line and not line.startswith('#')]
    assert rows[0][2].strip() == '0'
