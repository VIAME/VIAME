import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_segment", Path(__file__).resolve().parents[2] / "tools" / "segment.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_validation_preserves_all_non_polygon_fields(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('7,img.png,0,0,0,10,10,1,-1,fish,1\n')
    b.write_text('7,other.png,0,0,0,10,10,0.1,-1,shark,1\n')
    assert not m.validate_unit(a, b)['ok']


def test_manual_polygon_cannot_be_replaced(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    row = '7,img.png,0,0,0,10,10,1,-1,fish,1,'
    a.write_text(row + '(poly) 0 0 10 0 0 10\n')
    b.write_text(row + '(poly) 0 0 5 0 0 5\n')
    assert not m.validate_unit(a, b)['ok']


def test_atomic_write_failure_keeps_original(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_text('original')
    def broken():
        yield 'partial'
        raise OSError('disk failure')
    with pytest.raises(OSError):
        m.write_lines_atomic(p, broken())
    assert p.read_text() == 'original'
