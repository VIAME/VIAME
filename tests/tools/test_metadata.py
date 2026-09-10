import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_metadata", Path(__file__).resolve().parents[2] / "tools" / "metadata.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_failed_gps_match_does_not_borrow_pose():
    logs = [{'lat': 50., 'lon': 50., 'yaw': 90}]
    with patch.object(m, 'load_exif_meta', return_value={'lat': 0., 'lon': 0.}):
        records, stats = m.link_imagelog('.', ['a.jpg'], logs)
    assert records['a.jpg']['lat'] == 0.
    assert 'yaw' not in records['a.jpg']
    assert records['a.jpg']['match_method'] == 'gps-unmatched'
    assert stats['by_order'] == 0
