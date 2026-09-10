import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_rectify", Path(__file__).resolve().parents[2] / "tools" / "rectify.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import numpy as np


def test_output_failure_and_input_depth(tmp_path):
    intr = tmp_path / 'intr.yml'
    extr = tmp_path / 'extr.yml'
    fs = m.cv2.FileStorage(str(intr), m.cv2.FILE_STORAGE_WRITE)
    for name in ['M1', 'M2']:
        fs.write(name, np.eye(3))
    for name in ['D1', 'D2']:
        fs.write(name, np.zeros((1, 5)))
    fs.release()
    fs = m.cv2.FileStorage(str(extr), m.cv2.FILE_STORAGE_WRITE)
    for name in ['R1', 'R2']:
        fs.write(name, np.eye(3))
    for name in ['P1', 'P2']:
        fs.write(name, np.hstack([np.eye(3), np.zeros((3, 1))]))
    fs.release()
    with patch.object(m.sys, 'argv', ['rectify', 'input.png', 'output.png', str(intr), str(extr)]), patch.object(m.cv2, 'imread', return_value=np.zeros((4, 8), dtype=np.uint16)), patch.object(m.cv2, 'imwrite', return_value=False) as write:
        with pytest.raises(ValueError, match='Failed to write'):
            m.main()
    assert write.call_args.args[1].dtype == np.uint16
