"""`tools/rectify.py`, over the two things it is easy to get wrong.

That it reports a failed write rather than exiting zero, and that a 16-bit
input comes out 16-bit -- a rectification that quietly narrowed to 8 bits
would still look right.

Written against cv2 and now against the replacements: the calibration is
written with `opencv_yaml` rather than `cv::FileStorage`, the input is
`imageops.read_unchanged` rather than `cv2.imread`, and the two are patched
on the module because that is how the tool reaches them.
"""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from viame.utilities import opencv_yaml

spec = importlib.util.spec_from_file_location(
    "review_rectify",
    Path(__file__).resolve().parents[2] / "tools" / "rectify.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_output_failure_and_input_depth(tmp_path):
    intr = tmp_path / 'intr.yml'
    extr = tmp_path / 'extr.yml'

    opencv_yaml.write(str(intr), {
        'M1': np.eye(3), 'M2': np.eye(3),
        'D1': np.zeros((1, 5)), 'D2': np.zeros((1, 5)),
    })
    opencv_yaml.write(str(extr), {
        'R1': np.eye(3), 'R2': np.eye(3),
        'P1': np.hstack([np.eye(3), np.zeros((3, 1))]),
        'P2': np.hstack([np.eye(3), np.zeros((3, 1))]),
    })

    def refuse(path, array):
        raise OSError("no")

    with patch.object(m.sys, 'argv',
                      ['rectify', 'input.png', 'output.png',
                       str(intr), str(extr)]), \
         patch.object(m.imageops, 'read_unchanged',
                      return_value=np.zeros((4, 8), dtype=np.uint16)), \
         patch.object(m.imageops, 'write_image',
                      side_effect=refuse) as write:
        with pytest.raises(ValueError, match='Failed to write'):
            m.main()

    assert write.call_args.args[1].dtype == np.uint16


def test_an_unreadable_input_is_reported(tmp_path):
    """The read used to return None and now raises, and either is an error."""
    intr = tmp_path / 'intr.yml'
    extr = tmp_path / 'extr.yml'

    opencv_yaml.write(str(intr), {
        'M1': np.eye(3), 'M2': np.eye(3),
        'D1': np.zeros((1, 5)), 'D2': np.zeros((1, 5)),
    })
    opencv_yaml.write(str(extr), {
        'R1': np.eye(3), 'R2': np.eye(3),
        'P1': np.hstack([np.eye(3), np.zeros((3, 1))]),
        'P2': np.hstack([np.eye(3), np.zeros((3, 1))]),
    })

    with patch.object(m.sys, 'argv',
                      ['rectify', 'missing.png', 'output.png',
                       str(intr), str(extr)]), \
         patch.object(m.imageops, 'read_unchanged',
                      side_effect=OSError("no such file")):
        with pytest.raises(ValueError, match='Failed to read'):
            m.main()
