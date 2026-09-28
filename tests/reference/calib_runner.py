"""Read a calibration source and an OpenCV FileStorage document.

`load_stereo_calibration` is VIAME's own reader, and the recorder and the
golden test go through the same call for it.

The document readers are the one place in this framework where they
deliberately do not. `dump_document` here is `library/file_io/opencv_yaml`,
VIAME's own parser, and the replay calls it; the recording it is held to was
taken through the format's own parser, which lives in
`tests/reference/opencv/recorders/file_storage_reference.py` because that is the
only part of the tree allowed to reach for the reference implementation.
"""

import numpy as np


def load_calibration(path):
    """`viame::read_stereo_rig` on `path`, as {key: float64 array}."""
    from viame.measurement import _measurement

    loaded = _measurement.load_stereo_calibration(str(path))

    return {key: np.asarray(value, dtype=np.float64)
            for key, value in loaded.items()}




def dump_document(path):
    """Every top level node, through VIAME's own reader.

    `viame.file_io._opencv_yaml` is `library/file_io/opencv_yaml` -- the
    same C++ the calibration algorithms read and write with.
    """
    from viame.file_io import _opencv_yaml

    return _opencv_yaml.read(str(path))
