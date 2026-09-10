"""Read a calibration source and an OpenCV FileStorage document.

Shared by the recorder and the golden test, so both go through exactly the
same call. `load_stereo_calibration` is VIAME's own reader; `dump_document`
is `cv::FileStorage` itself, which is what the in-house YAML reader has to
agree with.
"""

import numpy as np


def load_calibration(path):
    """`viame::read_stereo_rig` on `path`, as {key: float64 array}."""
    from viame.core import _measurement

    loaded = _measurement.load_stereo_calibration(str(path))

    return {key: np.asarray(value, dtype=np.float64)
            for key, value in loaded.items()}


def _node(node):
    """One FileStorage node, as JSON-able values.

    Kept structural rather than flattened: a map stays a map and a sequence
    stays a sequence, because the shape is half of what a reader has to get
    right. An `!!opencv-matrix` is a map of rows, cols, dt and data, and is
    recorded as one rather than as a decoded array, so that a reader is held
    to the four fields the format actually has.
    """
    if node.isNone():
        return None
    if node.isInt():
        return int(node.real())
    if node.isReal():
        return float(node.real())
    if node.isString():
        return node.string()
    if node.isMap():
        return {key: _node(node.getNode(key)) for key in node.keys()}
    if node.isSeq():
        return [_node(node.at(index)) for index in range(node.size())]

    # A matrix reads back as an array; everything else that is left is a type
    # no VIAME document uses, and is worth failing on rather than guessing.
    matrix = node.mat()
    if matrix is not None:
        return {"__matrix__": {"rows": int(matrix.shape[0]),
                               "cols": int(matrix.shape[1]),
                               "dtype": str(matrix.dtype),
                               "data": matrix.reshape(-1).tolist()}}

    raise ValueError("unhandled FileStorage node type {}".format(node.type()))


def dump_document(path):
    """Every top level node of an OpenCV YAML or XML file."""
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)

    if not storage.isOpened():
        raise IOError("FileStorage could not open {}".format(path))

    try:
        root = storage.root()
        return {key: _node(root.getNode(key)) for key in root.keys()}
    finally:
        storage.release()
