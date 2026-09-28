# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Read a calibration document through the reference implementation.

The document readers are the one place in the golden framework where the
recorder and the replay deliberately do not go through the same call. This
is `cv::FileStorage` itself, which is the definition of the format and what
the recording was taken through; `calib_runner.dump_document` is
`library/file_io/opencv_yaml`, which replaced it in P7-T05. The recorder
calls this and the replay calls ours, so the recording says "this is what
the format's own parser produced" and the test says "and this reader
agrees".

Everywhere else in the framework one registered name changes implementation
underneath and the two calls stay identical; a file format reader has no
registry to change, so the asymmetry is written out instead -- and it is why
this one helper lives here rather than beside the runner that uses it.
"""


def _node(node):
    """One node, as JSON-able values.

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

    raise ValueError("unhandled node type {}".format(node.type()))


def dump_document_reference(path):
    """Every top level node, through the reference parser."""
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)

    if not storage.isOpened():
        raise IOError("FileStorage could not open {}".format(path))

    try:
        root = storage.root()
        return {key: _node(root.getNode(key)) for key in root.keys()}
    finally:
        storage.release()
