# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Reading the YAML that `cv::FileStorage` writes.

Calibration files in the wild are OpenCV's, and they will go on being
OpenCV's whatever VIAME uses to read them -- a rig calibrated five years ago
is not going to be recalibrated because the reader changed. So this is not
a format to be migrated away from; it is one to be able to read without
linking OpenCV.

Two things make it not plain YAML:

* the header line `%YAML:1.0`, with no space after the colon, which PyYAML
  rejects as an unknown directive;
* matrices carried as a mapping tagged `!!opencv-matrix`, with `rows`,
  `cols`, `dt` and a flat `data` list.

Both are handled here and nothing else is: this reads what OpenCV writes,
not the whole of YAML's tag zoo. `cv::FileStorage`'s XML and JSON dialects
are not supported, because nothing in the tree reads one.

`write` emits the same dialect, so a calibration this tree produces still
opens in anything that expects OpenCV's format -- which matters, because
these files outlive the tool that wrote them.
"""

import re

import numpy as np


_HEADER = re.compile(r"^%YAML:(\d+\.\d+)\s*$", re.MULTILINE)


def _yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - a missing dependency
        raise ImportError(
            "reading an OpenCV YAML calibration needs PyYAML") from exc
    return yaml


def _matrix(loader, node):
    """`!!opencv-matrix`: rows, cols, dt and a flat data list."""
    mapping = loader.construct_mapping(node, deep=True)

    rows = int(mapping["rows"])
    cols = int(mapping["cols"])
    data = mapping.get("data") or []

    # `dt` is OpenCV's element type letter: u, c, w, s, i, f, d. Only the
    # width matters here, and every calibration value is read as double.
    values = np.asarray(data, dtype=np.float64)

    if values.size != rows * cols:
        raise ValueError(
            "opencv-matrix says {}x{} but carries {} values".format(
                rows, cols, values.size))

    return values.reshape(rows, cols)


def loads(text):
    """Every top level entry of an OpenCV YAML document, as a dict."""
    yaml = _yaml()

    class Loader(yaml.SafeLoader):
        pass

    # OpenCV writes the tag as `!!opencv-matrix`, which resolves against the
    # default `tag:yaml.org,2002:` prefix.
    Loader.add_constructor("tag:yaml.org,2002:opencv-matrix", _matrix)
    Loader.add_constructor("!opencv-matrix", _matrix)

    # `%YAML:1.0` is not a directive PyYAML will take; it is also the only
    # thing the header line says, so dropping it loses nothing.
    text = _HEADER.sub("", text, count=1)

    loaded = yaml.load(text, Loader=Loader)
    return loaded if loaded is not None else {}


def load(path):
    """`loads` on a file, which is what `cv2.FileStorage(path, READ)` did."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return loads(handle.read())


def read(path, names):
    """The named entries of an OpenCV YAML file, `None` where absent.

    The shape `cv::FileStorage` callers want: ask for the keys you need and
    get `None` back for the ones the file does not have, rather than a
    `KeyError` per missing optional.
    """
    document = load(path)
    return {name: document.get(name) for name in names}


def dumps(values):
    """An OpenCV YAML document for a mapping of name to value.

    Matrices go out as `!!opencv-matrix` with `rows`, `cols`, `dt` and a
    flat `data` list, which is what `cv::FileStorage` writes and what
    `loads` above reads back. Everything else is emitted as a plain scalar.

    The `%YAML:1.0` header has no space after the colon. That is not a typo:
    it is what OpenCV writes and what its own parser expects, and a space
    there makes `cv::FileStorage` reject the file.
    """
    lines = ["%YAML:1.0", "---"]

    for name, value in values.items():
        array = np.asarray(value)

        if array.ndim == 0 and array.dtype.kind in "USO":
            lines.append("{}: {}".format(name, value))
            continue

        if array.ndim == 0:
            lines.append("{}: {:.16g}".format(name, float(array)))
            continue

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                "{}: only scalars and two dimensional matrices are "
                "written".format(name))

        rows, cols = array.shape
        # `d` for double, `i` for integer, as OpenCV spells its element types
        letter = "i" if array.dtype.kind in "iu" else "d"
        flat = array.reshape(-1)

        if letter == "i":
            entries = ", ".join(str(int(v)) for v in flat)
        else:
            entries = ", ".join("{:.16g}.".format(float(v))
                                if float(v) == int(float(v))
                                else "{:.16g}".format(float(v)) for v in flat)

        lines.append("{}: !!opencv-matrix".format(name))
        lines.append("   rows: {}".format(rows))
        lines.append("   cols: {}".format(cols))
        lines.append("   dt: {}".format(letter))
        lines.append("   data: [ {} ]".format(entries))

    return "\n".join(lines) + "\n"


def write(path, values):
    """`dumps` to a file, which is what `cv2.FileStorage(path, WRITE)` did."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps(values))
