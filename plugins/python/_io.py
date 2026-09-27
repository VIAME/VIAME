"""Convenient loading through VIAME's registered image/video/annotation readers."""
import builtins
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

from ._file_formats import (file_kind, json_annotation_format, is_viame_csv_row,
                            image_list_entries)


@lru_cache(maxsize=1)
def _algorithms():
    from kwiver.vital import algo, types
    from kwiver.vital.modules import load_known_modules
    load_known_modules()
    return algo, types


def _create(interface, names):
    errors = []
    for name in names:
        try:
            reader = interface.create(name)
            if reader is not None:
                return reader
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("No {} reader available (tried {}): {}".format(
        interface.__name__, ", ".join(names), "; ".join(errors)))


def _image_reader():
    algo, _ = _algorithms()
    return _create(algo.ImageIO, ("ocv", "vxl"))


def _load_image(reader, path):
    image = reader.load(str(path))
    if image is None:
        raise OSError("Could not decode image: {}".format(path))
    return image


class ImageSequence:
    """Single-pass, lazy iterator of ImageContainers, in image-list order.

    ``timestamp`` and ``filename`` describe the last yielded image. Frame
    numbers start at one. Use ``enumerate(sequence)`` for Python indices.
    """
    def __init__(self, paths):
        self.paths = tuple(paths)
        self._index = 0
        self._reader = _image_reader()
        _, self._types = _algorithms()
        self.timestamp = self._types.Timestamp()
        self.filename = None
        self.closed = False

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or self._index >= len(self.paths):
            self.close()
            raise StopIteration
        try:
            path = self.paths[self._index]
            image = _load_image(self._reader, path)
            self._index += 1
            self.filename = str(path)
            self.timestamp = self._types.Timestamp()
            self.timestamp.set_frame(self._index)
            return image
        except Exception:
            self.close()
            raise

    def close(self):
        self.closed = True
        self._reader = None

    def __enter__(self):
        if self.closed:
            raise ValueError("Image sequence is closed")
        return self

    def __exit__(self, *args):
        self.close()


class VideoSequence:
    """Single-pass iterator of ImageContainers with optional timestamp sampling.

    Sampling keeps the first frame and the first frame at or after each
    requested time, without duplicating frames. ``timestamp`` retains the
    source frame number/time. Use a context manager when stopping early.
    """
    def __init__(self, path, frame_rate=None):
        algo, types = _algorithms()
        self.reader = _create(algo.VideoInput, ("ffmpeg", "vidl_ffmpeg"))
        self.filename = str(path)
        self.timestamp = types.Timestamp()
        self.closed = False
        self._rate = frame_rate
        self._start = None
        self._next_sample = 0
        self._index = 0
        try:
            self.reader.open(self.filename)
            self.source_frame_rate = self.reader.frame_rate()
        except Exception:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            while self.reader.next_frame():
                stamp = self.reader.frame_timestamp()
                index = self._index
                self._index += 1
                if self._rate is not None:
                    if stamp.has_valid_time():
                        seconds = stamp.get_time_seconds()
                    elif math.isfinite(self.source_frame_rate) and self.source_frame_rate > 0:
                        seconds = index / self.source_frame_rate
                    else:
                        raise ValueError("Video has neither timestamps nor a usable frame rate")
                    if self._start is None:
                        self._start = seconds
                    sample = (seconds - self._start) * self._rate
                    if sample + 1e-6 < self._next_sample:
                        continue
                    self._next_sample = math.floor(sample + 1e-6) + 1
                image = self.reader.frame_image()
                if image is None:
                    raise OSError("Reader returned no image: {}".format(self.filename))
                self.timestamp = stamp
                return image
        except Exception:
            self.close()
            raise
        self.close()
        raise StopIteration

    def close(self):
        if not self.closed:
            self.closed = True
            self.reader.close()

    def __enter__(self):
        if self.closed:
            raise ValueError("Video sequence is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()


def _annotations(path, format_name):
    algo, types = _algorithms()
    if format_name == "coco":
        from viame.core.read_object_track_set_coco import ReadObjectTrackSetCoco
        reader = ReadObjectTrackSetCoco()
    else:
        reader = _create(algo.ReadObjectTrackSet, (format_name,))
    cfg = reader.get_configuration()
    cfg.set_value("batch_load", "true")
    reader.set_configuration(cfg)
    try:
        reader.open(str(path))
        from ._io_native import read_tracks
        tracks = read_tracks(reader)
        return tracks if tracks is not None else types.ObjectTrackSet()
    finally:
        reader.close()


def _pipeline_root():
    """Find the same installed templates used by the run applet."""
    install = os.environ.get("VIAME_INSTALL")
    if install:
        return Path(install) / "configs" / "pipelines"
    executable = shutil.which("viame")
    candidates = ([Path(executable).resolve().parent.parent] if executable else [])
    candidates.extend(Path(__file__).resolve().parents)
    for root in candidates:
        if (root / "configs" / "pipelines").is_dir():
            return root / "configs" / "pipelines"
    raise RuntimeError("Cannot find VIAME pipelines; source setup_viame.sh or set VIAME_INSTALL")


class Pipeline:
    """Prepared pipeline/model using the same wrappers and runner as viame run.

    Opening prepares files without starting inference. ``run(input_path)``
    processes a file or directory; ``run()`` executes a self-contained pipe.
    ``path`` names the prepared pipe. Extracted/generated files remain valid
    until ``close()``; use a context manager to release them promptly.
    """
    def __init__(self, filename, pipeline=None):
        self.filename = str(Path(filename).resolve())
        self.path = self.filename
        self.info = None
        self.closed = False
        self._work = None
        try:
            if file_kind(self.filename) == "pipe":
                if pipeline is not None:
                    raise ValueError("pipeline selects a member of a ZIP, not a standalone pipe")
                return
            from viame.core import model_wrap
            self._work = tempfile.TemporaryDirectory(prefix="viame_open_")
            self.info = model_wrap.identify(self.filename, self._work.name)
            if not self.info.runnable:
                raise ValueError(self.info.describe())
            if pipeline is not None:
                if pipeline not in self.info.pipes:
                    raise ValueError("Unknown pipeline {!r}; choices: {}".format(
                        pipeline, ", ".join(self.info.pipes)))
                self.info.pipes = [pipeline]
            if len(self.info.pipes) > 1:
                raise ValueError("ZIP contains several pipelines; pass pipeline= with one of: "
                                 + ", ".join(self.info.pipes))
            self.path = model_wrap.build_pipeline(
                self.info, self._work.name, str(_pipeline_root()))
        except Exception:
            self.close()
            raise

    def run(self, input=None, *, output_dir=None, frame_rate=None, args=(), **kwargs):
        """Run synchronously and return subprocess.CompletedProcess.

        ``args`` is a sequence of additional ``viame run`` CLI arguments.
        Remaining keywords go to subprocess.run (e.g. timeout, capture_output).
        Nonzero exits raise CalledProcessError unless check=False is supplied.
        Input sampling and outputs follow the run command's defaults.
        """
        if self.closed:
            raise ValueError("Pipeline is closed")
        if isinstance(args, (str, bytes)):
            raise TypeError("args must be a sequence of CLI arguments, not a string")
        if input is None and (output_dir is not None or frame_rate is not None):
            raise ValueError("output_dir and frame_rate require an input path")
        if frame_rate is not None:
            if isinstance(frame_rate, bool) or not math.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
                raise ValueError("frame_rate must be a positive finite number")
        executable = "viame.exe" if os.name == "nt" else "viame"
        install = os.environ.get("VIAME_INSTALL")
        if install and (Path(install) / "bin" / executable).is_file():
            executable = str(Path(install) / "bin" / executable)
        command = [executable, "run", self.path]
        if input is not None:
            command.append(str(Path(input).expanduser().resolve()))
        if output_dir is not None:
            command.extend(["-o", str(Path(output_dir).expanduser().resolve())])
        if frame_rate is not None:
            command.extend(["-frate", str(frame_rate)])
        command.extend(os.fspath(arg) for arg in args)
        kwargs.setdefault("check", True)
        kwargs.setdefault("text", True)
        return subprocess.run(command, **kwargs)

    def close(self):
        self.closed = True
        if self._work is not None:
            self._work.cleanup()
            self._work = None

    def __enter__(self):
        if self.closed:
            raise ValueError("Pipeline is closed")
        return self

    def __exit__(self, *args):
        self.close()


def open(filename, frame_rate=None, *, pipeline=None, embedded=False,
         inputs=None, outputs=None):
    """Load imagery, annotations, a pipeline, or a supported model bundle.

    An image returns an ImageContainer. Image lists and videos return lazy
    iterators of ImageContainers. VIAME CSV, DIVE JSON and COCO JSON return
    ObjectTrackSets through their respective readers. Paths in lists resolve
    relative to the list first, then to the working directory.

    Pipelines and models return a Pipeline handle; call its run() method to
    execute it. For a ZIP with several pipes, select a member with pipeline=.
    embedded=True replaces readers and writers with in-memory adapters.
    inputs/outputs optionally select source/sink process names to replace.

    ``frame_rate`` is an optional positive, finite video sampling rate in Hz;
    it cannot increase the source rate. Both ``open(path, 5)`` and
    ``open(path, frame_rate=5)`` are supported. Non-video inputs reject it.
    """
    path = Path(filename).expanduser()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file() and not path.is_dir():
        raise ValueError("Expected a regular file or image directory: {}".format(path))
    if frame_rate is not None:
        if isinstance(frame_rate, bool):
            raise ValueError("frame_rate must be a positive finite number")
        frame_rate = float(frame_rate)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be a positive finite number")
    kind = file_kind(path)
    if frame_rate is not None and kind != "video":
        raise ValueError("frame_rate is supported only for video inputs")
    if pipeline is not None and kind != "model":
        raise ValueError("pipeline is supported only for ZIP pipeline selection")
    if (embedded or inputs is not None or outputs is not None) and kind not in ("pipe", "model"):
        raise ValueError("Embedded options are supported only for pipelines and models")
    if not embedded and (inputs is not None or outputs is not None):
        raise ValueError("inputs and outputs require embedded=True")
    if kind in ("pipe", "model"):
        if embedded:
            from ._embedded import EmbeddedPipeline
            return EmbeddedPipeline(path, pipeline=pipeline, inputs=inputs, outputs=outputs)
        return Pipeline(path, pipeline=pipeline)
    if kind == "image":
        return _load_image(_image_reader(), path)
    if kind == "video":
        return VideoSequence(path, frame_rate)
    if kind in ("image_list", "directory"):
        if kind == "directory":
            paths = [p for p in sorted(path.iterdir()) if p.is_file() and file_kind(p) == "image"]
        else:
            with builtins.open(path, encoding="utf-8-sig") as stream:
                paths = image_list_entries(path, stream)
        if not paths:
            raise ValueError("No images in {}".format(path))
        for entry in paths:
            if not os.path.isfile(entry):
                raise FileNotFoundError(str(entry))
            if file_kind(entry) != "image":
                raise ValueError("Image list contains a non-image: {}".format(entry))
        return ImageSequence(paths)
    if kind == "json":
        with builtins.open(path, encoding="utf-8-sig") as stream:
            format_name = json_annotation_format(json.load(stream))
        if format_name:
            return _annotations(path, format_name)
    if kind == "csv":
        with builtins.open(path, newline="", encoding="utf-8-sig") as stream:
            header = False
            for row in csv.reader(stream):
                if not row or not any(c.strip() for c in row):
                    continue
                if row[0].lstrip().startswith("#"):
                    header |= row[0].startswith("# 1: Detection or Track-id")
                    continue
                if is_viame_csv_row(row):
                    return _annotations(path, "viame_csv")
                break
            else:
                if header:
                    return _annotations(path, "viame_csv")
    raise ValueError("Unsupported input for viame.open: {} ({})".format(path, kind))
