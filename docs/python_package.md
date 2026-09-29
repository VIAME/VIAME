VIAME as a Python Package
=========================

`pip install viame` installs two things: the `viame` command line tool, and
the `viame` python package that the tool itself is built on. Anything the
pipelines do can be driven directly from python, because the pipelines are
assembled out of the same algorithm interfaces this package exposes.

    pip install viame

Linux and Windows, CPython 3.10 through 3.14. On Linux, GPU support comes
from whichever CUDA `torch` pulls in and no separate CUDA installation is
needed; on Windows the torch pins below are part of the install command.
Every example here was run against the published wheel with nothing
downloaded beyond the package itself.

Windows
-------

    pip install viame torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
                --extra-index-url https://download.pytorch.org/whl/cu128

The wheels are built for CUDA 12, which is what runs on the Pascal and Volta
cards CUDA 13 dropped. They carry no `+cu12` local version: the `win_amd64`
platform tag already tells them apart from the Linux wheels, and a local
version is the one thing PyPI will not accept.

**The torch pins belong in that same command.** VIAME asks for
`torch>=2.3.1,<2.11`, and on Windows PyPI answers that with the **CPU**
build, because torch marks every one of its `nvidia-*` requirements
`platform_system == "Linux"`. So:

* `pip install viame` on its own gives you a CPU torch.
* Installing a CUDA torch and *then* VIAME replaces it with the CPU one, and
  says nothing about it -- `torch.cuda.is_available()` just becomes `False`.
* Installing VIAME and *then* `pip install torch --index-url ...` does
  nothing at all: the CPU torch already satisfies `torch`, so pip leaves it
  alone. This is the one that looks like it worked.

A single resolution with the version pinned to a `+cu128` build is
unambiguous. `torchvision` is pinned beside it because it depends on an exact
torch version and pip will otherwise pair it with one it does not match. And
`--extra-index-url` rather than `--index-url`, because VIAME itself comes
from PyPI and the pytorch index does not carry it.

A CPU-only install -- plain `pip install viame` -- is a supported
configuration: VIAME's own CUDA kernels come from the `nvidia-*-cu12` wheels
it declares, not from torch, so they work either way. Nothing in torch will
use the GPU.

The two do not each load their own CUDA runtime. With both installed,
`cudart64_12.dll` is loaded once, out of `torch/lib`, and VIAME's extensions
bind to that copy.

**What the GPU has to be.** Torch's CUDA 12 builds ship `sm_70` and newer, so
on anything older than Volta -- a GTX 1080 Ti is `sm_61` -- torch imports,
reports `cuda.is_available()` as `True`, and then fails the first kernel
launch with `no kernel image is available for execution on the device`.
VIAME's own kernels are built from `sm_60` and do run on those cards.

Loading files
-------------

`viame.open` recognizes the same file formats as `viame inspect` and loads
its reader plugins automatically:

```python
import viame

image = viame.open("image.png")
array = image.image().asarray()       # RGB, preserving the reader's bit depth

for index, image in enumerate(viame.open("image_list.txt")):
    process(image)

with viame.open("example.mp4", 5) as frames:  # sample at 5 Hz
    for image in frames:
        process(image)
        print(frames.timestamp.get_frame(), frames.timestamp.get_time_seconds())

# Equivalent keyword form:
with viame.open("example.mp4", frame_rate=5) as frames:
    first_image = next(frames)

tracks = viame.open("annotations.csv")  # VIAME CSV
tracks = viame.open("annotations.json") # recognizes DIVE or COCO JSON
for track in tracks.tracks():
    print(track.id, len(track))
```

| Input | Return value |
|---|---|
| Still image | Native `ImageContainer` |
| Image list (`.txt`) | Lazy `ImageSequence` of image containers |
| Directory | Lazy image sequence, sorted by filename; subdirectories are skipped |
| Video | Lazy `VideoSequence` of image containers |
| VIAME CSV, DIVE JSON, COCO JSON | Native `ObjectTrackSet`, loaded through that format's reader |
| `.pipe`, supported model file or ZIP bundle | `Pipeline` handle, prepared using the same model wrappers as `viame run` |

An image list contains one path per line; blank lines and comments beginning
with `#` are ignored. Relative paths resolve against the list's directory
first, then against the current working directory for older lists. A directory
loads its immediate image files. Frames are loaded as iteration advances,
so opening a sequence does not load all its images into memory.

Video sampling keeps the first frame and then the first available frame at
or after each requested time. It uses presentation timestamps, including for
variable-rate video, and falls back to the source frame rate if timestamps
are unavailable. Requesting more than the source rate returns available
frames without duplicating them. The rate must be positive and finite and
is accepted only for video inputs. Native frame numbers and timestamps are
available as `frames.timestamp`; image lists have one-based frame numbers.

Iterators close at end of input or on a read error. Use `with` or call
`close()` when stopping early. Each iterator is single-pass; call `viame.open`
again to start over. Unsupported formats and invalid options raise errors;
opening a file never runs a pipeline or executes a model. Annotation files
are loaded as a complete track set, preserving the respective reader's
track IDs, frame numbering and detection data. COCO annotations without a
track ID become individual single-state tracks.

Pipeline and model files are prepared at open time, then executed explicitly:

```python
with viame.open("detector.pipe") as detector:
    result = detector.run("example.mp4", output_dir="results", frame_rate=5)

# A ZIP may contain a pipeline and its weights, or any model package that
# viame run recognizes (ONNX, netharn, or weights with companion files).
with viame.open("model.zip") as detector:
    detector.run("image_list.txt", output_dir="results")
    print(detector.path)  # prepared .pipe; valid until the handle is closed

# Multiple pipelines require an exact archive member name; no stdin prompt.
with viame.open("models.zip", pipeline="configs/detector.pipe") as detector:
    detector.run("images/", output_dir="results")

# A self-contained pipe can supply its own input and output configuration.
with viame.open("complete.pipe") as pipeline:
    pipeline.run()
```

Bare `.pt`, `.pth`, `.ckpt`, `.weights` and `.onnx` files use the same
identification and templates as `viame run`. Opening prepares configuration;
model weights load when execution begins. The installed algorithms must
support the selected model. Set up the VIAME environment as for the CLI.

`Pipeline.run` calls `viame run` synchronously and returns a
`subprocess.CompletedProcess`. It accepts file or directory inputs and uses
the command's defaults, including its default sampling rate. Additional CLI
arguments go in `args=["--no-reset-prompt", ...]`; subprocess options such as
`capture_output=True` and `timeout=60` are also accepted. Nonzero exit codes
raise `subprocess.CalledProcessError` unless `check=False` is supplied.
`frame_rate` belongs on `.run()` for pipelines; the second argument to
`viame.open` remains reserved for opening videos. Extracted files and rendered
templates are removed on `close()` or context exit. A handle can run multiple
inputs before closing. For in-memory processing, use `embedded=True` as described below, or the
lower-level native `EmbeddedPipeline` interface.

Algorithm `.create()` and native `EmbeddedPipeline()` automatically initialize
plugins when first used. `viame.open` also initializes the readers it needs.
The plugin manager remembers completed registration, so later calls do not
repeat it. The examples below need no explicit initialization call.

Explicit loading remains available for eager initialization or for inspecting
the registry (for example, `registered_names()`) before creating anything:

    from viame.modules import modules
    modules.load_known_modules()


Running a pipeline
------------------

Use the file loader for automatic adaptation, or the lower-level interfaces below.

### Open a normal pipeline for in-memory processing

Use `embedded=True` to replace file readers and standard output writers with
native memory adapters. The conversion uses the [shared C++ API](https://github.com/VIAME/VIAME/blob/main/docs/embedded_pipeline.md);
includes, configuration substitutions and relative model paths are resolved
by the native pipeline parser. Models and ZIP
bundles use the same preparation as file-based `viame.open`.

```python
with viame.open("detector.pipe", embedded=True) as detector:
    print(detector.input_names)   # e.g. ('input',)
    print(detector.output_ports)  # original writer process.port names
    detector.send(viame.open("image.png"))  # also accepts a numpy image array
    result = detector.receive(timeout=30)
    detections = result["detector_writer.detected_object_set"]

with viame.open("stereo.pipe", embedded=True) as stereo:
    stereo.send({"input1": left_image, "input2": right_image})
    result = stereo.receive(timeout=30)
```

One `send` supplies a synchronized set of camera frames. It accepts native
image containers without converting them to arrays, or 2D/3D NumPy arrays.
Camera names come from the original reader processes, including three-camera
and larger graphs. `input_ports` lists the connected reader ports. Standard
image, timestamp, filename and frame-rate ports are filled automatically:

```python
pipeline.send(image, timestamp=source_timestamp, frame_rate=30,
              values={"input.file_name": "frame00042.png"})
```

Without a timestamp, frame numbers start at one and time starts at zero at
the supplied `frame_rate` (default 1 Hz). Extra connected ports, such as
metadata or custom inputs, must be supplied through `values` using the
original `process.port` name.

Video readers and existing input adapters are recognized by process type.
Standard detection, track, image, video, homography and track-descriptor
writers, plus existing output adapters, are replaced. Custom source and sink
processes can be selected explicitly with
`inputs=["camera_a", "camera_b"]` and `outputs=["custom_writer"]`.
Selected inputs must be sources and selected outputs must be sinks. Other
processes retain their configuration and behavior, including any other file
I/O they perform. Unsupported graphs fail during preparation or native setup.

`receive()` returns a dictionary keyed by the original writer input ports,
for example `detector_writer.detected_object_set`. Result values use the
native adapter types. Sampling and batching remain active: a `send` need
not produce a result, and output branches must have compatible output rates
because the output adapter synchronizes their ports. `receive(timeout=...)`
raises `TimeoutError` if no result arrives; the handle remains usable.
Send and receive calls use bounded native queues, so interleave them instead
of sending an entire video before reading results. Calls on a handle should
be made from one calling thread.

Opening an embedded pipeline configures its algorithms, loads its models and
starts it waiting for input. Use `with` or call `close()` to send end-of-input,
finish queued work, discard unread results and release the native pipeline
and temporary files. Embedded handles use `send`/`receive`; their file-based
`run` method is disabled.

### In memory, through an embedded pipeline

`EmbeddedPipeline` runs a `.pipe` in process and lets you push data in and
pull results out, one frame at a time, with nothing written to disk. The
pipeline needs an `input_adapter` and an `output_adapter` where it would
otherwise have a reader and a writer:

    # detect.pipe
    process in
      :: input_adapter

    process detector
      :: image_object_detector
      :detector:type                               hough_circle

    process out
      :: output_adapter

    connect from in.image                     to detector.image
    connect from detector.detected_object_set to out.detected_object_set
    connect from in.image                     to out.image

Then drive it:

    import cv2, numpy as np

    from viame.adapters import EmbeddedPipeline, AdapterDataSet
    from viame.types import Image, ImageContainer

    pipeline = EmbeddedPipeline()
    pipeline.build_pipeline("detect.pipe", ".")
    pipeline.start()

    for frame in frames:                       # any numpy array
        data = AdapterDataSet.create()
        data["image"] = ImageContainer(Image(frame))
        pipeline.send(data)

        output = pipeline.receive()
        detections = output["detected_object_set"]
        print(len(detections))

    pipeline.send_end_of_input()
    pipeline.wait()

`input_port_names()` and `output_port_names()` report what the adapters
expose, which is how you find out what a given pipeline wants to be fed.
`build_pipeline`'s second argument anchors `relativepath` config entries to
the pipe file rather than the working directory.

### Through the command line tool

For a pipeline that reads and writes files anyway -- a whole video in, a
csv out -- there is nothing to gain from holding it in memory, and the tool
already handles input types, output directories and batching:

    import subprocess, pathlib

    result = subprocess.run(
        ["viame", "run", "filter_enhance", "clip.mp4", "-o", "output"],
        capture_output=True, text=True)

    frames = sorted(pathlib.Path("output").rglob("frame*.png"))
    print(result.returncode, len(frames))

The shipped pipelines live in `<sys.prefix>/configs/pipelines`, so a
pipeline can be named directly as above or given as a path. `viame run`
takes a video, an image list, or a folder.


Running a detector on your own imagery
--------------------------------------

`ImageObjectDetector` takes an `ImageContainer` and returns a detected
object set. Any numpy array will do, so the imagery need not come from a
file:

    import cv2, numpy as np

    from viame.algo import ImageObjectDetector
    from viame.types import Image, ImageContainer

    detector = ImageObjectDetector.create('hough_circle')

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (110, 120), 40, (255, 255, 255), 2)

    detections = detector.detect(ImageContainer(Image(frame)))

    for d in detections:
        box = d.bounding_box
        print(box.min_x(), box.min_y(), box.max_x(), box.max_y(), d.confidence)

`hough_circle` is used here because it needs no model, so the example runs
on a bare install. `ImageObjectDetector.registered_names()` lists what this
build has; `netharn`, `rf_detr`, `ultralytics`, `mmdet`, `onnx`, `darknet`
and the rest need a model, which comes from an add-on pack:

    viame add-ons

A detector that needs a model is configured the same way any algorithm is,
through its config block -- see `get_configuration` and `set_configuration`
below.


Running a tracker
-----------------

`TrackObjects` consumes per frame detections and returns the track set so
far. It is stateful: call it once per frame, in order.

    from viame.algo import ImageObjectDetector, TrackObjects
    from viame.types import Image, ImageContainer, Timestamp

    detector = ImageObjectDetector.create('hough_circle')
    tracker = TrackObjects.create('ocsort')
    tracker.set_configuration(tracker.get_configuration())

    for number, frame in enumerate(frames):
        image = ImageContainer(Image(frame))

        timestamp = Timestamp()
        timestamp.set_frame(number)
        timestamp.set_time_seconds(number / 30.0)

        detections = detector.detect(image)
        tracks = tracker.track(timestamp, image, detections)

        print(number, len(detections), len(tracks.tracks()))

The `set_configuration(get_configuration())` line is not optional. It is
what applies an implementation's defaults; without it a tracker raises an
`AttributeError` on its first frame for a parameter it never initialised.

`ocsort`, `bytetrack` and `botsort` all work against detections alone.
`srnn`, `siammask`, `deepsort`, `motr` and `sam3_tracker` need models.


Reading video
-------------

`VideoInput` is the same reader the pipelines use, so it handles whatever
they do -- video files through PyAV, or a list of images:

    from viame.algo import VideoInput

    reader = VideoInput.create('vidl_ffmpeg')
    reader.open('clip.mp4')

    while reader.next_frame():
        image = reader.frame_image()
        print(image.width(), image.height(), image.depth())

    reader.close()

`vidl_ffmpeg`, `ffmpeg` and `pyav` are the same PyAV backed reader under
three names, which is where the FFmpeg comes from -- this package bundles
none of its own. `image_list` reads a text file of image paths instead, and
`VideoInput.registered_names()` lists them all.

To go the other way, `viame.algo.ImageIO` reads and writes single images
and `VideoOutput` writes video.


Finding your way around
-----------------------

Every algorithm interface follows the same shape:

    Interface.registered_names()          # what this build can create
    algo = Interface.create('name')       # make one
    config = algo.get_configuration()     # its parameters, with defaults
    algo.set_configuration(config)        # apply them, after any edits

`viame.algo` holds the interfaces -- detectors, trackers, readers, writers,
filters, stereo and measurement among them. `viame.types` holds what they
exchange: `Image`, `ImageContainer`, `DetectedObject`, `Timestamp`,
`BoundingBoxD` and so on.
