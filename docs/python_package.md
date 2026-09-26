VIAME as a Python Package
=========================

`pip install viame` installs two things: the `viame` command line tool, and
the `viame` python package that the tool itself is built on. Anything the
pipelines do can be driven directly from python, because the pipelines are
assembled out of the same algorithm interfaces this package exposes.

    pip install viame

Linux, CPython 3.10 through 3.14. GPU support comes from whichever CUDA
`torch` pulls in; no separate CUDA installation is needed. Every example
below was run against the published wheel with nothing downloaded beyond
the package itself.

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

For the lower-level algorithm interfaces below, load the plugin modules
before calling `create`:

    from viame.modules import modules
    modules.load_known_modules()


Running a pipeline
------------------

There are two ways, and the first is usually the one you want.

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
    from viame.modules import modules
    modules.load_known_modules()

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
    from viame.modules import modules
    modules.load_known_modules()

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
