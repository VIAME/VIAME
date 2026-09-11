# Golden behaviour recordings

A recording is what an implementation does *before* it is replaced. It is the
evidence that a replacement behaves the same, which registry and pipeline
checks cannot give: those say a name still resolves, not that it still
computes the same pixels.

Each group under this directory belongs to one dependency removal:

| Group | Recorded from | Replaced by |
|---|---|---|
| `vxl` | `arrows/vxl` filters and image_io, `plugins/vxl` filters | phase 3 |
| `video` | `arrows/ffmpeg` `video_input` and `video_output` | phase 4 |
| `codecs` | the `ocv` image_io, decoding and writing every container | phase 7 |

| `calib` | `viame::read_stereo_rig`, and `cv::FileStorage` on every calibration document | phase 7 |
| `opencv` | the `ocv_*` filters, splits, motion detector and detectors, the SIFT/SURF/FLANN feature chain, and the pipelines that use them | phase 7 |
| `measurement` | the stereo disparity matcher, the calibration target detector, and the stereo calibration pipeline end to end | phase 7 |

The video group is replayed against every reader and writer registered under it, not just the replacement: `ffmpeg` and `pyav` and `ffmpeg_cli` all have to reproduce the same recording.

## Kinds of case

A case's `kind` says what shape its outputs have and which runner produces
them. Most are one array per input; four are not.

| Kind | Output | Compared |
|---|---|---|
| `image_filter`, `image_io`, `decode`, `round_trip`, `split_image`, `detect_motion`, `disparity`, `pipeline` | one image per input | shape, dtype and values, at the implementation's tolerance |
| `detect` | a list of detections as JSON | box exactly, scores to 1e-6 relative, notes exactly |
| `features`, `matches`, `tracks`, `homography`, `fundamental` | named arrays in an `.npz` | exactly, or on agreement where the implementation is randomised |
| `calibration`, `nodes` | parsed structure as JSON | exactly |
| `calibration_pipeline` | the calibration a stereo pipeline wrote | against the recording **and** against the rig the fixtures were rendered through |

`calibration_pipeline` is the only case with a right answer rather than only
a previous answer. `measurement_fixtures.py` renders twelve views of a
chessboard through a stereo rig whose parameters are known exactly -- a
planar target through a pinhole camera is a homography, so the rendering is
exact -- and `check_calibration_truth` holds the result to it. A failure can
then say not merely "this differs from the recording" but "and the recording
was correct".

## Layout

```
inputs/                     committed fixture images, never regenerated
inputs/codecs/              committed encoded containers, never regenerated
<group>/manifest.json       every case: impl, config, inputs, output digests
<group>/<kind>/<impl>/<variant>/<input>.png|.npz
```

Outputs are PNG when the result is 8 or 16 bit (viewable, exact) and a
compressed `.npz` otherwise, which covers float and boolean results and
multi-channel float. Neither ever rescales.

The fixtures are small synthetic images built by `fixtures.py` from a fixed
seed: a gradient, bars and a disc so conversion and morphology have edges, a
mask for the filters that only take one, seeded noise so the histogram and
percentile paths see a distribution, and a 12 frame sequence with a moving
disc for the temporal filters. They are committed, so replaying a golden does
not depend on `fixtures.py` or on any image decoder.

## Recording

Against an install that still has the old implementation:

```
source <install>/setup_viame.sh
python3 tests/golden/record.py vxl
```

The video group is recorded separately, because a decoded 1080p frame is not
worth committing and would mostly be measuring the codec:

```
source <install>/setup_viame.sh
python3 tests/golden/record_video.py
```

`video/manifest.json` holds, per clip, the frame count, every presentation
time, and digests of the first, middle and last frame; a decode throughput
figure measured on a 1080p clip built from the committed fixtures rather than
committed itself; and, per pipeline that writes a video, what `ffprobe` makes
of the file.

The codec group's inputs are encoded files rather than arrays -- PNG at 8 and
16 bit, gray, RGB and RGBA; JPEG at 4:2:0 and 4:4:4; BMP; TIFF stripped at 8
and 16 bit, uncompressed, LZW and PackBits; and one tiled TIFF, which is the
case `lite-removals.md` 2.2 expects the in-house reader to hand off rather
than decode. They wrap the same pixels as the still fixtures, so a decoded
PNG can be compared against `inputs/rgb8.png` directly. `codec_fixtures.py`
generates them and needs ImageMagick's `convert` for the tiled one: Pillow
writes strips only.

Two kinds of case are recorded per container set. `decode` is what the
image_io reads each container as. `round_trip` writes a fixture out through
the image_io and reads it back with the same one, which is what catches a
writer that changes channel order or bit depth on the way out -- the BMP
round trip records 16 bit gray coming back as 8 bit, because that is what BMP
can hold.

Since P7-T02 the codec group is replayed twice, once under each name, and
both run the same code: `core` decodes through `library/video_io/codecs` --
stb for PNG, JPEG and BMP, in house for TIFF -- and hands anything those
decline to the python `pil` image_io, and `ocv` is an alias of it now that
`arrows/ocv`'s reader is gone. The replacement table for a group is
`REPLACEMENTS_BY_GROUP` in `test_golden.py`, keyed by group because `ocv` is
an image_io here and a split_image and a merge_images elsewhere.

Three things in that recording are what a self-consistent implementation
would have got wrong, and are worth knowing before changing a codec:
OpenCV **saturates** when narrowing 16 bit to 8, rather than shifting or
rescaling; a gray BMP is palettised 8 bit, not replicated 24 bit BGR; and a
palettised BMP decodes to one plane, not to the three that expanding the
palette gives.

The calibration group records two things. `calibration` is what
`viame::read_stereo_rig` -- through
`viame.core._measurement.load_stereo_calibration` -- makes of each committed
calibration source, which is the contract every VIAME caller sees. `nodes` is
one level below: every node of each OpenCV YAML and XML document as
`cv::FileStorage` parses it, which is what `library/file_io/opencv_yaml` has
to reproduce in P7-T05, including the nodes VIAME does not read yet. Both are
JSON and are compared exactly -- these are text files holding decimal
literals, so a parser that rounds differently is wrong.

The OpenCV group covers five kinds of case: `image_filter` and `split_image`
and `detect_motion` as arrays, `detect` as the detections themselves -- a box
moving by a pixel is what a golden should say, not a few thousand changed
pixels -- and `pipeline` end to end. Three fixtures are its own, in
`opencv_fixtures.py`: clean circles for the Hough detector, a heat map with
blobs either side of the shipped `min_area`, and a BG Bayer mosaic of the
shared RGB fixture. The VXL fixtures are gradients, bars and a disc, which is
enough for a per-pixel filter and says nothing about a detector.

It also records what each implementation **refuses**. `ocv_convert_color`
rejects a single channel image and `ocv_enhancer`'s denoising rejects
anything but 8 bit colour; a replacement that quietly started accepting them
would be a change no output comparison could see, because there is no output
to compare. `opencv_cases.REFUSES` says which and why, and the replay asserts
that they still raise -- except where the refusal is not deterministic, which
`UNSTABLE_REFUSAL` names.

`record.py` and `record_video.py` both refuse to overwrite an existing
recording without `--force`, so a golden cannot be quietly redefined by the
code it is meant to be checking. Re-record only when a task says the recorded
behaviour changes, and say why in `design/STATUS.md`.

The config variants in `cases.py` are the ones the shipped pipelines use, read
out of every `.pipe` and `.conf` in the install, plus each implementation's own
defaults. Adding a variant means adding it there and re-recording.

## Replaying

```
ctest -L GOLDEN
```

or directly:

```
source <install>/setup_viame.sh
python3 -m pytest tests/golden/test_golden.py -v
```

Shape and dtype always have to match. Values have to match within the
tolerance for that implementation in `test_golden.py`, which is exact by
default; a replacement that genuinely cannot be bit exact adds its own entry
there with the reason and the measured numbers.

## Where the recorder and the replay differ

They do not, with one exception, and the exception is written into
`calib_runner.py`. Everywhere else a golden works because one *registered
name* changes implementation underneath and both calls stay identical. A file
format has no registry to change, so the `nodes` case records through
`cv::FileStorage` (`dump_document_reference`) and replays through
`library/file_io/opencv_yaml` (`dump_document`).

## Cases whose values are not a contract

`cases.py` lists five, with reasons. Two are `vxl_convert_image`'s
`random_grayscale` augmentation, which draws from the global RNG. The other
three are VXL bugs found while recording:

- `vxl_threshold` in percentile mode on a multi-plane image returns an
  uninitialised buffer. `percentile_threshold_above` sizes its output to one
  plane, then hands `vil_plane( dst, 0 )` to `vil_threshold_above`, which
  resizes the view it is given to the source's plane count; the resize
  detaches the view and `dst` is never written. Single-plane inputs are fine,
  and the one shipped pipeline that uses `vxl_threshold` uses absolute mode,
  so the path is latent.
- `vxl_color_commonality` in grid mode leaves part of its output
  uninitialised, and differs from call to call inside one process.

For these the recording pins the shape, dtype and that the configuration is
accepted, and nothing else. The replacement should be correct rather than
bug compatible.

The video group has one of its own, in `test_video.py` rather than here
because it is a whole-file property. The C++ writer muxed its packets with no
duration, so the mp4 muxer derived each sample's duration from the next
sample's decode time and gave the last one zero; the track then ends before
its final sample and every decoder trims it. A pipeline that writes N frames
produces a file that plays N-1, and `filter_to_video.pipe` over six fixture
frames writes five. The recording says five, and `WRITER_DIVERGENCE` says a
replacement writes six.

## Implementations whose values move run to run

Not a bug in the replacement and not a divergence to document: some
implementations are randomised, and a recording of one can only be a contract
on agreement.

`ocv_flann_based` builds randomised KD-trees and OpenCV seeds them from the
clock, so it gives 45 or 46 matches from the same 81 descriptors twice in one
process. `feature_cases.UNSTABLE` names it and everything downstream, and
`feature_runner.compare_unstable` says what is still held: a match set keeps
its size to within 10% and 90% of its recorded pairs, and a track set's
feature locations are exact -- those are the detector's, and the detector is
deterministic -- while how many of them link into a two-frame track may move
by 10%.

Both halves matter together. A port that lost the extractor's replaced
feature set would leave the locations untouched and make the linking
nonsense, so the exact half catches what the loose half cannot.

## Recording something the framework has no runner for

Four files, in the order they are needed:

1. `<group>_fixtures.py`, if the existing fixtures will not do. Say in its
   docstring why they will not: the 96 by 64 gradient is enough for a
   per-pixel filter and says nothing about a stereo matcher.
2. `<group>_cases.py`, holding the implementations, the variants and the
   reasons each variant exists. This is the specification; a reader should be
   able to tell from it what is covered and what is not.
3. `<group>_runner.py`, if driving it needs more than `runner.py` has.
   Whatever the recorder and the replay share goes here, so neither can drift.
4. A `record_<group>` function in `record.py` and a branch in
   `run_case`/`is_registered`.

Then record once, against an install that still has the old implementation,
and commit the result before touching the implementation. `record.py
--append` adds only the cases the manifest does not have, which is what to
use once part of a group has already been replaced -- a recorder must never
be run against the code it is meant to be checking.
