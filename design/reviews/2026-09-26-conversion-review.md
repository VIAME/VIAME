# Lite conversion review — 2026-09-26

## Scope and result

Reviewed the committed tree at `2ca6de3f8`, using VIAME main at
`62b174c481` and their common ancestor `293520113` to distinguish conversion
changes from later main changes. The review surveyed the 393 commits on
lite's first-parent history since that ancestor and the conversion plans,
then inspected replacement implementations and callers. Detailed checks
concentrated on image kernels, image I/O, video I/O, preprocessing and
bindings; this is not a line-by-line certification of all 4,057 changed
files. Main also has commits not merged into lite, so parity with current
main requires a separate merge review.

**Nine actionable findings are below, plus measured runtime regressions.**
All numerical examples and failures below were reproduced locally. P1 means
high priority because normal supported operations fail or silently change
data; P2 means a narrower failure or compatibility gap.

The other conversion session committed `9f660a85f` during this review. Its
new MOG2 and morphology work is outside this snapshot. No implementation,
shared status document, build configuration or installed binary was changed
by this review.

## 1. P1 — Python resize binds the VXL kernel instead of the OpenCV kernel

Location: `library/image_kernels/image_kernels_python.cxx:121`;
`library/image_kernels/resample.h:77`.

The Python `resize` binding calls `resize_bilinear`, whose grid spans the
first and last source pixels with VXL's `0.9999999` shortfall and truncates
the result. Converted `cv2.resize` callers need OpenCV's half-pixel grid
and byte rounding. The compatible implementation already exists as
`image_kernels::resize` in `warp.h:402`, including the byte fixed-point path.

A 2x2 checkerboard enlarged to 4x4 produces first row `[0,84,169,254]`;
OpenCV produces `[0,64,191,255]`. This changes model inputs in SiamMask,
MOTR, tracker training crops and ONNX preprocessing, and changes resized
segmentation logits. No downstream model accuracy claim is needed to
establish that these are different input tensors.

**Fix:** expose the OpenCV-compatible kernel to the converted Python callers.
Keep the VXL kernel for callers whose original contract was VXL. Add exact
reference tests for byte resize and appropriate float tests, including
noninteger ratios and identity resizing.

The Python test file explicitly accepts a 25-count resize difference in
its introduction. That acceptance conflicts with the conversion goal and
with the earlier fixed-point work documented in `lite-findings.md`.

## 2. P1 — Area resize does not integrate fractional pixel coverage

Location: `library/image_kernels/resample.h:142–163`.

The replacement rounds footprint boundaries and averages whole pixels.
OpenCV weights boundary pixels by fractional coverage. Downsampling
`[[0,0,255,0,0]]` from width 5 to width 3 returns `[[0,128,0]]` instead of
`[[0,153,0]]`. This exceeds the binding's documented one-count tolerance.
ONNX defaults to this operation, and calibration tools also use it.

There is a second arithmetic error in the same function: `+0.5` is added
unconditionally before casting, including when `T` is float. A constant
float32 image of ones becomes **1.5 everywhere**, even for integer shrink
ratios. The enlargement fallback also calls the VXL kernel from finding 1.

**Fix:** integrate fractional coverage; round only when the destination is
integer; reproduce the reference enlargement behavior. Test noninteger
ratios, mixed enlargement/shrink, and constant float images.

## 3. P1 — Native image kernels hold the GIL throughout computation

Location: `library/image_kernels/image_kernels_python.cxx`, for example
`gaussian_blur:476` and the `optical_flow` wrapper.

The wrappers never release Python's GIL. The converted OpenCV calls allow
other Python threads to execute while native computation runs. As a result,
one expensive image operation now stalls other Python pipeline stages,
including stages that would otherwise prepare GPU work.

In the probe, a second Python thread attempts to wake every 2 ms. A 1080p,
31-tap lite blur took 340 ms and produced a **340 ms gap** in that thread.
The OpenCV call allowed wakeups approximately every 2 ms. The single tick
counted during lite's call is consistent with execution at the return
boundary; it does not demonstrate concurrency during the kernel.

**Fix:** obtain buffer views and Python arguments with the GIL held, release
it around native computation, and reacquire it before constructing Python
results. Do not add a blanket release around wrappers that call Python APIs
or allocate NumPy arrays. Give stateful wrappers a separate synchronization
review. Add a concurrency check as well as throughput measurements.

## 4. P1 — Video seek bypasses the filter graph and leaves stale frames queued

Location: `library/video_io/pyav_video_input.py:335–367` and `426–453`.

Both seek methods assign a raw decoded frame to `_frame`, whereas ordinary
stepping emits a filtered planar RGB frame. `frame_image()` assumes the
latter layout. On committed `tests/golden/inputs/clip.mp4`, seeking to frame
15 yields `yuv420p`; `frame_image()` raises:

```text
could not broadcast input array from shape (135,256) into shape (270,480)
```

The existing graph is also retained across the seek. After reading the
first frame, seeking to frame 15 (1.4 s), and stepping once, the timestamp
goes backwards to **0.1 s** because an old filtered frame remains queued.
Other filters and pixel formats can silently yield incorrect data rather
than the demonstrated exception.

**Fix:** rebuild/reset the graph at a seek and feed decoded frames through
the same conversion path used by `next_frame`. Preserve a consistent
timestamp origin across filtered and decoded time bases. Test pixel data
after both kinds of seek, continued playback, and seeking after EOF. The
existing seek tests check timestamps at the landing only.

## 5. P2 — Querying frame count can rewind and corrupt an active video reader

Location: `library/video_io/pyav_video_input.py:590–605`.

When the container reports no count, `num_frames()` demuxes the active
container from its current position and then seeks it to zero. This counts
remaining packets instead of all frames, and changes playback without
resetting the frame number or filter state.

On a generated 12-frame FFV1 Matroska file, after three frames have been
read, `num_frames()` returns **0** and the next frame returns time **0.0 s**
instead of 0.3 s. Decoder buffering explains the zero remaining packets;
the rewind is independently visible. A complete continuation emits 12 more
frames numbered 4 through 15.

**Fix:** count through a separate container/decoder, or cache a full count
without touching playback state. Do not assume a packet always equals a
frame. Check queries both before and during playback on formats that lack
a declared frame count.

## 6. P1 — Image writes lose bit depth and corrupt RGBA pixels

Location: `library/utilities/imageops.py:68–74`; the same conversion appears
in `encode_image`.

Every non-uint8 input is clipped to 255 and cast to uint8. A 16-bit PNG
round trip maps `[0,256,2000,65535]` to `[0,255,255,255]`. Every three-axis
array is also forced into Pillow's RGB mode. RGBA input
`[[[10,20,30,40],[50,60,70,80]]]` is written as
`[[[10,20,30],[40,50,60]]]`: alpha bytes shift subsequent pixel channels.

These are reachable through `tools/rectify.py`, which reads unchanged image
data and writes the rectified result, and the registration blackout writer,
once finding 7 is fixed. The old `cv2.imwrite` preserved supported PNG/TIFF
bit depth and alpha. This finding concerns the Python helper; the separate
C++ codecs and `PILImageIO` have their own implementations.

**Fix:** select encoding from dtype and channel count, preserve supported
16-bit and RGBA formats, and reject unsupported combinations explicitly.
Add image write and byte-encoding round trips for both cases.

## 7. P1 — Pillow reads return read-only arrays to drawing callers

Location: `library/utilities/imageops.py:52,65`;
`tools/register.py:679–685`.

`np.asarray(PIL.Image)` produces a read-only array. Registration's
`write_blackout_images` passes it through `np.ascontiguousarray`, which
returns the same array when it is already contiguous, then calls
`fill_polygon`. The binding requests a writable buffer and raises
**`ValueError: buffer source array is read-only`**. Any ordinary byte image
with a blackout polygon reaches this failure; an image without polygons is
copied through and does not exercise it.

**Fix:** either return writable owned arrays from image-reading helpers to
preserve the old `cv2.imread` contract, or explicitly copy at every mutation
boundary. Test the actual read → blackout → write path with a polygon.

## 8. P1 — SLEAP training fails after successfully writing its first crop

Location: `library/classifiers/sleap/sleap_trainer.py:127`;
`library/utilities/imageops.py:74`.

The mechanical `cv2.imwrite` replacement retained
`if not imageops.write_image(...): raise OSError(...)`. The helper returns
`None` on success. Using the real trainer and bound detection/keypoint
types, the first valid annotated crop is written, then training immediately
raises `OSError: Unable to write crop`. No SLEAP model needs to be loaded to
reproduce this.

**Fix:** make the success contract consistent: return a success boolean from
the helper, or change this caller to rely on exceptions. Check that the
record is appended and training reaches the launcher.

## 9. P2 — ONNX preprocessing silently ignores nearest/cubic interpolation

Location: `library/object_detectors/onnx/onnx_predictor.py:242–248`.

Main maps `area`, `bilinear`, `linear`, `cubic` and `nearest` to distinct
OpenCV modes. Lite maps every value except `area` to one bilinear kernel.
Thus model specifications requesting nearest or cubic are silently changed.
This remains a bug even after fixing finding 1. With a nearest-mode 2x2
checkerboard input, the probe produces intermediate values such as 84 and
169; nearest interpolation must retain the original 0/255 values.

**Fix:** preserve the model specification's interpolation choice and the
existing fallback semantics. Test each supported mode directly on the
preprocessing tensor, independently of inference.

## Measured performance regressions

These are local operation timings, not end-to-end detector benchmarks.
Measurements use the installed Release build, NumPy uint8 input, one warmup
and the median of three calls. OpenCV 5.0.0 is explicitly limited to one
thread; lite uses its normal settings. Another session was active, so treat
the exact numbers as indicative and repeat on a quiet machine before setting
performance thresholds.

| Operation | Lite, ms | OpenCV, ms | Lite / OpenCV |
| --- | ---: | ---: | ---: |
| RGB 1920x1080 → 640x360, bilinear | 7.645 | 0.611 | 12.5x |
| RGB 1920x1080 → 640x360, area | 13.594 | 3.280 | 4.1x |
| RGB 1920x1080 → grayscale | 10.679 | 0.771 | 13.9x |
| Grayscale 1920x1080, 17-tap Gaussian, sigma 2.5 | 180.698 | 4.536 | 39.8x |

The resize timings currently compare different numerical operations, as
findings 1 and 2 show. They are evidence that the replacements are slower,
not an equivalence benchmark. The GIL regression compounds these costs in
pipelines with multiple Python stages.

The binding also forces contiguous input and copies every output pixel
through `as_array` (`image_kernels_python.cxx:87–114`). Profile those copies
and layout conversions separately from arithmetic before optimizing. The
existing work on separable filters improved lite versus its earlier version;
it has not closed the measured gap against the replaced operation.

## Reproduction and follow-up

From the source root, using the existing matching install:

```bash
source ../build/install/setup_viame.sh
OPENBLAS_NUM_THREADS=1 /usr/bin/python3.10 design/reviews/2026-09-26-probe.py
```

The probe reads Python source and its video fixture from `2ca6de3f8` using
`git show`, creates temporary media, exercises the real native bindings and
prints JSON. It neither configures nor rebuilds the project. Native kernels
come from the active install, so rerunning after another session installs
fixes will change those results. It needs the existing NumPy, Pillow, PyAV,
OpenCV and VIAME installation; it installs nothing.

The reviewed snapshot's existing Python tests for image kernels, PyAV input
and imageops also ran against the install: **176 passed in 2.71 seconds**.
Those passing tests coexist with the reproductions above. The full project
suite and model accuracy benchmarks were not run, and the shared build was
not reconfigured. The active kernel compile flags were checked: Release,
`-O3 -DNDEBUG`, with no `-march=native` or fast-math flag.

Prioritize the resize and area contracts, video seek path, image write
contract, and GIL handling. Before further broad replacements, require
reference comparisons at actual Python call sites as well as kernel-level
goldens. Cover byte/float/16-bit data, writable ownership, noninteger sizes,
model interpolation settings and stateful video transitions. Add performance
measurements on representative pipeline resolutions and a small end-to-end
model evaluation; the tests currently tolerate or never exercise several
failures above.
