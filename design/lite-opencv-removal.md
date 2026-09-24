Removing the OpenCV python dependency
=====================================

Goal: `pip install viame` needs no `opencv-python-headless`. The algorithms
that are OpenCV *by definition* -- the ones registered under `ocv_*` -- keep
it, behind an optional extra.

Where it stands
---------------

    96 python files import cv2
    789 call sites
    189 distinct cv2 APIs

Removing it entirely is not the goal and should not be: four of those APIs
have no VIAME equivalent and reimplementing them is a numerical project in
its own right, not a port.

    cv2.findHomography          12 sites   RANSAC + DLT + LM refinement
    cv2.calibrateCamera          9 sites   full bundle adjustment
    cv2.stereoCalibrate          3 sites   ditto, two cameras
    cv2.findChessboardCorners    3 sites   corner detection + subpixel

The rest is mostly a handful of primitives, and VIAME already has the C++
for most of them -- `library/image_kernels` (25 headers: resample, warp,
color, draw, morphology, contours, polygon, threshold, convert, filter,
match) and `library/measurement/projection.h` (`rodrigues`,
`undistort_point`, `stereo_rectify`, `rectification_maps`, `project_point`).
None of it is bound to python. That is the gap this plan closes.

What "done" looks like
----------------------

    pip install viame            no OpenCV, every shipped pipeline runs
    pip install viame[opencv]    adds the ocv_* algorithms

and a test that fails if `import cv2` reappears on a core path.

Phases
------

Each is independently shippable and testable. Golden results are the risk
throughout: anything touching resampling or camera geometry can move them,
so each phase ends with `ctest -L "GOLDEN|CRITICAL"`.

**0. Bind what already exists.** `image_kernels` and `projection` to python
as `viame.image_kernels` and additions to `viame.measurement`. No behaviour
change; this is the foundation everything else stands on. Lowest risk,
blocks everything.

**1. I/O and colour.** `imread`/`imwrite` (36 + 16 sites) to Pillow, which
is already a dependency; `cvtColor` (54 sites) to `image_kernels/color` or
numpy. Roughly 100 sites, no numerical subtlety beyond rounding.

**2. Drawing and contours.** `rectangle`, `putText`, `fillPoly`,
`findContours`, `contourArea` to `image_kernels/draw`, `polygon` and
`contours`. Visual output only, so goldens that compare rendered frames are
the check.

**3. Resampling.** `resize` (51), `remap` (18), `warpAffine` (7).

Measured against cv2 on a natural frame before committing to an approach,
because the obvious route -- Pillow, already a dependency -- turns out not
to be a drop-in:

    area       max  1   mean 0.24    drop-in
    bilinear   max 14   mean 0.55    3.9% of pixels differ by >2
    bicubic    max 19   mean 0.74    6.1%
    nearest    max 50   mean 2.26    24%, a half-pixel convention difference

So Pillow covers `INTER_AREA` and nothing else safely. The other three want
`image_kernels/resample`, which was written during this port precisely to
replace OpenCV's resizing in C++ and is already held to the goldens. Binding
that is the work here; reaching for Pillow would quietly shift every
resampled result.

**4. Camera geometry.** `initUndistortRectifyMap` (16), `undistortPoints`
(15), `Rodrigues` (11), `projectPoints` (9) to `projection.h`, which already
implements all four. The highest golden risk in the plan and the reason
`library/measurement/projection` exists at all.

**5. Quarantine the rest.** The four hard APIs, and the files that are
OpenCV by name, move behind `viame.utilities.opencv`, which imports cv2
lazily and raises a message naming the extra. 11 files are already
`ocv_*`/`opencv*` and stay as they are.

**6. Make the dependency optional.** `opencv-python-headless` moves from
`requirements.txt` to an `[opencv]` extra, and a test asserts no core module
imports cv2.

Order of attack within a phase
------------------------------

Files the shipped pipelines actually reach come first. Of the 51
implementations the 106 shipped pipelines name, these are cv2 backed:

    ocv_SURF, ocv_detect_calibration_targets, ocv_enhancer,
    ocv_optical_flow, adaptive (optical_flow), add_keypoints_from_mask,
    botsort, svm (prior_coverage_opencv)

Four of those are `ocv_*` and stay. The rest are the real target.
