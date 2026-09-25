Removing the OpenCV python dependency
=====================================

Goal: no `opencv-python-headless`, anywhere, for anything. Not an optional
extra -- gone.

Where it stands
---------------

Started at 96 files in our own code. As of the drawing and filtering
bindings:

    76 python files under library/, tools/ and tests/ import cv2
    82 more under packages/, which are vendored and are the problem below

**The vendored forks are what actually blocks the goal.** `mmcv`, `mmdet`,
`imgaug`, `mmdeploy` and `sam2` are third-party sources carried in
`packages/pytorch-libs`, they import cv2 in 82 files, and all five are
installed into `site-packages` by a normal build. So porting every one of
our own call sites gets the tree to

    grep -r "import cv2" library/     no matches

while `opencv-python-headless` still has to stay in `requirements.txt`,
because removing it breaks `import mmdet`. Phase 6 as written cannot
happen without a decision on those five, and the choice is not ours to
make silently:

  * port them too -- 82 files of upstream code we would then be
    maintaining a fork of, against forks we already struggle to rebase;
  * make them an optional extra, so the base wheel has no cv2 and the
    pipelines that need mmdet pull it in with them;
  * drop them from lite.

The second is the one that fits what lite is for, and it is cheap: the
declaration moves from the base list to an extra. Nothing below depends on
which is chosen, so the porting continues either way.

All 189 go. The four with no VIAME equivalent are written rather than
quarantined -- they are classical multi-view geometry, not OpenCV secrets:

    cv2.findHomography          12 sites   normalised DLT + RANSAC
    cv2.calibrateCamera          9 sites   Zhang's method + LM refinement
    cv2.stereoCalibrate          3 sites   the same, two cameras
    cv2.findChessboardCorners    3 sites   quad detection + subpixel refine

**All four are now written**, in `library/utilities/geometry.py`,
`calibration.py` and `chessboard.py`. What the estimate above got wrong is
which one was hard. The three fits were an afternoon each and validate
against ground truth directly. The corner detector was not, and not because
finding corners is difficult -- it matches OpenCV to a fortieth of a pixel --
but because a detector has to **refuse** as well as find, and nothing in a
recording of successful detections says so. Finding 2.33 has it.

What makes that safe to attempt is how the calibration goldens are written:
they check against **ground truth** with tolerances -- focal 2%, centre
0.5%, baseline 1% -- not against OpenCV's recorded numbers. A replacement
has to be accurate, not bit identical, and the synthetic scene has known
answers (fx 600, fy 610, baseline 120) to be accurate against. That is a
gentler target than reproducing OpenCV, and it is already in the tree.

The rest is mostly a handful of primitives, and VIAME already has the C++
for most of them -- `library/image_kernels` (25 headers: resample, warp,
color, draw, morphology, contours, polygon, threshold, convert, filter,
match) and `library/measurement/projection.h` (`rodrigues`,
`undistort_point`, `stereo_rectify`, `rectification_maps`, `project_point`).
None of it is bound to python. That is the gap this plan closes.

What "done" looks like
----------------------

    grep -r "import cv2" library/     no matches
    pip install viame                every shipped pipeline runs

and a test that fails if it comes back.

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

**1a. The float colour spaces.** `cv2.cvtColor` has *two* conventions and
the kernels carry one. On uint8, `COLOR_RGB2HSV` gives hue 0..179 so it fits
a byte; on float32 it gives hue 0..**360** and saturation and value 0..1.
`image_kernels.to_hsv` is the uint8 one, and netharn's augmenter uses the
float one -- `img01 = img / 255.0` then `hsv[:, :, 0] + hue_bound * dh` with
`hue_bound` 360. Porting those three sites to `to_hsv` rescales hue by two
without a word.

They are left on cv2 with a comment saying why, and the fix is a float
overload of the four colour conversions following cv2's float scaling. The
bindings refuse a float32 array rather than casting it, so this failed
loudly rather than silently -- which is the argument for having dropped
`forcecast`.

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

**5. The four estimators.** `findHomography` first -- normalised DLT and
RANSAC are a page of numpy and validate directly against cv2 on synthetic
correspondences. Then `findChessboardCorners`, then `calibrateCamera` by
Zhang's method with an LM refinement, then `stereoCalibrate` on top of it.
`scipy.optimize.least_squares` does the refinement; scipy is already in the
tree transitively and gets declared.

**6. Delete the dependency.** `opencv-python-headless` comes out of
`requirements.txt` entirely, the `ocv_*` algorithms are renamed or retired,
and a test fails if `import cv2` appears anywhere under `library/`.

Order of attack within a phase
------------------------------

Files the shipped pipelines actually reach come first. Of the 51
implementations the 106 shipped pipelines name, these are cv2 backed:

    ocv_SURF, ocv_detect_calibration_targets, ocv_enhancer,
    ocv_optical_flow, adaptive (optical_flow), add_keypoints_from_mask,
    botsort, svm (prior_coverage_opencv)

Four of those are `ocv_*` and stay. The rest are the real target.
