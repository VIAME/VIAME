# OpenCV Python removal: implementation and remaining work

> **Superseded in part, 2026-09-28.** This was written on 2026-09-27, before
> the forks were ported. Three of its statements are no longer true:
>
> * "The package requirements still include OpenCV" -- they do not. No
>   requirements file names an imaging library (`4bba550ab`).
> * "Import checks with `cv2` blocked still fail for `imgaug`, `mmcv`,
>   `mmdet`" -- all three import and run with `cv2` blocked, by
>   `packages/patches/<fork>.patch`. `mmdeploy` and `rf-detr` likewise.
> * "kwimage/kwcoco/ndsampler ... still reach OpenCV" -- true, and no longer
>   a blocker: they are dependencies, they declare what they need, and
>   `kwimage[headless]` is how VIAME asks for it.
>
> What remains open is in `lite-completion.md` section 4.3: `sam3` has eight
> files on `cv2` and no patch, and thirteen of nineteen forks have not been
> checked at all. The rest of this document stands as the record of what
> was implemented and how it was verified.

## Scope and status

The requested target is that every feature, including optional tools and models,
works without `cv2`. **That target is not complete.** The package requirements
still include OpenCV because removing the declaration now would leave model and
training features broken. Moving those features behind an OpenCV extra does not
meet the requested target.

## Implemented

- Calibration uses a five-point essential-matrix solver and the local pose
  recovery. The polynomial coefficients retain the original author's license.
  The eight-point estimator now rejects rank-deficient samples and avoids a
  quadratic-size left singular-vector matrix on large fits.
- Stereo WLS uses the local matchers, confidence computation, and smoothing.
  The three existing WLS golden recordings pass with zero tolerance changes.
- Image viewing and calibration windows use Tk/Pillow in a dedicated process.
  Windows remain responsive while the pipeline processes subsequent frames.
  Desktop Python installations need Tk; annotations now use Pillow's font.
- CPU classifier letterboxing shares the CUDA coefficient-plan implementation.
  It supports uint8, uint16, and float32. Both training and inference classifier
  datasets use it.
- Deployed models that use only `ndsampler.CategoryTree` load a licensed local
  copy of its tensor operations, without importing the sampler's image demos.
  Archives and weights are unchanged. General sampler users are not rewritten.
- SAM2 mask cleanup, center-point selection, dataset video decoding, frame
  extraction, annotation display, and boundary scoring have replacement code.
  Source-build patches include the portable dataset/training directories on
  Linux as well as Windows.
- Video filter errors propagate instead of silently truncating a stream.
- Six GFIT pipeline baseline entries describe the existing consolidated motion
  process. No pipeline configurations were changed for this work.
- CSV ranking breaks exact score ties by class name. Previously, interned string
  addresses could change which tied class entered the top-N output when module
  imports changed. The fix was committed on main first and ported to main-lite.
  It retains the default positive-score threshold and copies only selected names.

## Verification

The private validation environment starts from an installed lite test wheel,
with changed Python files and a newly compiled image-kernel extension overlaid.
OpenCV's package is removed there, and a Python import hook rejects `cv2` and
its submodules. This is development validation, **not a newly published wheel**.

- Targeted Python tests: 182 passed across geometry, model loading, image
  resizing, video reading, GUI lifecycle, and SAM2 helpers/tools.
- Native ranking tests: three passed on main and three on lite.
- Full golden replay: 271 passed, 41 skipped; no tolerances were relaxed.
- Pipeline structural baseline: passes after the six GFIT entries were updated.
- GUI lifecycle passed under an isolated Xvfb display using Tk.
- Classifier resize: 63 frozen uint8 pixel hashes match OpenCV/kwimage exactly;
  uint16 matches exactly on the additional fixtures, and float32 is checked
  within floating-point rounding tolerance.
- SAM2 boundary metrics match 15 frozen results from the original OpenCV code.
  Region cleanup matches 2,400 OpenCV comparisons, including the block-order
  tie break when only one equally sized small island can be retained.
- The direct-import baseline checks 656 runtime Python files and now rejects
  deferred cv2 imports as well as module-level imports.

### GFIT comparison

After fixing tied-score ordering, all eight lite runs matched main in every
CSV data field (comment headers excluded). Four reference main runs and eight
lite runs covered groups/species trackers, Tules/SEAMAP clips, and CPU/auto
preprocessing. Auto selected CUDA in lite. Each clip supplied 41 frames at 5 Hz,
including frames past the motion detector's warmup.

| Clip | Detections per tracker | Track states per tracker |
| --- | ---: | ---: |
| Tules | 18 | 8 |
| SEAMAP | 709 | 369 |

The ranking replacement was compiled into small shared libraries and loaded
before the existing main/lite test libraries for this comparison. No published
pip package was changed. Clean rebuilt-wheel validation is still required before
claiming that released packages include these fixes.

### Pose comparison

The five-point solver was also compared with OpenCV on 80 synthetic scenes.
Angles below use signed translation directions, so reversed poses cannot pass.

| Scene regime (40 scenes each) | Translation median, local / OpenCV | Translation 90th percentile, local / OpenCV |
| --- | --- | --- |
| Baseline 0.25, depths 3–8, normalized feature noise 0.001 | 2.341° / 2.359° | 4.521° / 4.521° |
| Baseline 1.0, depths 3–8, normalized feature noise 0.0002 | 0.242° / 0.242° | 0.514° / 0.559° |

These are measurements on the specified scenes, not a claim of bit-identical
pose estimates for every input. Tests also cover outliers, nonfinite pairs,
invalid thresholds, degenerate samples, and preservation of mask indices.

## Remaining blockers to uninstalling OpenCV

Import checks with `cv2` blocked still fail for `imgaug`, `mmcv`, `mmdet`, and
`ndsampler`. Other packages import lazily, so an import succeeding does not prove
all their features work without OpenCV.

- MMCV/MMDetection: image transforms, mask/contour handling, video helpers, and
  visualization still call OpenCV.
- imgaug: geometric and color augmentation, blur, drawing, and artistic
  transforms still use OpenCV. Netharn training reaches this package.
- MMDeploy: deployment visualization and codebase-specific utilities still call
  OpenCV even though its top-level import succeeds.
- kwimage/kwcoco/ndsampler: general training, sampling, image transforms, and
  drawing still reach OpenCV. The CategoryTree deployment change addresses only
  the narrow deployed-classifier path.
- Dependency metadata and lock files: both direct and transitive OpenCV
  requirements must be removed only after their callers have working ports.
  Patching a local site-packages directory does not fix a published wheel.

Completing this requires maintained replacement implementations or distributable
forks for those dependency paths, followed by clean wheel installation and
feature tests with OpenCV absent. Unsupported operations must not be silently
skipped or replaced with no-op compatibility functions.
