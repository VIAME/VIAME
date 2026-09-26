# What the work has taught, and what is still open

`STATUS.md` records what each task did. This records what the doing of it
taught that the plan did not know, and what is still unresolved. A finding
here is one that changes how a *later* phase should be done; anything that
only mattered to the task it came from stays in that task's ledger note.

Written after phases 3 and 4 and the first half of phase 5.

## 1. Findings

### 1.1 One copy of a library per process, or it will not exit

Phase 5's P5-T02 said to copy `vital` into `library/` and leave kwiver
building its own for its arrows and its pipeline engine -- "temporary
duplication for one task". Both copies then load: kwiver's through sprokit,
the copy through VIAME's own libraries. The code is identical and so are the
namespace and the include guards, so it compiles, links, and produces correct
output. What it does not do is exit. Every `kwiver runner` process aborted in
`__run_exit_handlers` with a double free, eight times out of eight; taking the
copied library out of the install stopped it, three times out of three.

Identical code is not identical state. Each shared object carries its own
copy of every file-scope static, and tearing down two sets of them corrupts
the exit-handler chain. The symptom is at exit, after the work is done, which
makes it easy to mistake for a harmless nuisance -- the pipeline output was
correct in every case.

**For later phases:** an import replaces the original, it does not sit beside
it. P5-T03 and P5-T04 move sprokit and the arrows the same way vital moved:
delete kwiver's copy in the same change that builds VIAME's.

### 1.2 CMake allows a cycle only between static libraries

`core_types` and `algorithm_framework` name each other: the types need
configuration, logging and errors, and the algorithm interfaces are written in
terms of the types. Two shared libraries that link each other are rejected
outright by CMake, whatever the linker would make of it.

Kwiver cut the same code into eight libraries to keep the graph acyclic
(`logger` and `exceptions` at the bottom, then `util`, then `config`, then
`types`, then `algo`/`vpm`/`applets`/`io` on top). That split is preserved for
now because the imported code is built through kwiver's own CMakeLists.
Whether it is worth keeping is phase 8's question. `library/`'s own
CMakeLists, unused at present, folds both directories into one library and
aliases `viame_core_types` to it.

**For later phases:** the layout document's DAG describes directories, not
shared objects. Do not assume one library per directory.

### 1.3 The registry contract stops being enforced when an implementation
moves to python

`registry-dump` cannot introspect a python implementation's configuration:
the pybind trampoline returns the non-copyable `config_block` by copy, so the
dump records an `error` instead of the keys. `compare_registry.py` skips
config comparison for any entry with an `error`, in either direction. So the
moment a C++ implementation becomes a python one, its keys and defaults stop
being checked and nothing says so.

This bit in phase 4, where `video_input/ffmpeg` and `video_output/ffmpeg`
became python. The contract for those names is held instead by
`tests/library/video_io/test_pyav_video_input.py`, which reads the recorded
keys and defaults straight out of `tests/baseline/registry.json` and asserts
them against the live algorithm.

**For later phases:** every phase that moves a registered C++ implementation
to python needs the same test. Phase 7 moves several (`stabilize_image`,
`estimate_homography`, SIFT, `detect_calibration_targets`), and phase 6 moves
the stereo calibration processes.

### 1.4 What a closure over `#include` lines cannot see

`design/scripts/kwiver_reachable.py` computes which kwiver files VIAME
reaches. Five kinds of edge are invisible to it, and each one was found by a
build failure rather than by reading:

* **Registration by type.** Processes are registered as
  `reg.register_process< frame_list_process >()`, with the name coming off the
  class, so matching registered names against file contents finds nothing.
  Before the script also matched `<name>_process`, the list contained no
  processes at all -- which is most of what VIAME runs.
* **Interfaces only python implements.** `segment_via_points` and
  `perform_text_query` have no C++ implementation and no C++ includes;
  VIAME's python registers against them by name, and the binding has to
  exist. The script carries a list of these, regenerated from
  `from kwiver.vital.algo import ...` across the python sources.
* **Definitions in a differently-named source.** `get_logger` is declared in
  `logger/logger.h` and defined in `logger/kwiver_logger_manager.cxx`, which
  nothing includes. The script carries a short list of these, each one put
  there by a link error.
* **Generated headers.** `vital_config.h` and the export headers do not exist
  in the source tree, so they resolve to nothing and are simply absent from
  the closure.
* **Vendored libraries.** cereal is taken whole rather than file by file. A
  closure would have taken the reached subset and left the rest, and the
  arrows that use a different archive format would not build.

**For later phases:** the same script drives P5-T03 and P5-T04 for sprokit and
the arrows. Expect the same classes of miss, and expect the build to be what
finds them.

### 1.5 Generated code has two paths, and they are not the same path

The `kwiver.vital.algo` bindings are generated by castxml from the interface
headers. The generator uses one setting for both the directory it reads the
headers from and the prefix it writes into the `#include` lines of what it
generates. Those were the same string while both lived in kwiver; they are not
any more, and conflating them produced generated sources that included
themselves from a directory that does not exist.

**For later phases:** phase 8 rewrites these bindings by hand. Until then, any
move of an interface header has to touch both halves.

### 1.6 `-I` beats `-isystem`, and imported targets are `-isystem`

The tree this transitional build takes fletch from also holds an older
kwiver's installed headers. Once vital moved, those became stale, and they
won: CMake marks an imported target's interface includes as SYSTEM, GCC
searches every `-I` before any `-isystem`, and the dependency prefix arrived
as a plain `-I`. The compiler read `<vital/types/image.h>` from a tree nobody
was building.

The fix was to drop the prefix from the include path and have the two plugins
that genuinely needed a header from it name it themselves.

**For later phases:** phase 1 replaces this arrangement with a single build
and system dependencies, at which point the problem goes away. Until then, a
header that resolves to `~/Dev/viame/build/install/include` is a bug.

### 1.7 Aliases in python are subclasses

A C++ alias is a second `add_factory` call for the same class. Python
implementations are discovered by walking `Pluggable.__subclasses__`, so a
name is a class: `ffmpeg`, `vidl_ffmpeg` and `pyav` are three trivial
subclasses of one reader. This is not written down anywhere in the build
system document, and it is the shape every later python replacement will
need.

### 1.8 A usage scan has to read the tests as well as the pipelines

Phase 3's scan asked which names the 293 installed `.pipe` and `.conf` files
select, and that was enough there. It is not enough in general: VIAME's own
tests build pipelines in their source, and two of the eighty-four names
P5-T04 was about to drop -- the `frame_list_input` process and the
`example_detector` image_object_detector -- are used only from
`tests/plugins/pytorch/test_processes.py`. Dropping them turned four pytorch
tests red with "no such process in the registry".

**For later phases:** scan `tests/` alongside `configs/` and `examples/`.
Neither `registry.json` nor `pipes.json` covers a pipeline a test writes at
run time.

### 1.9 Installing does not delete

`cmake --install` leaves behind the plugin `.so` of anything that stops being
built. Those get loaded, fail on an undefined symbol from the library that
also stopped being built, and the failure looks like a build error rather than
stale state. This has now happened three times: the vxl and ffmpeg plugins in
phases 3 and 4, and klv, cuda and geocalc in phase 5.

**For later phases:** after turning an arrow off, delete its plugin and its
library from the install before running anything.

### 1.10 Defects found in the code being replaced

Recorded here because each is a real defect in VIAME or kwiver as shipped, not
an artefact of the port:

* The C++ FFmpeg writer **loses the last frame of every video it writes**. It
  muxed packets with no duration, so the mp4 muxer gave the final sample
  duration zero; the track then ends before its last sample and every decoder
  trims it. `filter_to_video.pipe` over six frames writes five.
* `video_output_process` built its `video_settings` only under `WITH_FFMPEG`
  and passed a null pointer otherwise, so the writer would have had no
  geometry or frame rate at all once FFmpeg went.
* Both video readers took the timestamp origin from the first frame they
  happened to see, so seeking before reading made the frame seeked to time
  zero and every later time wrong.
* `vxl_threshold` in percentile mode on a multi-plane image returns an
  uninitialised buffer; `vxl_color_commonality` in grid mode leaves part of
  its output unwritten. Both recorded in `tests/golden/README.md`.
* `viame::enhance_images` was defined twice, in `plugins/vxl` and
  `plugins/opencv`, with identical mangled symbols; the loader bound one for
  both factories, load-order dependent.
* Building a `vital::image` from an interleaved numpy array costs 14 ms for a
  1080p frame against 0.5 ms for the same data in vital's own planar layout,
  because `image::copy_from` falls off its memcpy path. Every python
  implementation that returns an image pays it.
* Kwiver registers `close_loops_appearance_indexed` twice, once as
  `appearance_indexed` and once as **`multi_method`**; `close_loops_multi_method`
  exists in the tree and is never registered at all. So
  `common_image_stabilizer.pipe`, which asks for `loop_closer:type =
  multi_method` and then configures `method_1` and `method_2` under it, has
  been getting the appearance-indexed closer, and those two blocks have never
  done anything. The port keeps the registration exactly as it was --
  behaviour first -- and `close_loops_multi_method` was dropped with the rest
  of the unregistered code.
* `draw_detected_object_set`'s colour configuration means the opposite of
  what it says. `default_color` is documented "(RGB)" and defaults to
  `"0 0 255"`, and `custom_class_color`'s example is `person/3/255 0 0` with
  "Color is in RGB" beside it -- but the triple went into a `cv::Scalar` over
  a **BGR** image, so the default drew a red box and the example would have
  drawn blue. P7-T04 reproduces the behaviour rather than the documentation,
  because every existing pipeline and screenshot depends on it; which of the
  two to correct is a decision for someone who knows who is relying on which.
* `ocv_random_hue_shift` **throws about half the time** on a single channel
  image, and returns it untouched the other half. It draws
  `rand() / (RAND_MAX + 1.0)` against `trigger_percent` and returns early when
  the draw misses; when it hits, it converts BGR to HSV, which needs three
  channels. Nothing about the input decides it. Measured at 15 failures in 30
  calls on `gray16` and 14 in 30 on `gray8`. Two shipped pipelines select it,
  both on colour, so it has never fired in practice -- but a pipeline is one
  `split_image_channels` away from it. `tests/golden/opencv` records the
  refusal without asserting it, since nothing can assert a coin flip.
* **The C++ `ocv_SIFT` and `ocv_SURF` wrappers ignored their configuration
  entirely.** All five keys of each, silently. The cause is one line,
  repeated in all four wrappers (detector and extractor, SIFT and SURF):

  ```cpp
  detector.constCast< cv::FeatureDetector >() = create( ... );
  ```

  `cv::Ptr::constCast` returns a **new `Ptr` by value**, so the assignment
  replaces a temporary and the freshly configured detector is destroyed on
  the next line. The wrapper reran it before every call, and every call used
  the detector built at construction from the defaults. The recording proves
  it: every non-default variant of `features` came out byte-identical to
  `defaults` -- 81 SIFT features whether `n_features` says 0 or 20, 64-wide
  SURF descriptors with `extended` set. Two shipped pipelines,
  `common_image_stabilizer.pipe` and `utility_register_frames_3-cam.pipe`,
  ask for `hessian_threshold = 5000` and `upright = true` and have been
  getting 100 and false -- which on a 4K frame is tens of thousands of
  features where the author wanted hundreds. The python port applies the
  configuration; `tests/golden/feature_cases.py` records the four cases that
  diverge from the recording, and why.
* `common_image_stabilizer.pipe` configures SURF with `n_octave_layers`,
  which is not a registered key: the wrappers registered `n_octaves_layers`,
  with the extra s. Latent while the config was being ignored; now that it is
  not, the key still does nothing, and the misspelling is in the pipeline
  rather than in the code. The registered name is kept, since
  `registry.json` records it.
* **`cv::FlannBasedMatcher` is not deterministic.** It builds randomised
  KD-trees and OpenCV seeds them from the clock, so `ocv_flann_based` gives a
  different answer on every call -- 45 or 46 matches out of the same 81
  descriptors, twice in one process. Nothing downstream of it can be recorded
  exactly either, which is why `tests/golden/opencv`'s `matches` and `tracks`
  cases are contracts on agreement rather than on bytes, and why the two
  estimators are recorded on synthetic correspondences instead of on matched
  features: one match in or out moved the estimated homography by 29% and the
  fundamental matrix by 52%.
* `register_using_homographies.pipe` **has never configured**, and neither
  has anything else that includes `common_image_stabilizer.pipe`. It asks for
  `homography_estimator:type = core`, and `estimate_homography` has only ever
  had `ocv` and `vxl` -- the P0 baseline registry, taken before any of this
  work, has no `core`. `pipe-config` prints the key, and the runner reports
  it unresolvable and refuses the process. Not caused by the port and not
  fixed by it: which implementation the author meant is a question for
  someone who knows what the pipeline was for. `baseline:pipes` passes
  because it records which config keys resolve to a name, not whether the
  name exists.
* `ocv_SURF` needs a cv2 built with the **non-free** modules
  (`cv2.xfeatures2d.SURF_create`). The build VIAME ships has them; the
  `opencv-python-headless` wheel that `lite-removals.md` section 2.7 leaves
  in the lock files does not. The python port registers the name either way,
  so it never silently disappears from the registry, and says what is missing
  when asked to run. Open question 2.9.
* Building a `vital::descriptor` in python costs about 0.7 ms per 128 float
  values, because there is no bulk assignment: `new_descriptor` then one
  `__setitem__` per element. A frame with five thousand features spends three
  seconds there. The same shape of problem as the interleaved image copy
  above, and the same remedy when it matters -- a buffer protocol on
  `descriptor`, which phase 8 is the place for.
* `ocv_detect_calibration_targets` compares `target_type` against
  **`checkerboard`**, and an unrecognised value turns both detectors off and
  returns nothing rather than complaining. `chessboard` -- the word the
  OpenCV function it wraps is named after, and the one a user would write --
  is such a value. `tests/golden/measurement` records both, side by side, so
  the difference is visible.
* The same detector's dot path ignores `target_width` and `target_height`:
  `detect_dots` takes area and circularity limits and no grid size at all,
  so a board configured 9 by 7 returns the same 35 dots a board configured 7
  by 5 does. The checkerboard path does use them, and returns nothing at the
  wrong size, so the two halves of one algorithm disagree about whether the
  configured grid is a requirement.
* **The shipped stereo calibration cannot calibrate a board with an odd
  number of corners**, which includes its own default. `optimize_stereo_cameras`
  checks

  ```cpp
  auto features_half_size = trks.size() / 2;
  auto landmarks_half_size = lms.size() / 2;
  if( features_half_size % 2 || landmarks_half_size % 2 ) { ... return; }
  ```

  and one track per corner per camera means `trks.size()` is twice the
  corner count, so this asks the **corner count** to be even. The default
  target is 7 by 5, which is thirty-five, and the optimiser logs
  "Inconsistant features or landmarks number" and returns without
  calibrating -- no error, no output files, and the pipeline exits zero. The
  check it meant to make is `trks.size() % 2`, that the tracks divide evenly
  between the two cameras. `tests/golden/measurement`'s fixture uses an 8 by
  5 board so the rest of the chain can be recorded at all.
* **`-s global:key=value` does not reach `$CONFIG{global:key}`.** The
  override is appended as a second `config global` block after the
  substitutions have been resolved against the first, so
  `measurement_calibrate_cameras_default.pipe` run with
  `-s global:square_size=30` silently calibrates with the shipped 80 and
  returns a baseline scaled by 80/30. `pipe-config` shows both blocks, the
  first still holding the original value. The key is marked `DIVE_PARAM` in
  the pipeline, so anything that sets it this way is affected. The golden
  sets the four per-detector keys instead.
* OpenCV's chessboard auto-detection will match a **sub-grid**: with
  `auto_detect_grid` on, an 8 by 5 board was detected as 6 by 5 on one
  camera's first view and correctly on the other, and the calibration then
  refused a pair whose corner counts disagreed. Not a defect so much as the
  nature of the function -- a smaller grid inside a chessboard is also a
  chessboard -- but it means `auto_detect_grid` is not safe to leave on for
  a stereo pair, since the two cameras latch independently.
* The calibrators fit **progressively**: full model first, then fixing the
  aspect ratio, then the principal point, then each distortion coefficient
  in turn, keeping every constraint that does not worsen the error past a
  threshold. On clean data every constraint holds, so the principal point
  ends up pinned at the image centre whatever it really was -- a rig
  decentred by five pixels came back with its focal length 2% out, the
  offset traded against it. Worth knowing before reading a calibration's
  numbers, and worth reproducing rather than improving: a port that fits the
  full model will disagree with every calibration file VIAME has written.
* `detect_calibration_targets` converts to grey with **red and blue
  swapped**. It asks the bridge for an `RGB_COLOR` mat -- the only caller in
  the tree that does; every other one asks for `BGR_COLOR` -- and then hands
  it to `to_grayscale`, which takes `COLOR_BGR2GRAY`. So the two swaps do
  not cancel and the luminance weights land the wrong way round. It makes no
  difference to a grey target and moves a corner by a fraction of a pixel on
  a coloured one; the port reproduces it, because changing it would change
  every calibration the tree has produced.
* `kmedians` seeds itself with `cv::kmeans`, which draws from
  `cv::theRNG()` -- a **thread-local** generator whose state depends on what
  else has used OpenCV on that thread and not on anything the algorithm was
  given. Each implementation is reproducible within itself and they disagree
  with each other: from the same twelve calibration frames the C++ selected
  {0, 1, 4, 6, 8, 11} and the python port {0, 1, 4, 6, 9, 11}. Seeding the
  generator by hand across thirty values gives exactly those two selections
  and no others, which is what says the difference is the seed rather than
  the port. So a calibration made with `frame_count_threshold` set is not
  reproducible across implementations, and `tests/golden/measurement` holds
  that case to a relative tolerance and to ground truth instead of to its
  bytes. Worth knowing before treating a calibration file as a record of
  what its inputs were.
* A minimum-area rectangle over a symmetric shape is achieved at **several
  orientations**, and which one comes back is a tie-break rather than a
  result. `cv::rotatingCalipers` keeps the **last** edge achieving the
  minimum (`area <= minarea`), and `image_kernels::min_area_rect` kept the first
  until P7-T04b. On the golden's elliptical mask both give 462 exactly and
  they differ by 90 degrees, which moved `add_keypoints_from_mask`'s head
  and tail six pixels -- and those keypoints are what the stereo measurement
  triangulates, so six pixels is a different fish length. One character, and
  only a recording of a *symmetric* shape would ever have found it: P7-T03's
  own test compared the area alone, noting that "the orientation is
  ambiguous".

* **`ocv_color_correction`'s `water_type` does nothing.** The filter has a
  `set_water_type_presets()` that would set the three attenuation
  coefficients from `oceanic`, `coastal` or `turbid`, and **nothing calls
  it** -- not the constructor, not `set_configuration`, not the filter. The
  coefficients always come from `red_attenuation`, `green_attenuation` and
  `blue_attenuation`. `examples/image_enhancement/README.rst` documents all
  three presets with their numbers, and its own worked example asks for
  `coastal`.

  The recording settles it: `underwater_coastal` and `underwater_turbid`
  are byte-identical to the default, while a manual attenuation moves the
  image by 12 counts. Reproduced rather than fixed, because a user who has
  been getting oceanic coefficients while asking for turbid ones has tuned
  around it.
* `ocv_color_correction`'s fusion path sets `c_gamma = 0.7` around its
  shadow-recovery pass and restores it after -- and `apply_gamma_correction`
  overwrites `gamma` from the histogram whenever `gamma_auto` is set, so
  with `gamma_auto` on that 0.7 is ignored and the pass uses the automatic
  gamma of an image that has already been gamma corrected once. Eight counts
  of difference from doing what the code appears to say, and only the
  combined case in the recording shows it: gamma and fusion each agree on
  their own.

* **`warp_image_ocv`'s alpha blend is wrong on every channel but the
  first.** It computes

  ```cpp
  cv::Mat blended = warped_float.mul( weights ) +
                    dest_float.mul( 1.0 - weights );
  ```

  and `1.0 - weights`, with a `double` on the left of a `Mat`, is
  `cv::Scalar( 1, 0, 0, 0 ) - weights`. Subtracting a Scalar from a three
  channel Mat is per channel, so the expression is `[ 1 - w, -w, -w ]`: only
  the first channel blends, and the other two compute
  `warped * w - dest * w`. On a colour image that is visible garbage rather
  than a rounding difference -- 80 counts of mean error against a correct
  blend, measured.

  Latent: the only caller of `warp_image` in the tree,
  `plugins/core/warp_image_process`, never passes an alpha mask. So it takes
  the same answer as `vxl_threshold`'s percentile mode did in phase 3 -- the
  replacement is correct rather than bug compatible, and the recording stays
  as evidence of what the old one did. `tests/golden/warp_cases.py` declares
  the two affected cases as divergences with the reason.

* **`cv::cvtColor`'s HSV and HLS disagree about where to wrap a negative
  hue.** A hue comes out slightly negative whenever the maximum channel is
  red and green is just below blue; the conversion then has to bring it into
  range and halve it to fit a byte. `RGB2HSV` halves first and wraps after,
  so -0.98 degrees becomes 0; `RGB2HLS` wraps first and halves after, so the
  same pixel becomes 180 -- outside the range the hue is documented to have.
  One pixel of the 6144 in `rgb8.png` lands there, which is enough to make a
  golden fail and not enough to make anyone suspect it. `image_kernels` does
  each one the way its own conversion does.

* **`ocv_debayer` swaps red and blue.** OpenCV's Bayer constants name the
  pattern the other way round from everyone else -- the constant that
  correctly decodes a blue-at-(0, 0) mosaic is `COLOR_BayerRG2*`, not
  `BayerBG2*` -- and `debayer_filter` passes its config letter straight
  through to the constant of the same name. So `pattern: BG` on a BG mosaic
  decodes as though red were at (0, 0), and every debayered image VIAME has
  produced has its red and blue channels exchanged.

  Measured rather than argued: the recorded output of `ocv_debayer` on a
  mosaic sampled from `rgb8.png` is 255 counts from that source image and
  within 138 of it once its channels are reversed, 138 being the demosaic's
  own interpolation error on a synthetic high-frequency fixture. P7-T04b's
  port reverses the letters on the way in so it reproduces this exactly; the
  decision of whether to correct it belongs to whoever knows how many models
  were trained on images that came out of it. Two shipped pipelines debayer,
  and `filter_debayer.pipe` is one of the seven recorded end to end.
* `filter_stereo_feature_tracks`'s extent matrix has sixteen columns and
  uses four. `get_world_point_corner_values` takes the bounds of the world
  points from the **first point of each frame** rather than from all of
  them, so on a board whose corner order is fixed -- which is every board a
  chessboard detector produces -- all four "corners" collapse to that one
  point and only the first extent pair is ever written. The clustering
  therefore groups frames by where one corner landed, not by the board's
  pose. It is not obviously wrong, since that corner's position does vary
  with pose, but it is not what the code reads as doing.

* `ocv_calibrate_single_camera` **never sees the image, and guesses its
  size**. The process is fed an object track set of corners and nothing
  else, so `estimate_image_size_from_tracks` takes the furthest corner,
  adds half a box, adds 100 and rounds down to the next hundred. On the
  golden's 640 by 480 views that returns **700 by 400** -- too wide by 60
  and too short by 80 -- and that pair is what goes into `calibration.json`
  and `intrinsics.yml` as `image_width` and `image_height`. Anything that
  builds a rectification map from the file builds it at the wrong size, and
  a height that is short by 80 cannot even cover the image. The fit itself
  survives, because `cv::calibrateCamera` only uses the size for its initial
  principal point guess and the recorded run still recovers 319.25, 239.51
  against a true 319.5, 239.5; it is the written file that is wrong. The
  stereo pipeline does not have this problem -- it reads the frames itself.
  Recorded as `mono_calibration` in `tests/golden/measurement`, so the port
  reproduces it and a decision to fix it is visible as a change to that
  recording.

* A right-camera track file has to be **named after the left image**.
  `common_stereo_input_with_tracks.pipe` connects `downsampler.output_2` --
  camera one's file name -- to both `track_reader1` and `track_reader2`, and
  `read_object_track` returns only the rows whose image identifier matches
  what it was handed. So a second-camera CSV that names the second camera's
  frames reads as empty, silently, and every track becomes left-only. Found
  building the measurement golden, which had to write its right track file
  under the left frame's name to get `input_pairs_only` to see anything.
* `input_pairs_only` **does nothing on its own**. `measure_objects_process`
  deliberately refuses to pair left and right by track id -- two independent
  trackers can reuse one -- so with no `detection_pairing_method` configured
  every track is left-only, and the process short-circuits the matching
  entirely when `input_pairs_only` is the only method. The shipped template
  leaves `detection_pairing_method` empty, so `matching_methods` of
  `input_pairs_only` measures nothing at all rather than using the pairs it
  was given. The golden's case sets `keypoint_projection` to get past it.
* `pair_stereo_tracks_pytorch` **cannot have been run**.
  `_get_track_descriptor_at_frame` writes `det = state.detection` where the
  binding makes `detection` a method, so `det` is a bound method, the
  `is not None` test passes and `det.descriptor` raises `AttributeError` on
  the first state it looks at. Every path through the process reaches it. No
  pipeline in the tree selects it, which is presumably why nobody noticed;
  one add-on selects the C++ `ocv_pair_stereo_tracks` instead. Fixed in
  passing while porting the single camera calibration, which hit the same
  binding.

* `compute_disparity` **saturates its own disparity map**. The shipped
  measurement config turns the WLS filter on, and `ocv_stereo_disparity`'s
  `raw` output format writes the filtered disparity back as sixteenths in an
  **int16**. The filter fills the regions the matcher left invalid, and those
  fill values overflow: the map's maximum is 32767, which is 2047 pixels of
  disparity on a rig whose real disparity is 45. The consumer,
  `find_corresponding_point_external_disparity`, rejects only values at or
  below zero, so it takes them, and the right keypoints land four hundred
  pixels off the left edge of the image. On the golden scene the method
  measures three of five targets at a tenth of their true length while every
  other method lands within two per cent. It reproduces on the reference
  build of `main` -- the python port of `ocv_stereo_disparity` is bit
  identical to the C++ on all three WLS configurations, checked -- so this is
  upstream rather than a phase 7 regression. It is latent in the shipped
  pipelines, whose `matching_methods` does not list `compute_disparity`.
  Recorded as a `measurement` case so that the port reproduces it and a fix
  shows up as a change to the recording.

* `find_corresponding_point_epipolar_strip_ncc`'s uniqueness test **zeroes
  the wrong thing**. It suppresses the neighbourhood of the best peak by
  writing `-1` into a copy and taking the maximum of what is left, which is
  right; but `cv::minMaxLoc` over the copy still sees the suppressed region,
  and `-1` is a legitimate correlation value, so on a correlation surface
  whose second peak is below -1 -- which cannot happen -- the test would
  pass. Harmless as written, and the port keeps it: it skips the suppressed
  rectangle outright instead, which is what the code reads as doing and
  gives the same answer on every surface NCC can produce.

* `ocv_pair_stereo_detections` reports every 3D position **sixteen times
  too close**. It hands the disparity map straight to
  `cv::reprojectImageTo3D`, which documents that a 16-bit signed disparity
  "is assumed to have no fractional bits"; SGBM's has four. On the golden
  scene, whose plane sits at a known 2200 mm, every `stereo3d_z` comes out
  at about 137. The `pair_stereo` cases record the ratio rather than the
  value, so the day someone divides by sixteen the recording says so loudly
  instead of drifting.
* `write_detected_object_set_viame_csv` numbered its detections from a
  **function-local `static std::atomic< unsigned >`**, so every writer in
  the process shared one counter. A stereo pipeline has two, in their own
  threads, and their ids interleaved by whatever order the scheduler
  happened to deliver frames in: three runs of the same pipeline over the
  same data gave `1 5 7 8 9` once and `1 4 7 8 9` twice, and neither file's
  ids were contiguous. Fixed here -- the counter is a member -- because a
  recording of it is a recording of the thread scheduler. A single-writer
  pipeline is unchanged, since one writer counting from zero is what it
  already got.
* `ocv_stereo_disparity`'s **`calibration_file` path could not run at all**
  after P7-T06 moved it to python. `load_stereo_calibration` hands every
  matrix back flat, because the binding takes and returns
  `std::vector< double >` to stay clear of pybind11's Eigen caster, and the
  python implementation passed the nine element vector straight to
  `cv2.stereoRectify`, which fails inside `cvConvertScale` with an assertion
  nowhere near the cause. Introduced by the port and found by this case,
  which is the first to configure a rectifying disparity. The eight recorded
  `disparity` cases all run unrectified, which is why they did not.
* `viame score --output-plots` wrote **eight pictures and the data for six of
  them**. The renderer drew the track purity and continuity histograms
  straight from the in-memory struct, and `export_plot_data` never wrote
  either one to disk, so the directory it left behind could not reproduce its
  own contents. Invisible for as long as one implementation did both jobs;
  the moment the drawing moved to a separate program reading that directory,
  two of the eight pictures had nothing to draw from.
* The same directory **lost the average precision** it was scored with. The
  PR curve CSV carries points and nothing else, so `viame plot eval` recovered
  AP by recomputing it from the sampled curve with **eleven-point
  interpolation**, where the evaluator interpolates at every point. Those are
  different definitions and give different numbers -- 0.9455 against 0.95 on
  the per-class curve of the new recording -- so a plot of a score disagreed
  with the score printed beside it. The curve's `average_precision`, `max_f1`
  and `best_threshold` are now a comment line above the header, and the
  reader falls back to recomputation only for a file that predates it.
* **`ocv_windowed` had been running somebody else's chipping since the
  merge, and it segfaulted.** Upstream added `plugins/core/windowed_utils` --
  an OpenCV-free windowed detector, refiner and trainer under the new name
  `windowed` -- and its `prepare_image_regions` matches the one P7-T04b left
  in `plugins/opencv/windowed_utils` signature for signature, in the same
  namespace. One mangled name, two definitions, two libraries: the loader
  binds one and hands it to every caller in both. `viame::enhance_images`
  again, and nothing warns at any stage.
  The implementation it silently switched to writes out of bounds when
  `black_pad` is set. Its padding loop pairs the row index with the **column**
  stride, the column index with the **plane** stride and the plane index with
  the **row** stride -- all three rotated by one. On a 40 by 25 three-plane
  chip the column stride it uses is 1000, so the third column of the first
  row is already past a 3000 byte allocation. The `black_pad` golden case
  recorded at P7-T04b went from passing to a segfault with no source change
  on either side of it; the merge reconciliation had no way to see it,
  because nothing in either file changed.
  Both are fixed here, and the collision is fixed by deletion: there is one
  `prepare_image_regions` now, core's, and it reproduces all seven
  `ocv_windowed` recordings exactly. `tests/baseline/check_symbols.py` is the
  standing guard -- no two VIAME libraries may define the same symbol -- so
  the next one is found on purpose rather than by a crash.
* **`darknet`, recorded before P7-T08 touches it, and five defects out of
  it.** All five are in shipped code and none needed the port to find them;
  the recording found them.
  * **A `chip_step` at or above the image's own height throws an OpenCV
    assertion.** The chipping loop runs while `li < cols - net_width +
    chip_step`, which with a large step lets the start walk past the image,
    and the region of interest is then negative. On the 1000 by 800 fixture
    every step from 800 up fails, out of `cv::Mat`'s ROI constructor with
    `0 <= roi.width` -- no message about the configuration that caused it.
  * **An image smaller than the network detects nothing, silently.** The same
    bound is negative when `cols` is under `net_width`, so the loop body never
    runs, no region is produced and the set comes back empty. A 96 by 64 frame
    gives six detections with resizing disabled and **zero** in `chip` mode.
    Every VIAME pipeline that wraps this detector configures chipping.
  * **`nms_threshold` does nothing.** It is declared with a default and a
    description, copied out of the config into `m_nms_threshold`, and never
    read again. 0.10 and 0.90 give byte-identical output.
  * **`gs_to_rgb` does nothing either**, though for a better reason: darknet's
    own `mat_to_image` converts a one-channel matrix with `GRAY2BGR` whatever
    VIAME did first, so the flag only decides who does the conversion. True
    and false give byte-identical output.
  * **`adaptive` latches on the first frame.** `detect` is a const method that
    writes its choice back into the private state, so the first image's pixel
    count decides the mode for every image after it. The same 900 by 750 frame
    under the same configuration gives **31** detections on a fresh detector
    and **135** after one larger frame has gone through it. Pinned by
    `adaptive_latch`, which the golden runner records the way it records
    everything -- one algorithm, then the inputs in order -- so the recording
    carries the latch rather than describing it.
* **`image_kernels::resize` had OpenCV's sample grid but not its arithmetic**,
  and for an 8-bit image those are different answers. OpenCV resizes bytes in
  fixed point -- eleven-bit coefficients, and a vertical pass whose 8-bit
  specialisation shifts twice rather than casting once -- where `image_kernels`
  interpolated in double and rounded. On natural imagery about **an eighth of
  the pixels** come out one count apart, which is why the recording carried a
  tolerance of one rather than zero.
  A count is noise to most callers and is not noise to a caller feeding a
  neural network: the same frame resized the two ways gives `darknet` **37
  detections one way and 35 the other**, with confidences moving in the
  second decimal. So the fixed point is reproduced exactly now, and the
  recorded tolerance for `resize_bilinear_*` is zero.
  Two details decide it and neither is guessable. The vertical pass is
  `((b0 * (S0 >> 4)) >> 16) + ((b1 * (S1 >> 4)) >> 16) + 2) >> 2`, not the
  generic fixed-point cast the template names -- there is a specialisation
  for `uchar` that overrides it. And OpenCV's two axes **clamp differently**:
  the horizontal setup pins a sample that falls outside the image to the edge
  and zeroes its fraction, the vertical setup leaves the coefficients alone
  and clamps only the row index when it reads. Clamping both the same way is
  wrong by a count along the first and last row, and only along those, which
  is how it was found.
  `plugins/core/windowed_utils` takes the same path now, so the shipped
  windowed detector chips the way OpenCV chipped. Its `ocv_windowed`
  recording could not have caught this: it runs `example_detector`, which
  returns a fixed box per image, so the recording is of the chip geometry and
  never of a chip's pixels.
* **`cv::resize` given a scale is not `cv::resize` given the size that scale
  works out to**, and the difference is visible rather than arithmetic.
  OpenCV sizes the output with `saturate_cast< int >( source * scale )` and
  then goes on sampling on the grid the **requested** scale defines, not the
  grid the rounded size implies. Fitting a 1000 by 800 image to 704 by 704
  scales by 0.704 and gives 704 by 563; sampling that at 800/563 rather than
  at 1/0.704 moves pixels by up to **twenty-eight counts**.
  `plugins/core/windowed_utils` rounded first and resized to the size, so the
  shipped `windowed` detector and every `maintain_ar` fit in the tree had
  been on the wrong grid. Found by the darknet recording: seventeen of its
  twenty-four cases reproduced the moment the port compiled, and the seven
  that did not were `maintain_ar` and the three options that reach it.
  `image_kernels::resize_by_scale` is the one that takes a scale now, and all
  twenty-four reproduce exactly -- every box and every confidence, through a
  neural network.
* **The IQR AdaBoost ranking has two values, so it does not rank.**
  `iqr_session_adaboost::predict_distance` asks `cv::ml::Boost::predict` for
  `cv::ml::StatModel::RAW_OUTPUT`, meaning to get the weighted sum over the
  weak classifiers. That flag on its own does not do that on a boosted
  model -- it returns the **class label**. Getting the sum needs
  `DTrees::PREDICT_SUM` alongside it, which nothing passes.
  So every descriptor scores 0 or 1, `predict_score` -- a sigmoid of that --
  returns 0.5 or 0.731059 and nothing else, and `ordered_results()`, whose
  entire job is to score the working index and sort it, produces an order
  that within each half is whatever the hash map happened to give. The
  recording made before the port is those two numbers repeated down every
  column, for all four boosting types.
  Found by recording rather than by porting, and it changes what the port
  is: sklearn's `decision_function` returns a real margin, so replacing
  `cv::ml::Boost` fixes this rather than merely substituting an algorithm.
  **Confirmed by the port**: on overlapping classes the scikit-learn session
  gives 117 distinct scores over 40 probes where the recording had two.
* A fixture can be **too easy to test a ranking**. The first IQR scenario put
  the two clusters 1.5 standard deviations apart on six of sixteen
  dimensions, which is linearly separable: every weak learner then agrees on
  every sample and the ensemble margin saturates at its extreme for all of
  them. A saturated margin and a class label look identical -- two values --
  so that fixture could not have told the defect above from correct
  behaviour, and did not. Half a standard deviation is what made the
  difference visible.
* `Py_Finalize` with numpy and scikit-learn loaded **segfaults on the way
  out**. The embedded interpreter in the IQR test passes every assertion and
  then dies in teardown, which ctest reports as the test failing. Leaking an
  interpreter that the process is about to exit out from under costs nothing
  and is what the test does.
* **The CPU and GPU builds register the same things**, which P0-T04 assumed
  they would not. Nothing in VIAME registers conditionally on CUDA -- not one
  `#if` in any `register_algorithms.cxx` or `register_processes.cxx`, and the
  only CUDA macros in the tree are compile definitions inside
  `plugins/darknet` and the vendored `remax` extension, neither of which
  gates a factory. The two pipelines the build swaps between configurations
  differ by one config **value**, `gpu_list` 0 against None. Dumping the
  registry with `CUDA_VISIBLE_DEVICES=` empty gives 642 registrations either
  way, nothing lost and nothing gained.
  So the union baseline and the `--config` switch P0-T06 specifies would be
  machinery for a distinction that does not exist. What replaces them is
  cheaper and catches the real risk: `baseline:registry:no_gpu` and
  `baseline:pipes:no_gpu` run the same comparison with the GPU hidden, which
  is what would fail the day a registration goes behind a CUDA guard or a
  python implementation probes the device while importing -- and that second
  one no build flag controls.
* **A second build configuration cannot be built at all right now**, which is
  its own finding. `VIAME_INSTALL_PREFIX` is forced equal to the install
  directory, and `plugins/svm` resolves libsvm out of it with
  `NO_DEFAULT_PATH`, as do the TinyXML and Darknet lookups; a second build
  into its own prefix fails at configure until that prefix has been populated
  by the superbuild. Removing that coupling is what P1's single build is for.
* **Correction to the `viame_iqr` rationale.** The commit that added it said
  the session was kept out of `viame_core` so that the library everything
  links stays free of libpython. `viame_core` already links libpython and has
  for a long time: `plugins/core/CMakeLists.txt` adds `${PYTHON_LIBRARIES}`
  for DINOv3 feature matching through the Python C API. The separate library
  is still the right shape -- it isolates the scikit-learn dependency, and
  `viame_processes_core` is a MODULE which CMake refuses to let a test link
  -- but the stated reason was wrong, and a later reader would have drawn the
  wrong conclusion from it.
* **An optional dependency inside a widely linked library reaches further
  than it looks.** Moving darknet's detector into `viame_object_detectors`
  compiled and installed cleanly, and then a gtest that touches none of it
  failed to start: `libdarknet.so` drags OpenCV behind it, and
  `viame_object_detectors` is what `plugins/core` links for the chipping, so
  libdarknet and `libopencv_highgui` landed on the load path of everything
  linking `viame_core`. The detector is `viame_object_detectors_darknet` now,
  its own library behind its own plugin, and `viame_core` needs neither.
* **Directory-scoped build settings do not travel with the files they
  govern**, and both failures a move causes look like something else.
  `python/CMakeLists.txt` sets two things for everything under it:
  `kwiver_python_package` to "kwiver", because it otherwise defaults to the
  project name; and `-Wl,--no-undefined` stripped from the link flags,
  because an extension module leaves the interpreter's symbols to be resolved
  at import. Moving the `vital.types` bindings to `library/core_types` in
  P8-T01 left both behind. The first put 62 modules in
  `site-packages/viame/vital/types` where nothing imports them -- a silent
  wrong answer, not an error -- and the second failed the link with undefined
  `pybind11::cast_error::set_error`, which says nothing about libpython. The
  fix for the second is to link `${PYTHON_LIBRARIES}` rather than strip the
  flag again, which is what `library/file_io` already did for `_opencv_yaml`
  and does not weaken the check for the C++ beside it.
* **A shared-library CPython cannot find its own `libpython`.** Building it
  with `--enable-shared` into a prefix outside the system paths produces an
  interpreter that records no RPATH and dies at startup with "cannot open
  shared object file", even though the library is in the very `lib` directory
  beside it. Found by running the thing rather than by reading the recipe.
  This is what fletch's CPython avoids by building the whole of CPython a
  **second** time, statically -- a full extra compile to solve what
  `LDFLAGS=-Wl,-rpath,<prefix>/lib` solves.
* **Where the two seconds of plugin loading actually go**, measured before
  starting P8-T03 because the plan's reasoning about it is a decade old.
  Loading every plugin costs **3.8 s**. Almost none of it is what the task
  text supposes.
  * The C++ side -- seventeen `dlopen`s and their registrations -- is about a
    second, and `viame help` already costs **0.03 s** because applet dispatch
    is lazy.
  * **2.0 s is python, and not the registration design.** kwiver's
    `loaders.py::_findPluginModules` walks the plugin packages and imports
    **every module it finds**, plugin or not: 134 of them. Four cost 1.9 s
    between them and all four pull in torch.
  * The worst, at 1.22 s, is `viame.onnx.epipolar_dino_matcher` -- a genuine
    algorithm whose `_DinoFeatures` derives from `nn.Module`, so torch is
    needed to define the class, not merely to run it. **No amount of tidying
    imports fixes that**: only registering by name and importing on first
    `create()` does, which is exactly the manifest P8-T03 describes.
  * `viame.onnx.export_stereo_mapping` was 1.5 s of the same and is **not a
    plugin at all** -- a command line tool that exports ONNX models and
    registers nothing, whose every use of torch is inside a function. It was
    imported on every plugin load in the whole of VIAME because it happens to
    live in a scanned package. Its import is deferred now.
  * `SPROKIT_PYTHON_MODULES` names two packages that do not exist,
    `kwiver.sprokit.processes.pytorch` and `kwiver.arrows.python`. Harmless,
    and a sign the list is maintained by hand and not checked.
  So the order the task text puts its three parts in is the reverse of their
  value. Lazy python registration is the whole of the runtime win; folding
  the `viame_<name>` libraries into one is structural tidiness worth about a
  second between them; and the startup regression the plan was written
  against -- fifteen seconds -- is long gone.
* **A third of the scanned modules could not register anything.** Following
  the measurement above: `_load_python_module` contributes a module to the
  registry only by calling `__sprokit_register__` or
  `__vital_algorithm_register__` on it, and logs "does not have registrar
  method" otherwise. So importing a module that defines neither is pure cost,
  and **52 of the 151** scanned did exactly that -- including
  `epipolar_dino_matcher`, the most expensive module in the tree, which
  registers nothing and cost 1.22 s because it defines an `nn.Module`
  subclass. Reading the source for `def __sprokit_register__` before
  importing is exact rather than a heuristic, and skips them.
  **The wall clock does not move**, and that is the finding rather than a
  disappointment: `viame.pytorch.mit_yolo_detector` is a genuine plugin, it
  imports torch at module scope to define its classes, and under the current
  design it must be imported for its name to be known. Registration has to
  become lazy before any of this shows up as time. What the filter does show
  is the shape of the remaining problem -- with the pytorch package out of
  the list, loading is **0.79 s** and torch is never touched.
* **A package directory on `sys.path` shadows the standard library.** Running
  python with `site-packages/viame` as the working directory makes
  `viame/types.py` the `types` module, and the interpreter dies inside
  `enum` before reaching any VIAME code. The same shape as the stale
  `configs/inspect.py` that broke matplotlib for `viame plot`: a plausible
  module name in a directory that ends up on the path. Nothing to fix in the
  tree -- `viame.utilities.types` is legitimate -- but worth knowing when a
  VIAME script fails in the standard library for no reason.
* **Finding 1.9 again, and it distorts a dependency count.** The install had
  `libviame_cppdb.so` and `libviame_darknet.so` in it long after both were
  deleted from the tree, because installing does not delete. Reading the
  install's link graph to answer "what does VIAME still depend on" gave
  CppDB as a live dependency of a plugin that no longer exists. The real set,
  after clearing them, is six libraries and CppDB is not one.

### 1.11 The two-build arrangement fixes which way a dependency can point

Kwiver is configured and built before VIAME, against the same install prefix,
so a kwiver target can link a kwiver target and VIAME can link either -- but
nothing left in kwiver can link a VIAME library, and it cannot even see the
export header of one, because that header is generated in the other build
tree. This is not a rule anyone wrote down; it is what falls out of P5-T02's
transitional two-build arrangement.

It decides the order of the P5-T04 moves. `sprokit/processes/examples/process_template`
had to move to `library/examples` in the same commit that deleted `arrows/ocv`,
not because the template belongs there yet but because its only remaining
dependency, the OpenCV bridge, had become a VIAME library. Anything still in
kwiver that names a moved header has to move with it or lose the dependency.

**For later phases:** when a move leaves something behind in kwiver, check
what that something links before assuming it can stay.

### 1.12 `library/opencv_bridge` is a transitional library the layout has no row for

`arrows/ocv`'s `image_container`, `mat_image_memory`, `descriptor_set`,
`feature_set`, `match_set` and `bounding_box` are the `cv::Mat` side of the
core types. Sixty-four files include one of them. `lite-library-layout.md`'s
target tree has no home for them, because in the finished tree there are none
-- P7-T04's "remove `arrows/ocv/image_container.h` uses" is exactly the task
of deleting them.

They could not stay in kwiver (P5-T05 removes the submodule), and they could
not go in `image_kernels`, which is the code that replaces OpenCV and so must not
link it. So they are their own directory, `library/opencv_bridge`, whose
CMakeLists says in its first paragraph that phase 7 deletes it. The layout
document has a row for it now, marked transitional.

**For later phases:** P7 should end with `git rm -r library/opencv_bridge`,
not with the directory quietly emptying out.

### 1.13 The bindings cannot sit beside the code they bind yet

`lite-library-layout.md` mapped `python/kwiver/vital/types` onto
`library/core_types` "alongside the C++", which is what the user asked for
and what VIAME's own python implementations already do. It does not work
here. 48 of the 67 files under `python/kwiver/vital/types` have the same
name as the vital source they bind -- `image.cxx` binds `image.cxx` -- and
`import kwiver.vital.types` needs an `__init__.py` at every level of a tree
three deep, which one flat directory cannot provide.

So P5-T05 moved the package to VIAME's `python/kwiver`, which is where the
layout document's own target tree puts it ("`python/`: `viame` package root,
`kwiver` shim (P11)"). The two rows disagreed; the `python/` row is the one
that is achievable now, and the mapping row now says "until P11, then
alongside the C++" with the reason. When P8 replaces the generated
`vital.algo` bindings and P11 drops the `kwiver` package name, the
collisions go with them.

### 1.14 What a build system move only shows you once

P5-T05 dissolved `packages/kwiver`'s CMake into VIAME's tree. Several things
were true before it and invisible until then:

* **`viame_lite_rebase` was hiding forty-eight files.** P5-T02's macro
  rebased kwiver's file lists onto the imported tree and dropped anything that
  had not come across, reporting a count at configure time. With the
  CMakeLists beside the files the lists are literal, so what it dropped had to
  be deleted from them: twelve algorithm interfaces, thirty-one utility
  headers and sources, five range headers, three types. Nobody would have
  noticed them in the report.
* **A python extension module cannot be linked with `-Wl,--no-undefined`.**
  It leaves the interpreter's symbols undefined and resolves them at import.
  VIAME sets that flag and kwiver did not, so the bindings only stopped
  linking when they moved out of kwiver's directory scope.
* **An export set is all-or-nothing.** `viame_image_kernels` is an INTERFACE
  library that half a dozen others link. That was fine while nothing exported
  them; the moment VIAME installed a config package of its own, every target
  in it had to be exportable too.
* **`examples/plugin_creation` had not compiled in years.** It is written
  against `kwiver::vital::algorithm_impl<>` and `algo/algorithm_factory.h`,
  which kwiver 2 replaced with `PLUGGABLE_IMPL`. Nothing built it, so nothing
  said so. P5-T05's "builds against the new config package" meant porting it
  first.

Two more that only a fresh build directory showed, after the incremental one
had been green:

* **`kwiver_python_package` cannot be global.** Set at VIAME's top level it
  put every VIAME python module into the `kwiver` package, which the
  incremental tree hid because the files were already where they belonged. It
  belongs in `python/`, the one directory whose modules are kwiver's.
* **The order of `kwiver-utils` and the python paths matters.**
  `kwiver-utils` pulls in `kwiver-setup-python`, which computes an output path
  from `KWIVER_BINARY_DIR`; VIAME's own assignment has to come after it, and
  `KWIVER_BINARY_DIR` itself before it.
* **`kwiver_python_install_path` is the directory above site-packages.** The
  castxml command that generates the algorithm bindings put it on PYTHONPATH
  and then could not import castxml's own module. Incrementally the generated
  files were already there and the command never ran.

**For later phases:** an example that is not built is documentation that is
not checked. `examples/plugin_creation` is configured and built by hand here;
it should be a test. And configure into a fresh tree before calling a
build-system change done -- three of the six things in this finding only
appear there.

### 1.15 The install prefix on the include path shadows the tree being built

`plugins/svm` and `plugins/darknet` take their third-party headers by prefix
rather than from a target: `VIAME_DEPENDENCY_INCLUDE_DIRS`, which is
`${VIAME_BUILD_INSTALL_PREFIX}/include`. That is the same prefix VIAME
installs its **own** headers into. As a plain `target_include_directories`
entry it becomes an `-I`, and every `-I` beats the `-isystem` an imported
target contributes (finding 1.6), so those two plugins were compiling against
whatever `viame/core_types/*.h` the previous `make install` had left in the
prefix rather than against `library/core_types` in the source tree.

Nothing said so while the two agreed. It surfaced in P6-T06, when the source
tree stopped including `<Eigen/Geometry>` and the installed copy still did:
`refine_detections_svm.cxx` failed on a header that no longer exists in the
tree it was supposedly building.

The fix is `target_include_directories( <target> SYSTEM PUBLIC ... )`, which
turns the prefix into an `-isystem` and puts it after every `-I`. That is also
the honest description of it -- it is a third-party prefix.

**For later phases:** this is open question 2.4 with the roles reversed. 2.4
worries about *another* checkout's stale kwiver headers; the same prefix
mechanism was serving *this* build its own stale headers. Any prefix on the
include path is a prefix that can win, so a prefix that is not a target should
be SYSTEM. Phase 1 vendoring darknet and libsvm into `third_party/` removes
the last two callers.

### 1.16 The golden replay was skipping cases by name alone

`test_golden.py` skipped any recording whose implementation name appeared in
`tests/baseline/removed.json`. The name was the whole key, and a name means
nothing on its own: `ocv` is registered for eleven interfaces and `vxl` for
nine, and phase 5 removed some of each while others stayed. So a `vxl`
`bundle_adjust` nobody used, removed in P5-T04, silently turned off the twelve
`vxl` **image_io** cases -- the reader that is still registered, still aliased
to `core`, and still supposed to reproduce its phase 3 recording.

Sixteen of 105 cases were being skipped for this reason when P7-T01 noticed
it: the twelve VXL reader cases and, immediately, the four new codec cases,
because `ocv` too is a removed name for four interfaces. Keying on
(name, interface), which is what `removed.json` already records, brought the
suite from 84 passing to 100. Every one of the sixteen passes, so nothing was
hiding behind the skip -- but nothing was checking either.

**For later phases:** this is open question 2.5 as a defect rather than a
question, and finding 1.3 again in a different place. Anywhere a name is used
as a key, ask what it is a name *of*. `compare_registry.py` gets this right;
the golden replay did not, and the two read the same file.

### 1.17 VIAME's python modules were installed where nothing imports them

`kwiver_python_install_path` is where kwiver's macros put a package: they
write `${it}/${project}/${modpath}/x.py`, so it has to be the site-packages
directory itself, and `kwiver_python_output_path` builds into
`${it}/${python_sitename}/...` to match. VIAME's top level set it to
`lib/python3.10` -- one directory above -- and set
`viame_python_install_path` to the same string. But those two names mean
different things: `cmake/linux-remove-duplicate-cvs.cmake` appends
`site-packages` to `viame_python_install_path` itself, so the VIAME one is
the directory above by design.

The result was that every VIAME python module installed to
`install/lib/python3.10/viame/`, which is not on `sys.path`, while
`install/lib/python3.10/site-packages/viame/` held an older copy that is. So
`import viame.video_io` worked and got whatever the last install to that path
had left -- one of the files there still has a `.bak_prepatch` suffix from
before this branch existed.

It surfaced in P7-T02, when a new module (`video_io/pil_image_io.py`) was
added and simply could not be imported: the build tree had it, the install
had it, and python could not see it. The fix is to give the two names their
two meanings.

**For later phases:** this is the third variant of the same shape -- finding
1.15, open question 2.4, and now this. Something is installed to a path that
is not the path being read, and a stale copy at the read path hides it for as
long as nothing changes. The tell each time was a change that *added*
something rather than modifying it: an addition has no stale copy to hide
behind. Phase 9 rebuilds the python packaging; a fresh install prefix, not an
incremental one, is what would have caught this on day one.

### 1.18 The generated pybind11 trampolines cannot carry an output parameter

`cpp_to_pybind11.py` writes every trampoline method as a
`PYBIND11_OVERLOAD`, which passes each argument to python **by value**. For
an input that is right. For an output it is silently total: a python
implementation can fill the parameter all it likes and the C++ caller sees
what it passed in.

Three interfaces in this tree have one, and phase 7 moves all three to
python:

| Interface | Parameter | What an empty one costs |
|---|---|---|
| `extract_descriptors` | `feature_set_sptr& features` | The extractor may reorder or drop features to line up with its descriptors; a caller left holding the old set pairs each descriptor with the wrong feature |
| `estimate_homography` | `std::vector<bool>& inliers` | `match_features_homography` keeps only the inlier matches, so it keeps none |
| `estimate_fundamental_matrix` | `std::vector<bool>& inliers` | The same, for its own matcher |

The calling side had the same hole from the other direction: a python
*caller* of either estimator could not obtain the inliers at all.

Both are fixed by hand: the generator now prefers a trampoline written by
hand when one exists in `python/kwiver/vital/algo/trampolines/`, and an
extras file re-binds the calling side, as `extract_descriptors_extras.cxx`
already did for the feature set. The convention, stated in that directory's
README: a python implementation returns a tuple of the return value followed
by each output parameter, and the trampoline writes them back; returning the
bare value still works and warns.

A second thing the same work turned up: **two C++ overloads can share one
python name**. Both estimators declare `estimate` twice, once taking point
lists (pure virtual) and once taking feature sets and a match set (with a
C++ body that reduces to the first). The generated trampoline sent both to
whatever `estimate` python defined, so an implementation that wrote the pure
one got called with the other one's arguments. The hand-written trampoline
looks for `estimate_matches` for the second and falls back to the C++ body,
which is what should happen.

This is P8-T02's territory -- "hand-written algorithm trampolines" -- brought
forward three files at a time, the way P6-T06 brought `opencv_bridge/matrix.h`
forward and P5-T05 brought P1-T05's `add_subdirectory` forward. Phase 8
should start from these three rather than from the generator.

### 1.19 A python override cannot call its C++ base from a helper

pybind11 stops an override calling itself by comparing the *calling python
frame* with the override it is about to dispatch to. So
`super( Interface, self ).get_configuration()` reaches C++ only when it is
written inside the override itself. Move it one frame down -- into a mixin's
`_get_configuration( base )`, say, to share it between four wrapper classes
-- and the guard does not fire, `get_configuration` calls itself, and the
stack runs out. The failure is a `RecursionError` with no hint of why.

What works is keeping the `super()` call in the method that is the override
and sharing only the part that fills the block in. An inherited method is
fine: the guard compares code objects, and a subclass that inherits
`_Detector.get_configuration` has the same one.

### 1.20 Assigning to `block()` compiles and does nothing

Eigen's `block()`, `row()`, `col()`, `head()` and their fellows return
**writable proxies**, so code written against Eigen is full of

```cpp
P.block< 3, 3 >( 0, 0 ) = R;
design_matrix.row( 0 ) = point[ 0 ] * pose.row( 2 ) - pose.row( 0 );
```

P6's `core_types/math` returns **values** from all of them and offers
`set_block`, `set_row`, `set_col` for writing. Assigning to a value compiles
without a warning: it copy-assigns to a temporary, which is then destroyed.
The statement does nothing, and nothing says so.

Ten of those shipped. Two of them mattered:

* `camera_perspective::pose_matrix()` and `::as_matrix()` filled nothing, so
  **every camera in VIAME had a zero projection matrix**.
* `Triangulate_DLT` filled none of its four rows, so the design matrix was
  zero, its null vector was `(0, 0, 0, 1)` and every triangulated point came
  back as the origin.

Together: **every stereo measurement VIAME computed on this branch came out
zero.** `measurement_from_annotations_default.pipe` wrote `length=0.000000`
for every track, on real data and on synthetic. The reference build of `main`
gives 361.14 mm where this gave 0. It had been that way since P6.

`similarity_::matrix()` was a third: it set its corner to 1 and left the
rotation and translation blocks zero, so every similarity transform was the
zero map.

**The fix is the guard, not the ten call sites.** `matrix_` and `vector_`
now declare their copy, move and compound assignment operators with an
lvalue ref qualifier (`operator=( matrix_ const& ) &`), which makes assigning
to a temporary a compile error. That turned the remaining silent no-ops into
build failures immediately -- `similarity.cxx` was found that way, not by
reading.

**Two reasons it survived this long, and both are fixed here.**

`tests/plugins/core/test_measurement_utilities.cxx` has a
`compute_stereo_measurement_full` case that fails on this. It was never run:
`kwiver_discover_gtests` passed no `LABELS` to `gtest_discover_tests`, so
every discovered gtest in the tree carried **no label at all** and
`ctest -L UNIT` selected none of them. The label set this work verifies
against went from 29 tests to 300 by adding one line to the helper. A second
failure surfaced the moment it did: `color.demosaic_keeps_the_measured_sample`
had been asserting the border rule P7-T04b deliberately changed, and had been
failing, unnoticed, since.

And the measurement chain had no golden. P7-T01 recorded the filters and the
detectors; `measurement_cases.py` says in its own docstring that the
`measurement_*` pipelines were left out because they need annotated input
that is not in the tree. What closed it is a synthetic scene -- a textured
plane at a known depth, viewed through the rig the calibration fixture
already defines, with segments of known length on it -- which is the same
trick the calibration fixture uses and costs two images.

**For later phases:** any time an Eigen expression type is replaced by a
value type, ref-qualify the assignment operators **first**. The compiler
will then find every site. Doing it the other way round -- porting, then
looking -- is how ten of these got through.

### 1.21 A library nothing calls by name is not linked

The static registry hands itself to the plugin manager from a file-scope
static initializer, which is the only thing connecting the two: no symbol in
`libviame_registry.so` is named anywhere else. On this toolchain
`--as-needed` is the default, so the linker dropped the library from every
executable that listed it, the initializer never ran, and every factory in
VIAME quietly went missing. The build was clean and the tests failed as if
the algorithms had never been written.

The fix is a `-Wl,--no-as-needed` around that one library, which is what
`viame::registry` is -- an INTERFACE target holding the flag pair so the rest
of the link line keeps its default. Worth knowing before designing anything
else that registers itself on load: **the pattern only works if something
references the library, or the link line is told to keep it.** A static
library has the same problem in a different shape -- an object file no symbol
reaches is never pulled out of the archive -- and `--whole-archive` is the
equivalent answer.

### 1.22 Static registration is eager, and eagerness has a price

`viame`'s applet dispatch deliberately loaded only the applet plugins, which
is why `viame help` cost 0.04 s against the 3.8 s a full plugin load costs.
Compiling every registration function into the process and calling them all
turned that into 2.57 s: the dispatch path was now importing every python
plugin before printing a list of subcommand names.

The directory layout had been carrying information -- `applets`, `processes`,
`modules` -- and deleting the directories deleted it. Putting it back is one
word per `kwiver_add_plugin` call (read off the `SUBDIR` it already passed)
and one `if` per entry in the generated registry, and `load_all_plugins`
already took the matching bitmask. `viame help` went back to 0.09 s, the
remaining 0.05 s being the dynamic loader mapping the libraries the registry
links.

The general form: when a scan is replaced by a list, the scan's *selectivity*
has to be replaced too, not just its contents.

### 1.23 The contract you cannot run is the one worth writing down

VIAME's log line is read by DIVE, which is not in this tree. Nothing here
parses it, so nothing here would have failed if P8-T04's rewrite had changed
the separators, dropped the milliseconds, or started printing the logger's
name. The rewrite deleted three classes and a `dlopen`; the chance of it
changing the output by accident was not small.

What closed that was writing the format down first, as thirteen tests, and
committing them before touching the code. They are not an opinion about what
the format should be -- several things in them are mildly wrong-looking, the
logger's name being absent from its own output most of all -- and that is the
point. A recording of a contract you cannot execute is worth more than a
recording of one you can, because it is the only thing standing in for the
consumer.

The general form, for the phases still to come: **when a component's real
consumer is outside the repository, the recording is the consumer.** Write it
before the change, commit it separately, and let the diff of the *test* file
be empty.

### 1.24 A seam is where you keep the old semantics, not where you adopt the new ones

`kwiversys::SystemTools` and `std::filesystem` answer most path questions the
same way, which is what makes the difference dangerous. Two hundred call
sites in this tree were written against the first; P8-T05 replaced it with
the second. Rewriting them directly -- `GetFilenamePath` to `parent_path`,
`GetFilenameLastExtension` to `extension` -- compiles everywhere and is
wrong in four places:

* `GetFilenamePath( "/a/b/c.txt/" )` is `"/a/b"`. `parent_path()` reads the
  trailing separator as an empty final component and answers `"/a/b/c.txt"`.
* `GetFilenameLastExtension( ".bashrc" )` is `".bashrc"`. `extension()` is
  empty, because a leading dot is not an extension in `std::filesystem`.
* `GetPath` **appends** to the list it is given and keeps the empty entries a
  `PATH` with a stray `:` produces. A fresh vector, or a skip of the empties,
  changes which directory is searched first.
* `Directory` lists `.` and `..`. Code that walks a directory and recurses
  into every entry that is one either knows this or does not terminate.

None of the four is a bug in either library, none produces a compiler
warning, and only the last would fail loudly. So the replacement is a seam --
`util/file_system.h`, twenty-six operations with kwiversys's semantics and
each difference marked at the declaration -- rather than two hundred
individual rewrites. The seam is thirty lines longer than a direct port would
have been and is the only place the question has to be asked.

`MakeDirectory` is a fifth of the same kind, kept for the same reason: it
answers whether the directory *exists afterwards*, not whether this call
created it. A caller reading it as "I made this" is wrong on the second run
and never on the first.

The general form: **when a vendored dependency goes, the thing to preserve is
what its callers observed, not what its documentation said or what the
standard library would do.** Take the recording against the old code, put the
differences behind one file, and mark each one where someone changing it will
look.

### 1.25 `sizeof` a buffer is not how many elements it holds

`GetModuleFileNameW( handle, pathBuf, sizeof( pathBuf ) )`, in
`python/kwiver/vital/plugins/register.cxx`, where `pathBuf` is
`wchar_t[ _MAX_PATH ]`. The argument is a count of `wchar_t`, so Windows was
told the buffer was twice the size it is, and a path longer than
`_MAX_PATH / 2` would have been written past the end of it.

It was found while removing `kwiversys::Encoding` from the line below it, and
that is the finding: **the value of touching old code is not only the change
you came to make.** This one had been there since the file was written, on a
platform this branch has no CI for, in a function that only runs when
something asks which library a symbol came from. Nothing was going to find it
by testing.

The Windows branches of `register.cxx` and `python_script_applet.cxx` are
still uncompiled here. They should be read, not trusted, before the branch
claims Windows support.

### 1.26 An include that looks decorative can be carrying the settings

Two files in VIAME parse JSON with rapidjson and include
`<cereal/archives/json.hpp>` above it, calling no cereal archive at all. Each
had a comment saying the include was "for the vendored rapidjson headers",
which is half of it. The other half is three macros that header sets on the
way past:

    #define CEREAL_RAPIDJSON_ASSERT( x )  ...throw...
    #define CEREAL_RAPIDJSON_WRITE_DEFAULT_FLAGS kWriteNanAndInfFlag
    #define CEREAL_RAPIDJSON_PARSE_DEFAULT_FLAGS \
      kParseFullPrecisionFlag | kParseNanAndInfFlag

So those two files threw on an internal assertion rather than calling
`assert` -- which under `NDEBUG` is nothing at all, making the alternative
undefined behaviour rather than a crash -- wrote NaN and infinity rather than
refusing them, and parsed at full precision, which is what guarantees a
`double` VIAME wrote reads back bit for bit.

None of that is visible at the point of use. Deleting an include that "only
provides headers we get another way" would have changed all three, silently,
in a build where the assertion is compiled out.

`library/file_io/json.h` is where they are stated now. The general form:
**when a dependency goes, look at what its headers configured, not only at
what its functions did.** A macro defined before an include is a
configuration decision with no call site, and nothing about the call sites
will tell you it was made.

### 1.27 A format with no test and no external reader still has a contract

`write_stereo_rig` is exported, declared in `camera_rig_io.h`, and called by
nothing: no applet, no pipeline, no test, no python binding. It would have
been reasonable to treat its output as unconstrained.

Writing the output down before porting it found a negative zero. The right
camera's translation is `t - R c` with the left camera at the origin, and the
subtraction produces `-0.0` where the input said `0.0`; cereal wrote `-0.0`
and so does the replacement. It reads back equal to zero and compares equal
to zero, so nothing in VIAME could ever have noticed -- and a rewrite that
normalised it away would have changed the bytes of every file the function
has ever written, for no reason and with nothing to catch it.

The value was not in finding a bug; there is no bug. It was in learning that
the format had a detail at all, which is only knowable by running the code
that defines it.

### 1.28 The grammar in the comment is not the grammar in the code

`parse_attrs`, in the `.pipe` parser, carries this three lines above itself:

    attr-list ::= attr
                | attr ',' attr_list

and could not parse the second production. It accepted the comma and then
went round the loop without fetching the token after it, so the comma itself
was tested for being a flag name. `[ro,local]` had never worked.

Nothing found it because nothing could: no shipped `.pipe` or `.conf` writes
two flags, which is exactly what one would expect of a feature that has never
worked -- the absence of users is the *symptom*, and it reads identically to
the absence of need. P8-T07's removal criterion is "no shipped pipeline uses
it", and this is the shape of case where that criterion quietly gets the
wrong answer.

What found it was writing the grammar down as tests before touching the
parser, and taking the productions from the **comments** rather than only
from what the shipped pipelines exercise. Three of the ten tests came
straight from doc comments; one of the three failed.

The general form: **when recording what code does before replacing it, take
at least some of the cases from what the code claims rather than from what
its callers do.** A recording built only from live traffic re-records the
same bugs, and cannot tell "nobody needs this" from "nobody can use this".

### 1.29 A contract that bakes is not a contract that runs

P8-T07 removed port frequency. The last thing `set_core_frequency` did, after
the pipeline had solved the whole graph for a consistent set of rates, was
call `make_output_stamps()` -- and with frequency gone, nothing called it.
Every process then failed its own `step()` precondition:

    The process 'sink' was stepped before initialization

**`ctest -L BASELINE` passed.** All five of its tests did: `registry.json`
compared clean, and `pipes.json` -- 292 shipped pipelines -- resolved
perfectly. It bakes each pipeline and reports what every process and
algorithm resolves to, and it never steps one. A build in which no pipeline
could run at all satisfied the compatibility baseline completely.

What caught it was the three-test scheduler recording written an hour
earlier, which does the one thing the baseline does not: bakes a pipeline
from text, runs it, and looks at the numbers that come out the other end.

The general form: **a contract that checks construction does not check
operation, and the gap between them is invisible until something falls into
it.** `registry.json` says a name exists, `pipes.json` says a pipeline
resolves; neither says a datum ever moved. That is the same gap `skip_process`
sat in for years (open question 2.13) -- and it is worth one cheap test that
actually runs a pipeline end to end, which is now
`tests/library/pipeline_framework/test_scheduler.cxx`.

### 1.30 A recording taken from a directory records what accumulated there

P8-T08 replaces every CMake helper that decides where a file is installed, so
it opens by writing down what VIAME installs. The obvious way to do that is
to walk the install prefix. It is wrong twice over.

**The prefix is shared.** VIAME installs into the same tree as fletch, which
puts thousands of headers of its own there -- `include/cppdb`, GDAL's,
OpenCV's. Of the 5,584 paths the first manifest recorded, **4,439 were not
installed by VIAME's build at all.**

**`make install` only ever adds.** Nothing removes a file that the build has
stopped producing, so the tree also holds whatever older configurations left
there. `lib/cmake/kwiver`, 22 entries, had been dead since P5-T05 dissolved
the submodule; it was still on disk and went straight into the baseline as
though something produced it. Worse, this makes the check blind in the
direction that matters most for a task about install rules: when P8-T08
stopped installing eighteen of kwiver's CMake files, the manifest reported
**"0 gone, 6 new"**, because the eighteen were still sitting in the tree.

The fix is to record what the install *says it placed*. `cmake --install`
names every file, one line each, and re-running it when everything is up to
date takes two seconds -- cheaper than walking 80,000 files, exact about
ownership, and symmetric: it notices a removal as readily as an addition. The
manifest went from 5,584 paths to 1,786, and the 1,786 include `configs/` and
`examples/`, which the directory walk had missed entirely because they were
not in the roots I thought to list.

The general form: **when recording a build's output, ask the build, not the
filesystem.** A directory is a record of history; the build is a record of
intent, and it is intent that a change is supposed to preserve.

### 1.31 What you are replacing is everything the old code did, not what it was for

P8-T10 replaces "import every module in the package so its classes exist for
the subclass walk" with "name the class and import it when somebody wants
one". The stated purpose of the import was to make the class exist. It was
not the only thing the import did.

A VIAME python algorithm defines `__init__` and its interface method and
nothing else. `from_config`, `get_default_config` and `plugin_name` are
attached to the class by `register_vital_algorithm`, which is called from the
module's own `__vital_algorithm_register__` -- a function the scanner called
**because it had imported the module anyway**. Importing on demand skipped
it, so the lazy path handed the factory a class with none of the three
methods it needs. Nothing failed at registration; `registry-dump` listed the
algorithm quite happily. The failure was an `AttributeError` at the moment
something first tried to build one, which is the last place anybody looks
when the change under suspicion is a registration change.

It was caught by a golden recording that constructs `srnn_tracker`, not by
the compatibility baseline, which had listed the name and moved on. The
general form is the one this project keeps meeting from different directions:
**a contract that checks a thing is registered does not check that it can be
built**, and a bulk operation you delete takes its side effects with it. When
replacing one, the question is not "what was this for" but "what else
happened while it ran".

### 1.32 A list that lives in the shell can name things that do not exist

The python packages VIAME loads were named in `setup_viame.sh`, sixteen
`export SPROKIT_PYTHON_MODULES=...` lines. **Four of the sixteen named
packages this tree does not install under any option** -- `kwiver.arrows.core`,
`kwiver.arrows.python`, `kwiver.sprokit.processes.pytorch` and
`kwiver.sprokit.tests.processes`, a quarter of the list. Nothing said so,
because a package that cannot be imported is logged at debug and skipped --
the list fails open, so being wrong and being right look identical from the
outside.

That is not the only thing it cost. The list is the runtime's own inventory
of itself, and keeping it in the environment meant `viame` could not find its
own plugins unless a shell script had run first; every test, every tool and
every embedding had to arrange that. `registry-dump` reported the variable's
contents as the packages it had loaded, so the baseline recorded the two
ghosts as fact, for a year.

Moving the list into `kwiver.vital.plugins.discovery`, where the code that
reads it lives, cost nothing and settled all of it: the two names were
obviously wrong the moment they sat next to the real ones, the build options
sort themselves out (a package that is not built is not importable and is
skipped, which is what the list already relied on), and `registry-dump` can
now report what actually loaded rather than what somebody asked for. The
environment variable that remains, `VIAME_PYTHON_PLUGINS`, is for packages
VIAME does not ship -- which is the only thing an environment variable was
ever needed for.

### 1.33 A build tree keeps the answer to a question you stopped asking

Phase 1's claim is that VIAME takes nothing from fletch. The obvious way to
check it is to grep the build tree for the reference install's path. Done in
the warm `merged-build`, that turns up four targets still linking OpenCV,
forty still adding `include/` to their command line, and an rpath pointing
at fletch's `lib` -- which reads as the claim being false.

Every one of them was a **leftover**. `library/opencv_bridge`,
`library/training`, `plugins/darknet`, `plugins/cppdb`, `tests/plugins/opencv`
and all of `packages/kwiver/` are directories that no longer exist in the
source tree; CMake does not remove the `CMakeFiles` it made for them, so
their `flags.make` and `link.txt` sit there forever, recording what the build
needed the last time those directories existed. `make` never reads them
again, because nothing in the current target graph points at them.

The test that means something is whether the **source** directory still
exists -- or, better, a configure in an empty tree, where there is nothing to
leave behind. In this case one live target survived that filter, and it was
the real finding (1.34).

The general form is the mirror of 1.30. There, a directory was a record of
history and the build was a record of intent. Here the *build tree* is the
history, and it accumulates the same way an install prefix does. A warm tree
answers "what did this project ever need", not "what does it need".

### 1.34 The dependency was gone; the lookup was the dependency

`find_package( fletch NO_MODULE )` stayed in the top-level CMakeLists long
after VXL, FFmpeg, Eigen, OpenCV and zlib had all left, because by then it
found nothing and so appeared to cost nothing. It was not finding nothing.
Fletch's config file defines variables, and one of them was
`pybind11_INCLUDE_DIRS`.

`python/kwiver/sprokit/util/CMakeLists.txt` asks for
`${pybind11_INCLUDE_DIRS}`. Nothing in VIAME sets it -- P1-T03 vendored
pybind11 and replaced the `find_package` with a target -- so the line should
have expanded to nothing. Instead it expanded to the whole of the reference
superbuild's `include/` prefix, ahead of every vendored header on that
target's command line. That is the `svm.h` shadowing hazard P1-T03 wrote
about, arriving by a different route: not a stale copy in the install prefix,
but a stale *variable* still being filled in by a package nobody meant to be
using.

Two things follow. **An unused `find_package` is not free** -- it is a
hundred `set()` calls into your scope, and a variable reference is not
checked against anything. And **an undefined CMake variable expands to
nothing silently**, so `${pybind11_INCLUDE_DIRS}` is indistinguishable from
correct until something defines it wrongly. Both were fixed by naming a
target instead: `pybind11::pybind11` either exists or the configure fails.

### 1.35 The contract stopped checking exactly the thing the project was doing

`compare_registry.py` skips an entry when either side records an `error`,
because an entry the dump could not introspect has no config to compare. That
is the right rule. What it hides is worse than what it saves.

**Every python algorithm carried an error.** `python_plugin_factory` passed
the caller's `config_block&` straight to pybind11, and `config_block` is
bound as `py::class_< config_block, config_block_sptr >` -- held by shared
pointer, non-copyable -- so every call threw "return_value_policy = copy, but
type config_block is non-copyable". 82 of the 177 algorithms.

Fixing that made the error a different one, because P8-T10's lazy
declarations refuse to import. Fixing *that* -- `registry-dump --introspect`,
which the baseline test now passes and an everyday caller does not -- made
the error a third one: `register_vital_algorithm` supplied
`get_default_config = lambda cls, c: None`, a stub that set nothing, so the
dump listed **no config keys at all** for any python implementation.

So three separate mechanisms, in three different layers, each of which alone
was enough to make a python algorithm's configuration invisible. With all
three fixed the dump reports 143 config keys that it had never reported, and
the comparison against the baseline runs for the first time on **24
implementations** -- `ocv_SIFT`, `ocv_SURF`, `ocv_enhancer`, `vxl_enhancer`,
`image_io:pil`, `hough_circle`, `ocv_stereo_disparity` and the rest.

Read the list again: those are the ports. Every one of the 24 is an
algorithm this project moved from C++ to python in phases 3 to 7, and the
move is what silenced the check. **The compatibility contract stopped
checking each algorithm at the moment it became the thing that needed
checking.** The golden recordings are what actually caught the defects in
those ports, which is why nothing was wrong -- all 143 keys and defaults
match the C++ originals exactly, first run. But that was luck in the sense
that matters: the contract said nothing either way for a year of work.

The lesson is narrower than "test your tests", and sharper. A check that
degrades gracefully when it cannot run will degrade *silently*, and the
cases it cannot run on are rarely random -- here they were precisely the
cases the work was creating. When a comparison has a skip path, count what
it skipped and say so. The dump could have printed "82 of 177 algorithms
could not be introspected" on every run, and somebody would have asked.

`compare_registry.py` does both those things now. It prints `58 of 285
entries not compared` on every run, passing or failing -- a number that can
move, where a list of sixty names is something a reader scrolls past -- and
an entry the **new** dump cannot introspect where the baseline could is a
failure in its own right, not a skip. Checked against the dump taken before
the fix: 19 of the 24 come back as regressions, named, with the binding
error as the reason. The remaining 58 are entries that were python on `main`
too, so the baseline recorded nothing for them and there is nothing to
compare against; the contract cannot be stronger than what was written down.

## 2. Open questions

### 2.13 `skip_process` deadlocks, and has since it was written

`library/examples/processes/examples/skip_process.cxx` declares

    set_input_port_frequency( priv::port_input, 1 + d->skip );

and its `_step` then grabs `d->skip` data -- one fewer than it said it would.
The stamp bookkeeping advances by the declared rate, so a pipeline containing
it does not error, does not warn and does not finish: the scheduler simply
never returns. Found by writing a port-frequency test for P8-T07 and watching
it hang.

It is the only implementation of `skip` there is, no shipped `.pipe` names
it, and nothing has ever run it -- which is why a process that cannot
complete a single pipeline has sat in the tree registered and advertised.

Not fixed. P8-T07 removes port frequency, and `skip` is an example of the
feature being removed, so the one-character fix would be work in the wrong
direction. The question it leaves is the general one: **`registry.json` lists
140 processes and says nothing about whether any of them runs.** The
compatibility baseline is a contract about names, ports and defaults, which
is what it was built for; `skip` passes it perfectly. Somewhere between that
and the 292-pipeline `pipes.json` there is a gap exactly the size of "a
registered process that has never been stepped", and this is one.

**A second instance, found by P8-T09.** `close_loops:appearance_indexed` is
registered, is in the baseline, and its `check_configuration` calls
`check_nested_algo_configuration< algo::match_descriptor_sets >`. **No
implementation of `match_descriptor_sets` is registered anywhere in the
tree** -- it is the only one of the 45 interfaces in
`algorithm_framework/algo` with none. So a pipeline that selected
`appearance_indexed` could not configure. No shipped `.pipe` or `.conf`
names it, so nothing has ever tried.

Two instances of the same shape in two tasks is the argument for closing the
gap rather than recording it a third time. The cheap version is not a test
per registration: it is one test that, for every registered name, creates it
with its default configuration and calls `check_configuration`. That would
have caught `appearance_indexed` immediately and `skip` not at all, since
`skip` configures and then hangs -- so the honest form is two checks, one
for "can be created and configured" and one for "a minimal pipeline
containing it terminates", and the second is only affordable for processes
with simple ports.

**Part of the first check now exists, as a side effect of finding 1.35.**
`registry-dump --introspect` constructs every python algorithm with its
defaults to read its configuration, and `baseline:registry` runs it, so an
implementation that cannot be built with no arguments now fails the
baseline. What is still missing is `check_configuration` -- which is what
`appearance_indexed` fails -- and the "it terminates" half, which is what
`skip` fails. A third instance turned up meanwhile:
`ocv_multimodal_registration`, whose module imports and whose class exists
but which raises `type trait name "homography" not registered` on
construction. It was never in the baseline, so P8-T10 left its declaration
out rather than advertise a name that cannot be built.

### 2.12 Four python test trees that have never run

`KWIVER_ENABLE_PYTHON_TESTS` gates `python/kwiver/vital/tests`,
`python/kwiver/arrows`, `python/kwiver/tools` and
`python/kwiver/sprokit/tests`. It is not in the build's cache, so none of
them is configured, and turning it on **fails to configure**: the vital tree
names `simple_bundle_adjust.py`, `simple_convert_image.py` and a dozen more
fixtures that earlier phases pruned.

The sprokit subtree is the interesting one. It holds about thirty tests --
`test-load.py`, `test-bake.py`, `test-process.py`, `test-scheduler.py`,
`test-pipeline.py` -- which are exactly the parser and scheduler tests
P8-T07's task text says to keep and pass. They have never run here.

P8-T07 wrote its own instead, in the style of the rest of this branch, rather
than repairing three unrelated trees to reach the fourth. That leaves the
question open in two parts:

* **the vital, arrows and tools trees** reference fixtures that are gone. If
  the code they test is also gone, they should be deleted rather than left
  looking like tests; if it is not, they are coverage this branch is missing
  and does not know it.
* **the sprokit tree** still compiles against an API P8-T07 changed --
  `test-process_cluster.py` and the cluster parts of `test-pipeline.py`,
  `test-process_registry.py`, `test-bake.py` and `test-load.py` test a
  feature that no longer exists.

Either the flag comes on and all four are made to pass, or the trees go. What
should not persist is the third state they are in now: present, unbuilt, and
indistinguishable at a glance from tests that run.

### 2.11 Two JSON precisions in one library

`plugins/core` parses JSON in four places. Two of them --
`read_transform_homography_json` and, in `tools`, the `json` applet --
included `<cereal/archives/json.hpp>` and so parsed with
`kParseFullPrecisionFlag | kParseNanAndInfFlag` and wrote with
`kWriteNanAndInfFlag`. The other three -- `read_detected_object_set_dive`,
`write_object_track_set_dive` and `convert_annotations` -- include rapidjson
directly and get its defaults: an approximate `strtod`, no NaN on the way in
or out, and `RAPIDJSON_ASSERT` as `assert`, which is nothing under `NDEBUG`.

So **`viame json` and the DIVE reader parse the same DIVE file to different
doubles**, by up to one unit in the last place. Neither was chosen; the split
is which files happened to include cereal.

P8-T06 did not unify them, because changing how five readers round is a
behaviour change and this was a dependency-removal task. What it did is make
the choice visible: `library/file_io/json.h` carries the settings and says
what they are, the three that do not use it say so at their includes, and
the question is now a question.

The answer is probably "full precision everywhere": it is what the
calibration goldens are already held to, the cost is a few hundred
nanoseconds per number, and a detection box that reads back differently
depending on which VIAME component opened the file is not a property anyone
chose. **P8-T09 or P2 is where to make that call**, with the DIVE goldens
re-run rather than re-recorded, so that a difference shows up as a failure
and gets looked at.

### 2.10 `$SYSENV{}` has no users

`token_type_sysenv` expands eighteen tokens in config files -- host name,
processor count, three kinds of memory, OS version, domain name, and the
rest. P8-T05 reimplemented all eighteen against `/proc`, `uname`, `sysconf`
and `getifaddrs`, and checked each one's answer against kwiversys's side by
side before the swap.

One file in the tree writes `$SYSENV{` -- nine times, in
`python/kwiver/vital/tests/data/test_config-valid_file.txt` -- and **no test
loads it**: `git grep test_config-valid_file` finds no reference outside the
data file's own name. It is an orphan, left behind by a test that went
earlier. So the reimplementation has no in-tree caller that runs, and the
effort went into preserving answers -- "virtual memory" meaning swap,
`availablephysicalmemory` meaning MemFree+Buffers+Cached, the domain name
coming from a reverse lookup of a non-loopback interface address rather than
from the host name -- that nothing here observes.

It was done that way because the alternative was to delete a documented
config feature in the middle of a dependency-removal task, which is the wrong
task to decide it in. **P8-T09 is where it belongs**: it already covers
removing `config_block` machinery nothing uses, and the question there is
whether an out-of-tree `.pipe` may reasonably use `$SYSENV{}` -- if not, the
token type and its 527 lines go, the orphaned data file with them, and the eighteen recorded answers become the
justification for the `removed.json` entry rather than for the code.

### 2.1 An intermittent segfault in `viame train`

`viame_examples:train_netharn_cfrnn_from_viame_csv` segfaulted once, in the
run before the one that recorded P5-T02, and has passed every time since,
including a clean 25 of 25. It did not reproduce in isolation, under gdb, or
on a rerun of the same ctest selection. Given that phase 5 is precisely about
duplicated state, it is worth watching rather than dismissing: if it recurs,
the first thing to check is whether anything is loading two copies of
something.

### 2.2 Phase ordering: phase 5 wants phase 1's build

The user's ordering puts dependency removal before restructuring, and phases
3 and 4 fit that well. Phase 5 does not: importing kwiver's core into VIAME
while kwiver is a separately-configured project means the imported sources are
compiled by kwiver's build rather than VIAME's, which is backwards. It works,
and it was the cheapest way to get to one vital, but the arrangement is odd
enough to be worth naming.

P1-T05 ("kwiver as a subdirectory") is what makes it right, and P5-T05
(removing the submodule) is what makes it moot. It is no longer an open call:
**P5-T05 cannot finish without P1-T05.** What is left in the submodule --
kwiversys, the CMake that builds `library/` and `python/kwiver`, the sprokit
example and cluster processes and the schedulers, the tool runner, kwiver's
config files and its macro library -- is all either linked by something else
in kwiver or the thing doing the linking. Finding 1.11 says nothing left in
kwiver can link a VIAME library, so none of it can move one piece at a time
the way the arrows did: kwiversys is linked by every kwiver target, so it
would have to move last, and it cannot move last because after it there is
nothing left to move it into.

**P1-T05 has since landed** and the rest of P5-T05 can proceed. What it took
in this transitional setting was smaller than the phase 1 task text: the build already
configures VIAME directly rather than through the superbuild, and the
`KWIVER_ENABLE_*` values are already written down in
`build/kwiver-cache.cmake`. What it needs is (a) kwiver's top-level
`KWIVER_CMAKE_DIR`, `KWIVER_SOURCE_DIR` and `KWIVER_BINARY_DIR` taken off
`CMAKE_SOURCE_DIR`/`CMAKE_BINARY_DIR` and put on the `CURRENT` pair, (b)
VIAME setting those cache values and calling `add_subdirectory( packages/kwiver )`
where it now calls `find_package( kwiver )`, (c) a `cmake/kwiver_aliases.cmake`
giving `kwiver::vital`, `kwiver::vital_algo`, `kwiver::vital_applets`,
`kwiver::vital_config`, `kwiver::vital_exceptions`, `kwiver::vital_logger`,
`kwiver::vital_util`, `kwiver::vital_vpm`, `kwiver::kwiversys`,
`kwiver::sprokit_pipeline`, `kwiver::sprokit_pipeline_util` and
`kwiver::kwiver_adapter` as `ALIAS` targets -- that is the whole list VIAME
names -- and (d) VIAME's `kwiver_export_name` and `kwiver_plugin_*_subdir`
assignments moved after the `add_subdirectory` so they stay in VIAME's
directory scope and kwiver's own stay in kwiver's. The superbuild path
(`cmake/add_project_kwiver.cmake`) is untouched until phase 1 proper deletes
it.

Three things that list did not predict, each of which only shows up once the
two projects share one configure:

* **Kwiver's warning flags are global, not scoped.** `kwiver_warnings` is a
  `define_property(GLOBAL ...)` list that both projects' `*-flags-gnu.cmake`
  append to and both read at the end. Kwiver appends first, so VIAME compiled
  with `-Werror=non-virtual-dtor` and
  `-Werror=zero-as-null-pointer-constant` for the first time; `viame.cxx` and
  `downsample_process.cxx` failed on them, and so did the `try_compile` that
  `FindTinyXML` uses to decide whether tinyxml was built with STL support --
  which silently turned `TIXML_USE_STL` off and broke a header VIAME's CVAT
  reader includes. VIAME clears the property before reading it.
* **The python package name came from the top-level project.**
  `kwiver-utils-python.cmake` derived it from `CMAKE_PROJECT_NAME`, so the
  whole `kwiver.vital` and `kwiver.sprokit` tree installed a second time as
  `viame.vital` and `viame.sprokit`, `viame/__init__.py` was overwritten with
  kwiver's, and the two copies of each extension module aborted every process
  that loaded them with "generic_type: type ConfigKeys is already
  registered". There is a `kwiver_python_package` variable now, set in
  kwiver's own scope.
* **A configure-time copy of a build-time file.** P5-T02's
  `viame_lite_generated` macro republishes a generated header with
  `configure_file(... COPYONLY)`. That cannot work for `version.h`, which
  `kwiver_configure_file` writes with the `kwiver_configure` target at build
  time; it had only ever worked because the file was left over from the
  previous build, and the first configure in a fresh tree failed on it.

**For later phases:** a fresh build directory is the only thing that catches
the third kind, and this branch had not had one since P0. Configure into a
new tree after any change to how a generated file is produced.

### 2.3 Two copies of kwiversys and cxxopts -- answered

Both were vendored to `third_party/` by the import, and both are still
present under `packages/kwiver/vital/` -- kwiversys because kwiver's build
still builds it from there, cxxopts because only `.h`, `.cxx` and `.txx` were
deleted and it is a `.hpp`. The pairs are byte-identical today, which is the
dangerous kind of duplicate: editing one is silent.

P5-T05 settled it. `third_party/kwiversys` is the copy the build compiles
and kwiver's is deleted; `vital/applets/cxxopts.hpp` is deleted too, and it
turned out the rebase had been silently dropping it all along -- everything
had been including `<cxxopts.hpp>` from `third_party/cxxopts` since P5-T02.

### 2.4 The stale headers in the dependency tree

`~/Dev/viame/build/install/include` holds an older kwiver's `vital/`,
`sprokit/` and `arrows/`. They are off VIAME's include path now, but they are
still there, and they belong to another checkout's install that this build
borrows fletch from. Deleting them would break that checkout; leaving them
means anything that puts the prefix back on the path silently reads a
different vital.

**Since P6-T06:** the same shape bit for real, from VIAME's own install
prefix rather than the borrowed one -- see finding 1.15. The rule that came
out of it applies to both: a prefix that is not a target belongs behind
`SYSTEM`.

### 2.5 Usage is attributed by name, not by name and interface -- mostly answered

The scan that decides what P5-T04 copies asks whether a name appears as a
`type` value anywhere in the shipped pipelines. Two names -- `ocv` and
`core` -- are registered for many interfaces each, and are selected for only
some of them: `ocv` for `image_io`, `draw_detected_object_set`,
`estimate_fundamental_matrix` and `merge_images`, `core` for `image_io`,
`compute_ref_homography` and `track_features`. Attributing use by name alone
therefore keeps eleven implementations nothing selects.

Attributing properly means reading the config key that precedes `:type`,
which names the nested algorithm rather than the interface, and that mapping
is a convention rather than a rule: `homography_generator:type = core` and
`homography_estimator:type = core` both appear, and neither key is an
interface name. Getting it wrong in the keeping direction costs some code
that P5-T06 prunes anyway; getting it wrong in the removing direction breaks
a pipeline silently. So `ocv` and `core` are kept for every interface, and
the eleven are for P5-T06 to prune by compile with the whole tree in hand.

**P5-T06's answer: one of the eleven.** With every `:type = ocv` and
`:type = core` in the shipped configs read with its key -- there are only
twelve distinct keys -- and each remaining implementation checked for a
nested user in the C++, `filter_tracks:core` is the only one nothing reaches:
no config names a `filter_tracks` key and no algorithm that came across nests
one. It is gone, and its interface with it, being the only implementation.

The other ten stay, each for a reason worth writing down rather than
re-deriving. `image_io` is selected as both `ocv` and `core`.
`compute_ref_homography:core` is what `ref_homography_computer`,
`ref_computer` and `homography_generator` name, and `track_features:core` and
`estimate_fundamental_matrix:ocv` are named directly.
`estimate_homography:ocv` is nested by `match_features_homography`,
`compute_ref_homography_core` and VIAME's own
`pair_stereo_detections_process`; `feature_descriptor_io:core` is nested by
`track_features_core`; `handle_descriptor_request:core` is the only
implementation the process in `query_retrieval_and_iqr.pipe` can get.
`draw_detected_object_set:ocv`, `merge_images:ocv`, `split_image:ocv` and
`warp_image:ocv` are not named by any config, but each is the sort of thing a
DIVE workflow or an add-on selects, and phase 7 rewrites all four anyway.

### 2.6 What did not come across, and whether it should have

The rebase macro reports, per directory, which of kwiver's own file-list
entries the import did not bring: 87 files in all, among them `geo_MGRS`,
`ground_control_point`, `mesh_io`, the `text_codec` family, and eight
algorithm interfaces. P5-T06 prunes by compile and should reach the same set;
if it does not, one of the two is wrong. The list is worth diffing then.

Four python bindings were dropped outright for the same reason -- `geo_MGRS`,
`geo_covariance`, `homography_f2w`, `mesh` -- and nothing imports them. They
are not in `removed.json`, because `registry.json` does not record python
types; if the contract should cover them, it needs somewhere to record them.

### 2.7 `design/lite-kwiver-files.txt` is a record, not a query

Once kwiver's vital was deleted, `kwiver_reachable.py` could no longer compute
a closure over it, and both it and the import script now refuse to run. The
file they produced is a record of what was imported. Re-running either after
P5-T03 and P5-T04 will need the same treatment: they are one-shot per group.

### 2.8 The python bindings still live in kwiver

They cannot move until kwiver's python package goes, because two
`kwiver.vital.types` extension modules in one interpreter is a pybind11
duplicate-registration error. That makes P5-T05 the task that moves them, not
P5-T02 as the plan had it. It also means P5-T05 is larger than its text
suggests.

### 2.9 Where python's cv2 comes from after phase 7

`lite-removals.md` section 2.7 says `opencv-python-headless` stays in the
python lock files once the C++ build drops OpenCV. Today it does not come
from there: cv2 is fletch's own OpenCV build, configured with the non-free
modules, and the python and C++ sides are the same library.

Two things depend on which it becomes.

`ocv_SURF` is `cv2.xfeatures2d.SURF_create`, which the headless wheel does
not carry -- it is built without the non-free modules for licensing reasons.
On the wheel the name registers and fails when run. SURF is what
`common_image_stabilizer.pipe` selects by default, so if the wheel is the
answer then either that pipeline's default changes to `ocv_SIFT` (free since
OpenCV 4.4, and in every build) or `ocv_SURF` goes to `removed.json`. Both
are decisions about what VIAME offers, not about how it is built.

And the goldens in `tests/golden/opencv` were recorded against fletch's
build. A different OpenCV is a different implementation, and the exactness
these recordings are held to -- byte-identical descriptors -- will not
survive a version change. Whoever makes the switch should re-record and diff,
rather than loosening the tolerances to make it pass.

### 2.10 CUDA versions picked for a source build of PyTorch

P1-T02 removed `VIAME_BUILD_PYTORCH_FROM_SOURCE`, and with it the gate on
the configure check that CUDA is 12.6, 13.0 or 13.2 when PyTorch is 2.12.0:
every build now installs the wheels the lock names, and those exist only for
those versions. Main's images and release scripts set the option on through
`build_cmake_base.cmake`, so their CUDA versions were never held to the
check, and several were not ones it accepts.

- The web and default images were on `nvidia/cuda:12.9.1`. The first build
  of `docker/Dockerfile` stopped at configure; it is on 12.6.3 now, which
  matches `cuda12.lock`'s cu126 index and this machine's builds.
- The CUDA 11.8 web image cannot be built: no cu118 lock, no 2.12 cu118
  wheel. The Dockerfile says so instead of documenting its arguments.
- `cmake/build_server_rocky.sh` moves `/usr/local/cuda-12.8` into place, and
  `manual_linux_install_gen_gpu.sh` and `docker/viame_gpu_installer.docker`
  run it on a 12.8.1 image, which the check rejects. `build-release.yml`
  runs the same script on a 12.6.3 image, where `cuda-12.8` does not exist,
  so the `mv` fails before configure. This was already true on main. The
  fix is to take CUDA from whichever image the script is running in, and
  bring the manual and installer images to 12.6.3; that is not done until a
  Rocky release build has been run.

### 2.11 A lock is for the python that compiled it

The locks in `python/requirements` were compiled with python 3.10, and the
build installed them into whatever python it had found. pip-compile keeps
only the requirements whose markers hold for the interpreter it runs on, and
`base.in` marks its numpy, numba, llvmlite, imageio, scikit-image,
matplotlib, kwcoco, networkx and pandas lines by python version -- so a 3.10
lock has none of them for 3.12, and installing it on 3.12 left all nine out
without an error. Pythons that were 3.12: the Docker images (Ubuntu 24.04),
`VIAME_PYTHON_STANDALONE`, and the CPU CI job, whose apt `python3` is 3.12
on `ubuntu-latest`. The image build showed it as pip's closing complaint
that `ultralytics`, `kwimage` and `ndsampler` required packages "not
installed"; the build did not fail.

The locks are now one set per version, `py3.10/` and `py3.12/`, the second
compiled with python-build-standalone's 3.12.14. Configure installs the set
for the python it found and stops when there is none. The CI locks job
resolves each set with its own python. Compiled for 3.12, the versions that
move include numpy 2.2.6 and pandas 3.0.5, against which nothing in VIAME
has been run on 3.12 until the Docker image's CRITICAL tests are.

### 2.12 The published add-on packs are older than their pipelines

`docker/Dockerfile`'s default image adds `DEFAULT-FISH` and `GENERIC` with
`download_viame_addons.sh`, which unzips the published archive into the
install. The web preset the image is built from turns those packs off, so
`configs/add-ons/<pack>/`, which CMake installs only for an enabled pack,
never reaches the image: what the pack's pipelines are is whatever was in the
archive when it was published.

That is behind the source. Compared file by file, with each archive's MD5
matching `download_viame_addons.csv`:

- `DEFAULT-FISH` lacks `stereo_detect_and_measure_default_fish.pipe`,
  `stereo_track_and_measure_default_fish.pipe` and the two
  `utility_add_*_fish_rf_detr.pipe`, still carries
  `measurement_default_fish_fully_auto.pipe`, which b05941487 renamed, and
  has older copies of `common_fish_detector_mask_and_kp.pipe`,
  `common_stereo_fish_detector.pipe` and `index_default_fish.svm.pipe`.
- `GENERIC` has older copies of `index_generic.pipe`,
  `index_generic.svm.pipe`, `index_generic.trk.pipe` and
  `query_image_exemplar.pipe`.

So the CRITICAL `TestMeasureViaDefaultFish` skips in the default image --
`measure_via_default_fish.sh` runs the renamed pipeline -- and
`build_server_docker_default.sh`, which requires every test to run in the
default image, fails it. Main has the same script, the same packs and the
same gate, so this is not the lite tree's doing; it is fixed by publishing
the packs again from `configs/add-ons`, which is a release step.

### 2.13 A process that throws does not end the pipeline

When a process raises out of its step, `thread_per_process_scheduler`
logs `Process '<name>' threw an exception: ...` and the pipeline then
neither finishes nor exits: `viame` stays up with nothing running until
something kills it. Seen twice. In the Docker web image, `filter_enhance.pipe`
hit a `cv2` that could not load `libGL.so.1` and ran until the CRITICAL
harness's timeout, so a broken package showed up as a hang rather than a
failure. And on this machine's install, the same pipeline with a stand-in
`cv2` that raises `ImportError` was still up after 90 s; with the real `cv2`
it finishes in 13.

Whether main behaves the same is not settled: its enhancer did not import
the stand-in, so the comparison run there proved nothing.

**Fixed.** The thread that throws leaves, and the threads beside it are
blocked inside a step on an edge -- waiting for data the failed process will
never send, or for space it will never make -- so they never see the error
flag, and `_wait()` joins them forever. Edges now have `interrupt()`: it wakes
every waiter, after which a push returns without waiting and a get or peek on
an empty edge throws `edge_interrupted`. `thread_per_process` collects the
pipeline's edges at start and interrupts them all on the first exception, so
every thread returns to its loop, sees the flag and exits, and `_wait()`
rethrows the original exception. `pythread_per_process` already avoided the
hang its own way, by daemonising its threads and not joining once a process
has failed.

`scheduler.thread_per_process_stops_when_a_process_throws` runs
`numbers -> failing_process -> print_number` and requires `wait()` to throw the
process's exception within 30 s: with the fix it does in 6 ms; with the two
`interrupt_edges()` calls taken out it is still blocked after 30 s. The
installed pipeline with the stand-in `cv2` now exits 255 in 4 s. Stopping a
pipeline on request (`stop()`) still only sets the flag, and could block the
same way; nothing exercises it.

### 2.14 The forks were built without the environment that compiles their code

`viame_python_forks.cmake` handed each fork build one environment variable,
`PYTHONUSERBASE`. The superbuild's `add_project_pytorch.cmake` set a dozen
for every one of them, and three forks depend on those to compile anything:

- mmcv reads `MMCV_WITH_OPS`, which defaults to 0. Unset, it builds as the
  pure-python `mmcv` wheel -- `mmcv-1.7.1-py2.py3-none-any.whl` in every
  Docker image build -- with no `mmcv._ext`, and the CRITICAL netharn
  training test fails on importing it. Its CUDA ops also need `FORCE_CUDA=1`
  where no GPU is visible, as in `docker build`.
- sam2 tries its CUDA extension by default but `SAM2_BUILD_ALLOW_ERRORS`
  also defaults to 1, so without an architecture list to build for it
  printed an error and installed without `sam2._C`. The reference install
  has it; the image did not.
- mmdeploy's ORT custom ops are a separate CMake build
  (`MMDEPLOY_TARGET_BACKENDS=ort`, `ONNXRUNTIME_DIR`) that the superbuild ran
  before the wheel. The fork step fetches onnxruntime for it and never runs
  that build, so `libmmdeploy_onnxruntime_ops.so` is not installed. Nothing
  in VIAME imports mmdeploy, so this one is recorded and not restored.

The fork builds now get `FORCE_CUDA`, `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`
from `CUDA_ARCHITECTURES`, and `-allow-unsupported-compiler` for nvcc when
CUDA is on; mmcv gets `MMCV_WITH_OPS=1`; and sam2 builds its extension with
`SAM2_BUILD_ALLOW_ERRORS=0`, so a failure stops the build instead of being
dropped.

That made a second sam2 problem visible. `packages/patches/sam2` is a Windows
fix, which the superbuild copied over sam2 only on Windows; the fork step
copied every patch directory on every platform. Its `setup.py` passes
`CUDAExtension(..., library_dirs=None)` when it found no Windows python
library directory, torch then adds a list to that, and on Linux sam2 fails
preparing its metadata -- in the image, and here with the patch laid over a
copy, while the same copy unpatched builds `sam2/_C.so`. The patch is copied
on Windows only again. It is the only fork patch the superbuild made
conditional.

### 2.15 mmcv's ops compile one file at a time

The image build spends over an hour in mmcv. torch's `BuildExtension` uses
ninja when it can find one and falls back to distutils, one translation unit
at a time, when it cannot; no ninja is installed in the image -- not by
`install_deps_apt`, which is the same list main uses, and no lock pins the
`ninja` wheel. Watched in the build container: one `cicc` at 100% of one
core, 61 objects in 38 minutes, load average 1.2 on a machine building with
eight jobs. mmcv's CUDA sources times six architectures is the whole cost.

Main's images were built the same way, so this is inherited rather than new,
and it is not correctness: the wheel that comes out has the ops. Installing
ninja -- `ninja-build` in `install_deps_apt`, or the `ninja` wheel in the
locks -- should cut it to the machine's job count, and is worth measuring
against a build where nothing else changed.

### 2.16 What the namespace rename touched that a rename should not have

P11-T01 renamed `kwiver::vital` to `viame`, `kwiver::arrows::<a>` to
`viame::<a>` and `sprokit` to `viame::pipeline`, in 1216 files. Three things
in the tree turned out not to be namespace text at all, and each broke the
build in its own way:

- **A qualified name split across a macro's line continuation.**
  `KWIVER_UNIQUE_PTR` in `pluggable_macro_magic.h` wrote `&kwiver::` on one
  line and `vital::` on the next. Rewriting a line at a time made
  `viame::viame::`, which compiles nowhere. It was the only one; the script
  now reports a file whose line ends in a namespace qualifier and a
  backslash rather than joining it silently.

- **Two implementations of the same functions, told apart only by their
  namespace.** `kwiver::vital::read_krtd_file` and VIAME's own
  `viame::read_krtd_file` were the same KRTD camera IO, imported once and
  written once, differing in whitespace. With both namespaces spelled
  `viame` they became one symbol defined twice, and the link said so. The
  imported copy, `library/algorithm_framework/io/camera_io.{h,cxx}`, had no
  in-tree consumer and is deleted; `library/file_io/camera_io.*`, which
  `camera_rig_io.cxx` and `calibrate_cameras_from_tracks_process.cxx` use,
  stays. The install manifest lost a header because of it.

- **An interface whose identity is its mangled C++ type name.**
  `embedded_pipeline_extension` registers under `typeid(T).name()` rather
  than a declared `interface_name()`, so the registry key
  `N6kwiver27embedded_pipeline_extensionE` became
  `N5viame27embedded_pipeline_extensionE` -- same algorithm, `test`, under a
  new key. `plugin_factory.h` says the accessor exists so that names do not
  depend on the mangling; this interface does not use it. Anything outside
  VIAME that looked up that key by string has to change, which is a thing
  the migration guide should say.

### 2.17 The python package root was one variable, not 152 call sites

`viame_add_python_module( path modpath module )` installs to
`${package}/${modpath}`, and `package` comes from `_viame_python_package`,
which reads `viame_python_package` if set and otherwise lowercases the
project name. Renaming `kwiver.*` to `viame.*` therefore did not touch the
152 call sites at all: three `set( kwiver_python_package "kwiver" )` lines
were deleted -- in `python/CMakeLists.txt` and the two `python.cmake` files
that had already moved beside their C++ -- and every module followed the
project name into `viame`. What the call sites did need was their `modpath`
shortened (`vital/types` to `types`), and that is an argument, not a root.

The hook is kept although nothing sets it. What it guards against is real: a
second copy of one extension module in an interpreter is a pybind11
duplicate-registration abort, not a warning.

### 2.18 A binding that moves beside its C++ collides with it by name

P11-T02b moved the bindings out of `python/kwiver/**` into the library
directories. Fifteen of them could not keep their names: `datum.cxx`,
`edge.cxx`, `pipeline.cxx`, `process.cxx`, `process_factory.cxx`,
`scheduler.cxx`, `scheduler_factory.cxx`, `stamp.cxx`, `utils.cxx` and
`version.cxx` are all names `library/pipeline_framework` already used for
the C++ they bind, as are `adapter_data_set.cxx` and
`embedded_pipeline.cxx` in `adapters/`, `camera_from_metadata.cxx` and
`metadata_io.cxx` in `io/`, and `kwiver_applet.cxx` in `applets/`. Each
took the `_python` suffix that `library/core_types` and
`library/algorithm_framework/algo` had already established for exactly this
reason.

`python_wrappers.cxx` is the opposite case and is named by no target: seven
bindings `#include` it rather than link it, so listing it would compile it a
second time as a translation unit of its own.

### 2.19 A stale install prefix makes a compatibility shim untestable

`make install` only ever adds. The prefix therefore still held the previous
generation of the python package -- 531 files under `site-packages/kwiver`,
88 of them `.so` -- after a build that installs two files there.

That is not merely untidy. The whole point of the `kwiver` shim is that
`import kwiver.vital.types` resolves to `viame.types` through an alias
finder; with the old tree present it resolves to a real leftover `.so`
instead, and the test passes while proving nothing. Worse, the leftover and
the new module are two copies of one pybind11 module in one interpreter.
The shim was only meaningful once the prefix was purged, and it is what
`kwiver.vital.types is viame.types` being `True` now actually measures.

The same residue reaches the baselines. `install.txt` is immune by
construction -- it is built from what the install step says it placed -- but
`pipes.json` walks the prefix, and the copy in the tree lists 28 pipelines
that no build installs and that three earlier commits deleted from the
source (`ebff6e158`, `42e3f2ec5`, `bab80119e`). It was recorded on a dirty
prefix.

### 2.20 A token rename has to fire inside a dotted path too

The module table and the identifier table in
`design/scripts/rename_python_modules.py` shared a lookbehind. `(?<![\w.])`
is right for the dotted table, where it stops a longer name that was already
rewritten from being rewritten again. On the identifier table it meant a
name preceded by a dot was skipped: `from viame.util import VitalPIL` was
rewritten and `from viame.util.VitalPIL import get_pil_image` was not.

Eight files were left importing a module that no longer existed. It did not
fail at build time -- nothing imports a python module during a build -- and
surfaced as 65 `ERROR` lines in the registry dump, two algorithms that could
no longer be introspected, and 48 pipeline regressions, all of them one
`ModuleNotFoundError`. The identifier table now uses `(?<!\w)`.

### 2.21 `.py` beside `.h` means `__pycache__` beside `.h`

Three libraries install their headers as a tree
(`install( DIRECTORY ... FILES_MATCHING PATTERN "*.h" )`). That was safe
while python lived in `python/`. It is not once `library/algorithm_framework/util`
holds `pil.py`, `entrypoint.py` and `find_python_library.py`: importing one
writes a `__pycache__` into the **source** tree, and `FILES_MATCHING` still
creates the directory at the destination even though it matches no file in
it. `baseline:install` caught it as a single new path,
`include/viame/algorithm_framework/util/__pycache__`. All three rules now
exclude it, next to the `tests` exclusion that is there for the same reason.

### 2.22 The python entry points were declared by a side channel

`python/CMakeLists.txt` had an `install_egg_info` target that ran
`setup.py egg_info` and installed the result. That `.egg-info` was the only
declaration of VIAME's entry point groups anywhere, so deleting the
directory with the rest of `python/` took `viame.python_plugins` with it:
`PythonImpl` and `PythonTheyImpl` stopped registering, and the only in-tree
exercise of entry point discovery went quiet. Nothing failed to build, and
`compare_registry.py` reported it as two algorithms simply gone.

The step is restored as `packaging/CMakeLists.txt`. `setup.py` needed the
layout it describes corrected first -- it still named `python/kwiver` as the
package root -- and `packages` is now empty on purpose: CMake installs the
package, setuptools does not, and naming a package setuptools cannot find
fails the whole step after writing every other piece of metadata.

### 2.23 A baseline tool run by hand is not the test

Three of the regressions chased during P11-T02 were the invocation, not the
tree. `viame registry-dump` without `--introspect` cannot read the defaults
of a lazily declared python algorithm, so 25 entries reported as "can no
longer be introspected" that `baseline:registry` reads perfectly well.
`viame pipe-check --all` walks `$VIAME_INSTALL` from its working directory,
so running it from the wrong one reported 28 pipelines as gone while the
ctest entry passed in 12 seconds.

The commands the tests actually run are in `tests/baseline/CMakeLists.txt`,
and `tests/baseline/README.md` did not match them: it documented
`install_manifest.py <install> --record`, which omits the required
`--prefix` and passes the install directory where the log belongs. It is
corrected.

### 2.24 The logger is the one variable its own helper cannot serve

P11-T03 gave each renamed environment variable the same rule: read the name
VIAME uses now, fall back to the one it used before, and say so once.
`viame::get_env_renamed` and `viame::environment_path_renamed` are that rule
in C++, and they can warn through the ordinary logger because `viame_util`
links `viame_logger`.

The pair the logger itself reads cannot use them. `kwiver_logger.cxx`
resolves `VIAME_LOG_LEVEL` and `KWIVER_DEFAULT_LOG_LEVEL` while the logger is
being built, and `viame_logger` cannot link `viame_util` -- that is the
dependency the other way round, and closing it would be a cycle. Warning from
inside level resolution would also mean logging before the sinks it is
configuring exist.

So that one pair stays silent, which costs little: the setup scripts fold the
old name into the new one (`VIAME_LOG_LEVEL=${VIAME_LOG_LEVEL:-${KWIVER_DEFAULT_LOG_LEVEL:-info}}`)
before anything reads either, so a sourced environment never reaches the
fallback at all. The python side has no such constraint and does warn.

An empty new name is a value rather than an absence, in both languages.
`VIAME_PIPE_INCLUDE_PATH=` is how a caller says "no extra directories", and
falling through to the old name there would search the directories it was
being asked not to.

### 2.25 OpenCV 5 broke the trainer's evaluation, and the suite cannot see it

The DIVE smoke's training leg trains successfully and then dies in the
evaluation that follows it. Sixteen epochs, loss falling 2.037 to 0.082, a
model deployed to
`deep_training/fit/runs/netharn/mhcsjpnz/deploy_netharn_mhcsjpnz_012_BRVPJJ.zip`
-- and then:

    cv2.error: OpenCV(5.0.0) .../imgproc/src/drawing_text.cpp:1430:
    (-215:Assertion failed) img.depth() == CV_8U in function 'putText'

through `detect_eval` -> `detect_predict` ->
`kwimage/structs/detections.py:draw_on` -> `boxes.py:draw_on` ->
`im_draw.py:draw_text_on_image`. The lock carries
`opencv-contrib-python-headless==5.0.0.93` and `kwimage==0.11.6`: OpenCV 5
tightened `putText` to require an 8-bit image and kwimage hands it a float
one. The same defect appears 53 times during training as a caught
`In draw_batch ex = error(...)` warning, which is why it is survivable there
and fatal in eval, where nothing catches it.

**The test suite cannot catch this.**
`viame_examples:train_netharn_cfrnn_from_viame_csv` is a CRITICAL test and it
passes, but `TRAINING_TIMEOUT = 120` with `timeout_is_success=True`: what it
asserts is that training *starts*. Sixteen epochs took about seven minutes
here, so the run the test makes never reaches the evaluation step at all. A
build whose trainer trains and then cannot evaluate is indistinguishable,
from the suite's point of view, from a working one.

It is not phase 11's doing. The whole failing call chain is inside
`site-packages`, VIAME's own code is not on it, and the renames touch
neither kwimage nor OpenCV. It is recorded here because the smoke is the
only thing in the tree that runs training far enough to see it.

**The OpenCV 5 half is fixed, by carrying upstream's own fix.** kwimage `ebaba74`, "Fix OpenCV 5
drawing and vector compatibility", adds `_cv2_put_text_compat`: try the
native call, and on a `cv2.error` naming `img.depth() == CV_8U` for a
non-uint8 image, rasterise a binary uint8 text mask and assign the colour
through it, preserving dtype, NaNs, masked arrays and inplace behaviour.
That commit is on `main` and unreleased -- the newest tag and the newest
release on PyPI are both 0.11.6, which the lock pins -- so it is carried in
`packaging/patches/apply.py` until 0.12.0 ships rather than submitted again
as a PR.

Two things about writing that patch are worth keeping, because both were got
wrong first and neither was caught by any test:

* **Where the helper goes.** Expressed as one substitution on the call line,
  the `def` lands *inside* `draw_text_on_image` and truncates it: the
  function returns `None` and loses its `return_info` branch. Tier 1 passed
  449 of 449 with it broken, because nothing in the suite draws text. It is
  two substitutions -- one to place the helper at module scope, one to change
  the call.
* **Anchors must not survive their own substitution.** `apply_one` reports a
  patch as already applied only when the old text is *absent*, so an insert
  anchored on `def _text_sizes(` -- which the replacement still ends with --
  re-inserts the helper on every build. The file grew 53 lines a run. The
  replacement now reflows that signature so the anchor is gone afterwards,
  and three consecutive runs leave the file byte-identical.

Training now survives drawing: across a second smoke run the
`img.depth() == CV_8U` assertion went from 54 occurrences to 0, `cv2.error`
to 0, and the 53 caught `draw_batch` warnings to 0, with the patch verified
in place sixteen minutes before the run started. It also got further -- 18
epochs finished rather than 16, loss 0.0765 rather than 0.0876, a model
deployed either way.

**It still does not finish, because a second defect was behind the first.**
Evaluation now gets past drawing and dies in kwcoco instead:

    harn.on_complete() -> evaluator.evaluate() -> coco_eval._init()
      kwcoco/coco_evaluator.py:389  pred_coco = pred_extra['coco_dset']
    KeyError: 'coco_dset'

`_coerce_dets` returns `(gid_to_dets, extra)` and the line above it does the
same for the truth set successfully, so the predictions come back without
that key. kwcoco 0.9.0, netharn's vendored `detect_eval`, and nothing of
VIAME's on the path. Run 1 never reached this code at all -- it mentions
`coco_evaluator` zero times against run 2's three -- so this is a defect the
OpenCV crash was **masking**, not one anything here introduced. It is 2.26.

VIAME's own OpenCV 5 break was separate and smaller: `siammask_tracker.py`
tested `cv2.__version__[0] == '4'`, which is False on 5.x, and took the
OpenCV 2/3 three-value `findContours` path, which throws. It is `found[-2]`
now, the idiom `utilities_coco.py` already used, on both `main` and `lite`.
The 33 other `[:, 0, :]` indexings of OpenCV returns were checked against
cv2 5.0.0 and are all still correct: `findContours`, `undistortPoints` and
`convertPointsToHomogeneous` return the same `(N, 1, 2)` shapes they did.


### 2.26 kwcoco's evaluator wants a key the prediction loader does not set

Uncovered by fixing 2.25: with OpenCV 5 no longer killing the trainer's
drawing, evaluation runs on and fails in kwcoco instead.

    kwcoco/coco_evaluator.py:389  pred_coco = pred_extra['coco_dset']
    KeyError: 'coco_dset'

`CocoEvaluator._coerce_dets` returns `(gid_to_dets, extra)`, and line 388
reads `true_extra['coco_dset']` for the truth set without complaint, so it is
the prediction side specifically that comes back without the key. Pinned:
`kwcoco==0.9.0`, against netharn's vendored `detect_eval`, which VIAME ships
in `library/object_detectors/netharn/`. Nothing of VIAME's own is on the call
path.

Not investigated further here: it is a different package and a different
defect from the OpenCV work, and the netharn trainer produces and deploys a
model before it happens -- what fails is the evaluation report afterwards.

Like 2.25, the suite cannot see it. `viame_examples:train_netharn_cfrnn_from_viame_csv`
asserts that training *starts* (`TRAINING_TIMEOUT = 120`,
`timeout_is_success=True`), and evaluation is eighteen epochs and several
minutes past that.

### 2.27 GFIT runs on lite, and what moved is the dependency set, not the port

The add-on pipelines had never been run on `lite`. They are not covered by
`GOLDEN` (fixture-based, no add-on pipeline in it) and their seven `ctest`
cases under `PIPELINES` are all `DISABLED` -- not for want of models, which are
installed and byte-identical to `main`, but because `tests/pipelines/cases.py`
names no GFIT stem, so `discover()` falls through to
`Case(skip="no test case defined")`. `pipes.json` covers them statically: every
process resolves (`rf_detr`, `ocv_windowed`, `netharn`, `nms`). Nothing ran them.

Eleven of the thirteen GFIT pipes are runnable. The two `seagis` variants are
blocked on both branches alike by `SC6_camera3_2024.CamCAL` and
`SC6_satelliteA_2024.CamCAL`, absent from the pack; they are the only two pipes
in the install that mention `CamCAL`. The `viame` stereo pair runs to
completion on both branches and resolves 0 tracks on both, and the `_sing`
variants exit 255 on both -- `viame.measurement.dino` on lite,
`viame.pytorch.dino_matcher` on main, the same missing module under lite's new
name. None of that is a lite regression.

Four mono pipelines were run on each branch, each in its own working directory
-- one directory per run, because a detector and a tracker both declare
`detector_writer` and sharing a directory silently overwrites the first
result (2.23's lesson, applied). All eight exited 0.

**The control first.** `detector_gfit_groups_v3` run twice on lite is
bit-identical, so the pipeline is deterministic and a difference against `main`
is signal rather than GPU noise. That control is what makes the rest meaningful,
and it has to be run before, not after, the numbers are quoted.

**Row-positional diffing said all 88 rows changed. It was wrong.** Two frames
hold a different count (9 vs 10, 10 vs 9), and one extra detection shifts every
later row against the wrong partner: positional diffing reported a 1336 px box
delta and 41 of 88 species mismatches, both artefacts of mis-pairing. Matched by
IoU instead:

| | |
|---|---|
| matched pairs | 87 |
| lite-only / main-only | 1 / 1 |
| median box delta | 0.32 px |
| 90th percentile | 2.47 px |
| worst box delta | 49.03 px, one detection, one frame |
| max confidence delta | 0.0389 |
| top-1 species differs | 6 of 87 (groups), 4 of 87 (species) |

The two unmatched detections sit at 0.1016 and 0.1214 against the detector's
`:threshold 0.10` -- threshold crossings, not appearances and disappearances.
Every species flip is a near-tie (0.294/0.301, 0.667/0.664, 0.202/0.205). Both
trackers' `track_output` match with **0** unmatched and **0** species
differences, and `utility_link_detections_gfit_v3` on shared groundtruth input
is byte-identical. Quoting the 49 px maximum alone would misrepresent a
distribution whose median is a third of a pixel.

**Two candidate causes were tested and refuted.** The image reader: `lite(core)`
and `lite(vxl)` are bit-identical, and forcing `vxl` on both branches leaves the
difference exactly unchanged, so the P3 reader swap contributes nothing. The
detector wrapper: the installed `rf_detr_detector.py` is 415 lines on both sides
and differs in 4 lines, all import renames, with **0** residual lines once module
paths are normalised.

What is left is the dependency set: opencv **5.0.0.93** contrib-headless against
**4.9.0.80**, timm 1.0.29 against 1.0.24, transformers 5.17.0 against 5.16.1,
pillow 12.3.0 against 12.1.0, torchvision `0.27.0+cu126` against
`0.27.0+78839c2`. torch, numpy and rfdetr are identical. Only the opencv move is
a decision on the record; the rest is resolution drift -- `base.in` leaves timm
unpinned and carries the same `transformers>=5.16.0,<5.18.0` range that
`add_project_python_deps.cmake:199` carries on `main`, so the two builds resolved
the same constraints on different days.

**Why it matters:** "the add-on pipelines produce the same output" is now a
measured claim rather than an assumption, and the measurement says the port did
not move them. It also says the remaining difference will not close by fixing
VIAME code, because the difference is not in VIAME code.

**How to apply:** run the control before quoting a delta; match detections by
IoU before calling a row difference a regression; and read a distribution, not
its maximum. The install also gained a `core` image reader that `main` cannot
provide -- forcing `image_reader:type=core` on `main` fails outright, its
registered impls being `add_timestamp_from_filename`, `write_disparity_maps`,
`ocv`, `ffmpeg`, `vxl`, `tiled_multifile` -- so `core` is not a name that can be
used to compare the two branches.

### 2.28 A rename is two passes, and the second one is the names

Renaming `image_ops` to `image_kernels` rewrote 118 files and still broke the
configure, because a substitution over file *contents* does not touch file
*names*. `git mv library/image_ops library/image_kernels` moved the directory
and kept every basename inside it, so `test_image_ops.cxx` survived -- while
the same sed had already rewritten `viame_discover_gtests( viame image_ops )`
to `image_kernels`, and that macro derives its source from its argument:
`set( _SOURCES test_${NAME}.cxx )`. The result was

    Cannot find source file: test_image_kernels.cxx

for a file nothing had ever been asked to create. One `find . -name
'*image_ops*'` would have named it in advance, and it found exactly one.

The second miss was the file types. The substitution ran over an `--include`
list -- `*.cxx *.h *.txx *.py *.cmake CMakeLists.txt *.json *.md` -- chosen
by thinking about where code lives, and `design/lite-file-map.tsv` is none of
those. It still pointed at `library/image_ops/` while
`lite-library-layout.md`, rewritten by the same sed, pointed at
`image_kernels`, and `check_file_map.py` exists to compare exactly those two.
The check that catches this is `grep -rI` with no `--include` at all, over
the whole tree minus `.git`, `packages/` and `library/tpl/`: it found the
`.tsv` and confirmed no `.sh`, `.yml`, `Dockerfile` or `.in` was involved.

Both are the same shape. A rename has a contents pass and a names pass, and
an exclusion list written from intuition is not an inventory.

A third, smaller trap cost a wrong conclusion rather than a broken build.
`pgrep -x verify_rename.sh` reported nothing while the script was running:
Linux truncates `comm` to 15 characters and the name is 16, so the match can
never succeed. `verify_split.sh` is exactly 15 and had matched fine an hour
earlier, which is what made the negative look meaningful. `pgrep -f` found
all three pids. Finding 2.23 was the same lesson pointed at a baseline tool:
the measurement was wrong, not the thing measured.

### 2.29 What a merge into a restructured tree gets wrong, and where it hides it

Merging 67 upstream commits into `lite` broke the build three times, and not
once where the conflicts were. The 18 conflicted files were the easy part:
almost all were rename collisions, upstream editing a file under its old
`plugins/` identity against `lite`'s renamed module paths, and they announce
themselves. The failures came from the parts git handled silently.

**Rename detection places new files by similarity, and similarity is not
meaning.** Upstream added `disparity_segment.{cxx,h}` to `plugins/core`. Git
decided they most resembled `library/file_io/` and put them there -- stereo
disparity-segment fitting, in the CSV and KW18 reader library. Nothing
conflicts, nothing warns; the file simply lands in the wrong library and
compiles there quite happily if its dependencies happen to be satisfied. What
named the right home was reading what the code does and who calls it:
`fit_disparity_segment` is called from `measure_objects_process.cxx` and
`interactive_stereo.py`, both `library/measurement/`.

**A file moved between libraries carries its export identity with it.**
`disparity_segment.h` included `viame_core_export.h` and used
`VIAME_CORE_EXPORT`, because upstream it belonged to the `viame_core` target.
Moved into `library/measurement/` it must use `viame_measurement_export.h` and
`VIAME_MEASUREMENT_EXPORT`, and the include guard moves too. The namespace,
though, does **not**: `measurement_utilities.cxx` is itself inside
`namespace viame { namespace core {` and calls `fit_disparity_segment`
unqualified, so renaming the namespace to match the directory would have
broken it. Export identity follows the target; namespace follows the callers.
They are different questions and the directory answers neither.

**A clean merge is not a safe merge.** Three `vital/` -> `viame/` include
blocks were fixed by hand because they conflicted. A fourth,
`#include <vital/types/image_container.h>` in upstream's new pybind block,
merged with no conflict at all -- upstream added it in a region `lite` had
not touched -- and broke the build after the first two fixes were in. Two
`from kwiver.vital.types import ...` lines came in the same way. The lesson is
to sweep for the whole class once, tree-wide, rather than discover it one
build at a time: `#include <vital/`, `<sprokit/`, `from kwiver.`, and the
renamed `viame.core.` / `viame.pytorch.` paths. Three of those five sweeps
found something; the compat shims under `library/compat/` and
`library/utilities/compat/` legitimately name the old paths and must be
excluded, not fixed.

**Upstream moving a helper can leave two of it.** `vital_config_update` moved
upstream from `plugins/pytorch/utilities.py` to `plugins/core/utils.py`. The
re-export merged cleanly into `library/object_detectors/base.py` while the
conflict resolution kept `lite`'s own definition below it -- an import
shadowed by a definition eleven lines of docstring longer, plus a third copy
arriving in `library/utilities/utils.py`. Diffing the bodies properly showed
the code identical and only the docstrings different; a first, cruder
extraction had reported them as differing, which would have argued for keeping
both. The keeper is the one lowest in the DAG.

**The verification that mattered was not the build.** A green build says the
merge compiles, not that it delivered anything. `nm -DC libviame.so` showing
`viame::core::fit_disparity_segment`, the four new config keys present as
strings, and ten new tests running by name -- seven `viame:disparity_segment.*`
cases among them, none disabled -- is what says upstream's work actually
arrived. Finding 2.27's GFIT tests existed too, and every one of them was
`(Disabled)`.

### 2.30 An out-of-range value can be a meaning, and a skipped test is not a passing one

Porting `rectification_alpha` looked like a one-line interpolation, and the
interpolation was right. What was wrong was the guard around it:

    alpha = std::min( 1.0, std::max( 0.0, alpha ) );

Clamping looks defensive. It is, for the top of the range -- cv2 gives the
same answer at 2.0 as at 1.0. At the bottom it is destructive, because a
negative alpha is not a small alpha: OpenCV reads it as "default scaling" and
skips the rescaling step entirely. Measured on cv2 5.0.0, alpha -1 and -0.5
both leave the rectified focal at the pre-scale value **exactly** -- 605.0 for
a rig whose focal pair is 600 and 610, 1099.510502 for 1101.74 and 1097.28 --
while alpha 0 gives 2322.09 for the same rig. Clamping -1 to 0 is not a
rounding error, it is the opposite end of the range.

It mattered because -1 is not exotic. `interactive_stereo.py` ships
`rectification_alpha: -1.0` in its dense-grid defaults, so the clamp would
have rezoomed the rectification of every interactive stereo session, quietly,
with no error anywhere.

**The golden test I wrote for the feature did not catch it.** I recorded alpha
in `{0.0, 0.5, 1.0}` -- the range I had reasoned about -- and all three passed.
What caught it was upstream's `test_dense_stereo_grid.py`, whose fixture
happens to use `rectification_alpha: -1.0`, and which had been **skipping**
itself in `lite` for the whole of its existence:

    if not hasattr(measurement, 'DenseStereoGrid'):
        pytest.skip('built without OpenCV', allow_module_level=True)

Unguarding the class put that test into service for the first time, and it
failed on its first run. A test that skips is not a test that passes, and the
ctest line looks identical either way.

That is three times in one session. GFIT's seven pipeline tests were
`(Disabled)` for want of a fixture (2.27). `test_dense_stereo_grid.py` skipped
on a missing class. And
`segment_refinement_uses_rectification_and_triangulation` was inside
`#ifdef VIAME_ENABLE_OPENCV`, so it never compiled -- which is why an
`ASSERT_TRUE` on a function that returned `false` unconditionally had never
once failed. Each of the three was hiding something real.

Two habits follow. Measure a foreign API's semantics instead of recalling
them: the linearity, the invariant principal point and the negative-alpha rule
were all established by running cv2 and reading the numbers, and the check
that the alpha=-1 focal equals the mean of the input focal pair -- for all
four rigs, both baseline orientations -- confirmed the port before it was
built. And when enabling an implementation, check whether its test is guarded
too: here the implementation and its test were behind the same `#ifdef`, so
removing one without the other would have produced a feature that works and a
suite that still says nothing about it.

### 2.31 `pipe-check` says "resolved" for an implementation nothing registers

`utility_register_frames` failed on this branch and ran on `main`. The cause
was not a removed algorithm: it was a **rename to a name that never existed**.
P3-T08's `rename_impls.py` rewrote `estimate_homography`'s `vxl` to `core` in
the stabilizer and frame-registration configs, but this branch implements that
interface once, as python, registered `ocv`. Nothing has ever answered to
`core`. The rewrite was also partial -- six `core` against three surviving
`vxl` across the seven configs, and neither name resolved -- so the three
`common_sea_lion_stabilizer_*` files, byte-identical to `main`'s and still
saying `vxl`, were broken here too.

`compute_ref_homography:core` sits on the line directly below
`homography_estimator:type` and *is* registered on both branches, which is
almost certainly what made `vxl -> core` look right.

**The baseline said this was fine.** `pipes.json` records

    "homography_estimator:type": { "impl": "core", "resolved": true }

for an implementation that does not exist. `pipe-check` reports what the
config *names*, not what the registry *has*, for a nested algorithm type, so
`baseline:pipes` passed throughout -- as did `baseline:registry`, which only
compares the set of registered names and cannot know a config asks for one
outside it. Every "registry and pipes compare clean" in this ledger is
therefore a weaker statement than it reads for nested algo keys: it means no
name that was there has gone, not that every name a pipeline asks for exists.
Closing that needs `pipe-check` to resolve each `:type` value against the
registry, which is a change to the applet, not to this branch's code.

**What actually caught it** was running the pipelines and comparing outputs
against `main` -- the thing 2.27 recommended and this ledger had not done at
scale until now. The static baseline cannot catch a name that is asked for and
never registered; only running it can.

Fixed by aliasing rather than migrating, so one pipeline file runs on both
branches: `vxl` and `core` now resolve here to the same
`EstimateHomographyOCV` that `ocv` does, the way `core_image_io` answers to
`core`, `vxl` and `ocv`. Safe because no shipped config sets an
estimator-specific key -- the `inlier_scale` and threshold keys beside those
lines belong to `compute_ref_homography:core`, a different algorithm.

The eight pipelines still do not *run* here, for an unrelated and recorded
reason: they ask for `ocv_SURF`, and every opencv-python wheel excludes the
non-free modules. Substituting `ocv_SIFT` -- registered on both branches, and
what the implementation's own error message recommends -- completes
`utility_register_frames` here with nine homographies against `main`'s nine.

## 2.32 A generated stub and a real file, installed to one path

`library/image_io/CMakeLists.txt` called `viame_create_python_init( image_io )`
and then `viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
image_io __init__ )`. Both end in an `install( FILES )` aimed at the same
destination, and the stub won: the package installed as `# Generated by VIAME`,
22 bytes, declaring nothing. `pil` and `image_viewer` were therefore absent
from the registry of every install built since the `image_io` split on
2026-09-17, and four `golden:replay` cases failed with

    Could not find factory where attr "plugin-name" is "pil"
    for interface type "image_io"

The mechanism is a timestamp race, not install ordering.
`viame_create_python_init` writes its stub at **configure** time; the glob's
copy of the real `__init__.py` is an `add_custom_command` that runs at
**build** time and only when the source is newer than the output. The stub is
always newer, so the copy never runs.

This is documented in `cmake/viame/viame-python.cmake`. `_viame_package_init`
exists for exactly this reason and says so -- "that is how it went wrong the
first time" -- and `viame_add_python_package` calls it. The split reintroduced
the bug by calling the low-level pair directly instead. `video_io` had the
same pairing and survived it only because its real `__init__.py` happened to
be newer than the stub; `file_io` called `viame_create_python_init` beside
`viame_add_python_package`, which is the same latent hazard with the guard
bypassed. All three now use `viame_add_python_package` alone.

**What it says about the ledger.** This ran green in Tier 1 for four days.
`baseline:registry` compares the set of *registered* names and both these are
python lazy declarations, which it does not enumerate; only `golden:replay`
exercises them, and Tier 2 had not been run since the split. 2.31 said a
static baseline cannot catch a name asked for and never registered. This is
the same hole seen from the other side: a name that was *supposed* to be
registered and silently was not. Tier 1 passing is not evidence that the
python plugins exist.

## 2.33 A smaller chessboard inside a chessboard, and which colour tells you

`findChessboardCorners` was the last of the four estimators
`lite-opencv-removal.md` phase 5 names. Reproducing *where* it puts the
corners turned out to be the easy half; reproducing *when it refuses* was the
part with a real defect behind it.

The detector traces the squares of one colour, works out how they tile, and
reads the inner corners off that tiling. Ask a seven by five board for seven
by five and it is found. Ask the same board for six by five and there are
genuinely two ways to lay one out inside it, both complete, both made of real
squares with real squares all around them. OpenCV refuses all of them. A first
cut here accepted them, and that is not a cosmetic difference: nothing
downstream can tell. The caller pairs whatever comes back against a fixed set
of object points and calibrates, and a sub-grid produces a confident
calibration of the wrong rig with a small, plausible residual.

Requiring the block to be *maximal* within its own lattice -- no complete
further row or column against any side -- removes most of them. It does not
remove `(5, 3)`, and the reason is worth keeping:

    block   0 dark :  24 quads,  35 lattice corners   5x3=-     7x5=YES
    block   0 light:  12 quads,  15 lattice corners   5x3=YES   7x5=-

The light squares of a board touch the margin around it, so border following
traces them and the margin as one blob and the area filter drops it. The light
pass therefore sees only the **interior** light squares -- twelve of them --
and those form a complete, maximal five by three lattice of their own. Every
test local to that lattice passes, because as far as it goes it is a real
board. Only the rest of the picture disagrees.

So the rest of the picture is what has to be asked, and the naive form of the
question is also wrong. Stepping one place beyond an edge and looking for a
traced corner rejects *every* board, the true one included: one step beyond a
real board is its outer rim, and a rim is made of corners too.

What separates them is colour. Inside a board, every corner is where two dark
squares and two light ones meet, so both passes have a quadrilateral touching
it. On the rim only one colour is present, the other side being the margin. An
edge with both colours the whole way past it has more board beyond; an edge
without has ended. That test agrees with `cv::findChessboardCorners` on all
ten grids tried against the recorded fixture, where maximality alone left one
disagreement.

The general shape of this is the one worth carrying: a detector for a
*physical object* has to reject as well as locate, and a replacement validated
only on images that contain the thing will reproduce the locating and lose the
rejecting. There is no test for it in a recording of successful detections,
which is why `wrong_grid` -- a recording of finding nothing -- is in
`measurement_cases.py`, and why `unit:utilities:chessboard` asks for four
sub-grids by name and demands the whole board in the same breath. Refusing
everything passes the first of those perfectly.

## 2.34 A blob is a hole, half the time

`cv::SimpleBlobDetector` is the last OpenCV call in the calibration target
detector, and porting it turned on a detail of `findContours` that is easy to
read past: the detector calls it with **`RETR_LIST`**, not `RETR_EXTERNAL`.

The thresholding is `THRESH_BINARY`, so at each level the marked region is
what is **lighter** than the level, and the contours traced are that region's
borders. For a light blob on a dark ground that is the blob's own outline and
the distinction does not arise. For a **dark** blob on a light ground there is
no outline to trace: the light region is the whole background, its outer
border is the image, and the blob appears only as one of that region's
**holes**. With outer borders alone a dark blob is invisible.

Both are real inputs. The first cut here traced outer borders only, which is
what `image_kernels.find_contours` gives, and measured exactly right on the
dot grid -- 35 of 35, centres identical to cv2's to 0.0000 px -- and found
nothing at all on the chessboard, where cv2 finds twelve. Twelve is what the
golden records, because `ocv_detect_calibration_targets` is run over both
fixtures under the `dots` variant, and a dot detector pointed at a chessboard
finds its dark squares. Without that second input in the recording the gap
would have shipped, and it would have shipped as "dot targets of one polarity
silently find nothing".

The fix was already written. `contours.h` has `find_borders`, Suzuki and Abe's
border following proper -- every border, outer and hole, in OpenCV's own
order, which the file's own comment calls `RETR_CCOMP`. It had never been
bound to python because no python caller had wanted a hole before. Binding it
took twenty lines and the detector then matches cv2 on every case tried:
0.0000 px on the dot grid, the chessboard, eight drawn discs at three noise
levels, and nothing on a blank frame.

Two lessons, and the second is the one worth carrying. A port that measures
perfectly on the input in front of it can still be wrong about a whole class
of input -- the dot grid agreed to the last decimal while the detector was
missing half of what it is for. And "the C++ already does this" is worth
checking before writing anything: the capability was in the tree, one
`m.def` away, and the same was true of the float HSV conversions a commit
earlier. Both times the gap was in the binding, not the algorithm.

## 2.35 CLAHE turns a rounding difference into a visible one

`ocv_color_correction` was ported off cv2 and the port is **not** in the tree.
It is worth writing down why, because the reason is a property of the pipeline
rather than a mistake in the port, and the same shape will come up again.

Seven kernels the filter needs -- `erode`, `dilate`, `gaussian_blur`,
`box_blur`, `normalize`, `add_weighted`, `clahe` -- were bound for uint8 alone
although the C++ behind each is a template. That half is in the tree and is
the third instance of 2.34's pattern. Against cv2 on float32: erode, dilate
and add_weighted exact; gaussian_blur 3.1e-5, box_blur 1.6e-5, normalize
6.0e-8. `clahe` stays integer-only, and correctly: it static_asserts on an
integer pixel because a histogram needs discrete levels to bin into, which is
why `cv2.createCLAHE` takes 8 and 16 bit and nothing else.

With those bound, the filter ports cleanly and **thirty-seven of its
thirty-nine recorded images come out bit identical**. The two that do not are
`underwater_fusion` and `readme_example`, the two variants that run CLAHE in
L*a*b*, and they come out at **max 16, mean 1.38, with 68% of pixels moved**.

The cause is not the colour conversion. Measured separately: `to_lab` is
within 2 of OpenCV and `from_lab` within 2, and on a grey ramp the lightness
differs on 40 of 256 levels, always by one. OpenCV converts through a
fixed-point cube-root table where `image_kernels` works in double, and the two
round apart on the last count -- which `TOLERANCES` in `test_golden.py`
already records for `ocv_convert_color` at (1.0, 0.5), with the deliberate
note that reproducing the tables was considered in P7-T03 and not done because
a port more accurate than what it replaces is the better one to keep.

**CLAHE is what turns that one count into sixteen.** It is a histogram remap:
a pixel whose lightness lands one count on the other side of a bin boundary is
redistributed by a different mapping, and the difference then comes back
through `Lab2RGB` and through two `addWeighted` fusions. Measured at each
step: L differs by at most 1 before CLAHE and by **13** after it.

So the port cannot be made exact without reversing the P7-T03 decision and
reproducing OpenCV's fixed-point tables, and it cannot be accepted at a
tolerance without setting one four times the loosest max in the table
(warp/ocv at 4.0) and three times the loosest mean -- which `TOLERANCES` keys
per *implementation*, so it would blanket the eleven variants that are
currently exact, on a group whose only contract is the recording and which has
no ground truth behind it. Neither is a call to make in passing, so the filter
stays on cv2 and this note stays here.

The general point: a difference of one count is not always a difference of one
count downstream. Anything that bins, thresholds, sorts or indexes on a pixel
value -- a histogram, a watershed queue, a label -- can turn the last bit into
a visible amount, and "within one of OpenCV" is only a useful statement about
a step whose output is read as a number rather than used as a key.

## 2.36 OpenCV moved its greyscale by one bit, and our copy stayed

`image_kernels`' `rgb_to_gray` was written to be OpenCV's greyscale exactly,
and it was:

    gray = ( R*4899 + G*9617 + B*1868 + (1 << 13) ) >> 14

BT.601 at fourteen fractional bits, which is what `modules/imgproc` used
through OpenCV 4. The build ships OpenCV **5.0**, which uses fifteen:

    gray = ( R*9798 + G*19235 + B*3735 + (1 << 14) ) >> 15

and the two are not the same conversion. The extra bit does not simply double
the weights -- green gains a count and blue loses one -- so the answers differ
by one grey level on about **a quarter of a percent** of colours.

It had been invisible for two reasons, both of which are worth noticing.

The first is that the golden that checks it, `color.rgb_to_gray_matches_opencv`,
was recorded with a tolerance of **1**, which is exactly the size of the gap.
A tolerance wide enough to absorb a rounding difference is also wide enough to
absorb a version change, and the test kept passing while the thing it was
holding us to moved. A tolerance is a statement about precision; it is not a
statement that the implementation underneath is still the same one.

The second is that nothing downstream read a single grey level as a number.
The moment one did -- `ocv_optical_flow`, where the greyscale feeds a
polynomial fit and the result is scaled by 255/8 into a byte -- the quarter of
a percent became **5.9% of the output pixels moved, by up to three counts**,
with the flow itself reproduced to 7e-5 of a pixel. The port looked wrong and
the port was right; the conversion in front of it was a version out of date.

Finding the current weights took a fit rather than a reading: the shift and
the three coefficients were searched over a small range around BT.601, with
the constraint that they sum to `1 << shift` so that white stays white, and
scored against `cv2.cvtColor` on 22500 random colours. One candidate matched
on every pixel, at every shift that is a multiple of it -- 15, 16, 17 and 18
all describe the same conversion -- and fourteen does not. That is worth
keeping as a method: **when a constant has to agree with a library, fit it to
the library rather than to the paper**, because the library is the thing the
recordings were made with.

With the shift corrected the optical flow golden goes from 5.9% of pixels
moved to **one pixel in seventy thousand, by one count**, which is the byte
truncation catching a magnitude that sits within a millionth of a grey level
of the boundary, and is the tolerance `ocv_optical_flow` now carries.

## 2.37 Canny came back exactly; the circle transform's radius did not

`hough_circle` is the one detector left whose only cv2 call is an algorithm
rather than a primitive, and it is two algorithms: `cv::Canny` and
`cv::HoughCircles` under `HOUGH_GRADIENT`, which runs Canny itself. Both were
prototyped and measured before anything was written in C++, and the two
results are worth recording separately because they came out differently.

**Canny reproduced bit for bit, first try**, over a noise field and a disc, at
three threshold pairs, in both L1 and L2 gradient: not one pixel of 4800
differs from `cv2.Canny` in any of the twelve runs. The details that have to
be right are the 16-bit Sobel with a replicated border, the quantisation of
the gradient direction by comparing `|dy| << 15` against `|dx| * TG22` rather
than by an angle, and the two conditions on the push -- a candidate is only
seeded if the **previous column** was not seeded and the pixel above is not
already an edge, which is what keeps a thick ridge from seeding along its
whole length.

**The circle transform took four passes, and the last two came from making
cv2 show its work.** The accumulation was right from the start -- each edge
pixel votes along its own gradient, both ways, from `min_radius` to
`max_radius` in a 1/1024 fixed point -- and the centres were right at `dp = 1`
immediately. Everything after that was wrong in a way that took a way of
*observing* cv2 to fix, not more reading.

The observation is `maxRadius < 0`. Under `HOUGH_GRADIENT` that makes
`HoughCircles` **return its centres with radius zero**, before the radius
stage has filtered anything -- which is exactly the internal list an earlier
attempt gave up for want of. With it, two rules could be fitted rather than
guessed:

* the peak test is **asymmetric**: strictly greater than the left neighbour
  and the one above, greater *or equal* against the right and below. Sixteen
  combinations were scored and this one wins by a factor of eight;
* the sweep **skips the first cell of each axis and keeps the last**. OpenCV
  walks its padded accumulator from index 1 to one before the end, and since
  its padding is all at the far end, that asymmetry falls out. Getting this
  wrong the obvious way -- skipping both ends -- loses real centres.

With those two, the centre lists are **identical over 35 configurations**,
five scenes by seven `(dp, param2)` pairs, `dp` from 1.0 to 3.0, odd image
sizes included. The ranked order matches too, which means the accumulator
values agree and not merely their maxima.

The radius is a **histogram**, not the sorted-distance walk the old source
describes: ten bins per `dr`, swept from the top a window of ten bins at a
time, keeping the window whose count weighted against its own radius is best.
That is why every radius cv2 returns is a multiple of `dp/20`. The outer
loop's own decrement -- which fires *after* the inner window loop has already
walked the index back -- is worth 0.4 of a pixel on its own.

**What is left is the output order, and only the order.** On the recorded
fixture with the shipped configuration the port finds the same four circles
with the same four radii; `(47.5, 46.5, 12.7)` and `(21.5, 21.5, 8.4)` come
back the other way round. Across scenes the accepted **set** is identical
every time and the order differs in chunks -- the first few entries follow the
accumulator ranking and then it scrambles. That looked like a parallel
collection until it was tested: the order is stable across three runs at 1, 2,
4, 8 and 16 threads, so it is deterministic and simply a rule not yet found.
It matters because `min_dist` de-duplication keeps the first of a cluster, so
order decides which survives.

**And a second obstacle that has nothing to do with Hough.** The detector
blurs with `cv2.GaussianBlur(gray, (7,7), 1.5)` before the transform, and
`image_kernels.gaussian_blur` does not reproduce that. It reproduces it
*exactly* when sigma is zero -- OpenCV's small-kernel table is dyadic, 1/32
and 7/64, so both sides compute it without rounding -- but for an **explicit
sigma** it differs by one count on about **20% of pixels**, because
`cv::GaussianBlur` on an 8-bit image converts the kernel to fixed point and
filters in integers where this filters in double. Same family as the L*a*b*
tables and `ocv_convert_color`'s recorded tolerance. Fed cv2's own blurred
image the port's circle *set* is exact; fed its own, one circle moves a pixel.

So `hough_circle` needs two things, both now named: the ordering rule, and a
decision about the 8-bit fixed-point filter path. Neither is "the algorithm is
wrong", which is where this started.

## 2.38 SGBM: everything around the aggregation reproduced, the aggregation did not

The disparity cluster is the largest single prize left -- one algorithm,
`cv::StereoSGBM`, is the only cv2 call in `netharn/disparity.py` and
`tools/disparity.py` and the matcher behind `ocv_stereo_disparity` -- so it
was prototyped the same way the optical flow was. It did not come out the
same way, and the shape of the failure is worth having written down.

**What reproduced exactly, and how far that goes.** On a **one row** image
with `P1 = 1` and `P2 = 2` -- where the smoothness term is almost nothing and
the answer is essentially the cost's own minimum -- the port is identical to
`cv2.StereoSGBM` across every width tried, subpixel and all. That is a strong
statement about a lot of machinery at once:

* the Birchfield-Tomasi cost, including its two "channels" -- a Sobel-x
  clipped through a table of `min(max(k, -ftzero), ftzero) + ftzero` with
  `ftzero = max(preFilterCap, 15) | 1`, and the raw row -- and the half-sample
  extremes `min(v, (v+vleft)/2, (v+vright)/2)` on both sides, with the second
  channel's contribution shifted right by two;
* the first and last column of **both** channels being forced to the table's
  zero entry;
* the SAD window as a box sum whose borders replicate;
* the winner-take-all with its uniqueness test, the quadratic subpixel
  interpolation -- `denom2 = max(S[d-1] + S[d+1] - 2 S[d], 1)` and a **C
  truncating** division, not a floor -- the left-right consistency check
  against a second disparity map built in the same backward sweep, and the
  three by three median the whole thing finishes with.

**What did not.** Over a sweep of three heights, five widths, two block sizes
and three penalty pairs, 7.8% of pixels disagree, and where they disagree it
is usually by a sixteenth or two but sometimes by whole pixels, on the
neighbourhoods where the cost is ambiguous enough for a small difference in
the accumulated path to pick the other minimum.

Two statements about *where*, and the second one is a correction of the
first, which is why both are here:

* the first reading was that the aggregation drifts only once `P2` is much
  larger than `P1`. That came from testing one row, which exercises only the
  direction along the row -- the three that read the row above are reading a
  cleared buffer;
* measured again across heights: **row 0 is exact at `P1 = 1, P2 = 2` every
  time**, and the differences appear in the later rows. But not uniformly --
  four rows of one width is exact and three rows of the same width is not --
  so "it drifts with height" is as far as the evidence goes. With `P2` much
  larger than `P1` the first row disagrees too, by a sixteenth or so, which
  is a second and smaller thing.

Three things were ruled out along the way and are worth not re-testing: the
direction count, since removing the fifth backward direction makes it three
times worse; the disparity-edge convention, since padding the recursion's
`d-1` and `d+1` with `SHRT_MAX`, with the edge value or with `minLr + P2` all
give the same answer; and which of `min L'` or `min L' + P2` is subtracted,
since the difference is constant in `d` and cancels.

A two-column image disagrees as well and **no** rule in that family
reproduces cv2 on it at all, which says the very narrow case takes a
different path and is the wrong place to start. Start instead from two rows
of a width where one row is exact, where exactly one step of the vertical
recursion separates a right answer from a wrong one.

**One quirk found on the way, and it is OpenCV's rather than ours.** The
vertical half of the SAD box is a running sum, and the row it adds is guarded
by `if (k < height)` while the row it indexes is `min(k, height-1)`. The
clamp suggests the intent was to replicate the last row; the guard means the
whole update is skipped instead. So for the last `SADWindowSize/2` rows of
every image **the cost is not updated at all** -- those rows reuse the last
cost that was computed. A port that replicates the border, which is what the
clamp says, is wrong at the bottom of every disparity map.

Not landed. The prototype's value is this entry: the surrounding machinery is
settled and the remaining question is one four-term minimum wide.

## 2.39 The same performance bug three times, and it was never the arithmetic

Three separate kernels written in this phase turned out to be several times
slower than the OpenCV call they replaced, and all three times the first
explanation reached for was vectorisation -- OpenCV sweeps a window eight or
sixteen pixels at a time and these sweep it one. That explanation was wrong
every time, or rather it was the *last* few times' worth of the gap and not
the first. What the profile actually said:

* the Farneback pyramid's Gaussian: **1.9 s** of a 1080p frame pair;
* `min_eigen_value`, which is the whole of Shi-Tomasi corner detection:
  **0.219 s** of the 0.245 the detector took;
* building the Lucas-Kanade pyramid: **0.085 s** of the tracker's 0.111,
  which left the 21 by 21 window loops -- the part that looks expensive, and
  the part SIMD would help -- accounting for 0.03.

The common cause is `filter_2d` and the shape of code around it:

**A square pass over a separable kernel.** `separable_kernel` builds the outer
product and `filter_2d` sweeps it, so an N by N kernel costs N^2 weighted
samples a pixel where two passes cost 2N. At the seventeen taps the coarsest
pyramid level asks for, that is 289 against 34.

**A border rule resolved per tap.** `sample_with_border` switches on the
border mode on every tap of every pixel, and for all but a rim of the kernel's
own radius the answer is "inside". Eighteen dispatched calls a pixel for two
Sobel derivatives; some fifty million a frame pair for a four level pyramid.

Both are fixed now, in the general place rather than in each caller:
`filter_2d` splits the inside from the rim, and `separable_filter` does the
two passes for `gaussian_blur` and `box_blur`. A 1080p Gaussian at seventeen
taps went from **1.871 s to 0.146**, at seven taps from 0.355 to 0.069, and
the corner detector from 0.245 to 0.102.

Two things worth taking from it beyond the numbers.

The first is that **both fixes are arithmetic-preserving, and that was
checked rather than assumed.** Splitting the rim off visits the taps in the
same order, so it is exact by construction. Going separable is not: the square
pass multiplies the two weights together before touching the pixel and two
passes do not, and floating point multiplication is not associative. So it was
measured -- three image sizes, four kernel widths, five border modes including
`CONSTANT`, a three plane byte image and a float one -- and it is identical in
all sixty. The optimised tracker was checked the same way, not against cv2 but
against the **previous build of itself**, by checking the committed headers
back out and diffing the output: byte for byte the same over twenty-six
configurations.

The second is the ordering. Each of these was found by timing the parts, and
each time the part that was slow was not the part that looked slow. The window
loops in Lucas-Kanade are the obvious suspect, they are what OpenCV spends its
intrinsics on, and they were a quarter of the cost.

**And then "what is left is vectorisation" turned out to be wrong too**, which
is the last and best part of the lesson. Two things had never been controlled
for:

* cv2's build reports `Parallel framework: pthreads` and its tracker
  parallelises over points. On sixteen cores it takes 0.0066 s; held to one
  thread it takes **0.0211**. A factor of 3.2 of the remaining gap was cv2
  using the other fifteen cores, not cv2 using wider registers;
* cv2 dispatches to AVX2 and AVX512 at runtime where this tree builds for the
  x86-64 baseline -- and that makes almost no difference to *us*, because
  `-march=native` moves the corner measure by 9% and makes the tracker
  slower, and `-ffast-math` on top changes nothing. GCC is not vectorising
  these loops whatever it is permitted to use.

At equal optimisation and equal thread count the gap is about three times, and
the cheapest way to close it is the **parallel sweep, not the intrinsics** --
each point is independent of every other, so it stays deterministic.

**Threading it then found a fourth instance of the same bug**, which is why
this note keeps growing. Sixteen cores bought 1.17 times, because the pyramid
build was serial and most of the cost; timing inside it put `pyr_down` at the
top; and `pyr_down` was slow for the same reason as everything else here -- it
reached its pixels through `image_of::operator()`. That one is the most
instructive of the four, because the fix was not to write vector code but to
stop preventing it: the loop is integer, a fixed stride apart, with no
associativity question, so the compiler widened it by itself the moment the
accessor was gone. 0.0052 s to 0.0018.

Which sharpens the rule. **An accessor that costs three multiplications does
not just cost three multiplications** -- it hides the access pattern, and a
loop whose pattern the compiler cannot see is a loop it will not vectorise.
Every hand-written intrinsic considered for these kernels would have been
competing against auto-vectorisation that was being suppressed a line above. Which is
the ordering lesson twice over: the explanation that sounds most like real
engineering was, both times, the one that had not been measured. The rule that
would have saved all of it is to compare like with like first -- same `-O`,
same thread count, same instruction set -- and only then ask what the code is
doing.

## 2.40 A test that compared nothing to nothing, and passed

The `stereo_algos` port was checked the right way -- the old cv2 file was
pulled out of git, imported beside the new one, and both run over the same
synthetic frames, comparing detections. It reported **30 frames of 30
identical** and it was worthless: the shipped shape filter wants at least 800
pixels, an aspect ratio between 3.5 and 7.5, and twelve pixels of clearance
from the edge, and the synthetic blobs were about 120 pixels. Both
implementations returned nothing, and nothing equals nothing.

The tell was there to be read and nearly was not: a comparison that passes
*first time, everywhere*, on a port that swapped seven different OpenCV calls,
should not be believed. Printing the count of detections the **reference**
produced -- not just the agreement -- is one extra line, and it turned "30 of
30" into "0 detections compared".

With a target the filter actually accepts, the comparison found a real
difference on 12 of 20 frames, traced to the structuring element below. So the
bad test was not merely uninformative; it was hiding a genuine defect that the
good test found in one run.

The rule worth keeping: **an equivalence test has to show that the thing it
compares happened at all.** The test file now opens with a case whose only job
is to assert the sequence produces detections, so a future change to the
filter's defaults cannot quietly empty the other tests out.

## 2.41 `disk` is VXL's and `ellipse` is OpenCV's, and they differ on the size that matters

`image_kernels.morphology` took a shape named `disk`, and the obvious reading
of it -- that it is what `cv2.MORPH_ELLIPSE` gives -- is wrong. `disk_element`
says so in its own comment: it reproduces
`vil_structuring_element::set_to_disk`, because the pipelines it was written
for were replacing VXL. Two differences, and both bite:

* OpenCV **keeps an even size**. A request for 10 gives a 10 by 10 element
  whose anchor sits at (5, 5), one row and one column off centre. VXL's disk
  rounds down to a symmetric 9 by 9. The motion detector's default smoothing
  size is `(10, 10)`, so this is the ordinary path;
* the row extents come from `c * sqrt((r^2 - dy^2) / r^2)` passed through
  `saturate_cast<int>`, which **rounds**. Truncating instead -- again the
  obvious reading -- leaves the element one pixel narrow on most of its rows,
  which is a difference of four pixels of area at size 10 and showed up as
  tens of differing pixels on a real mask.

So `ellipse` is a new shape beside `disk` rather than a correction of it:
changing `disk` would move the VXL recordings that pin it. With `ellipse` the
element matches `cv2.getStructuringElement` at twelve sizes and an
open-then-dilate matches cv2 on twenty-four real masks, exactly.

The general point is about naming. A shape called `disk` and a shape called
`ellipse` sound like the same idea at different eccentricities; they are two
libraries' conventions, and a port that reaches for the one whose *name* fits
gets a subtly different answer. The docstring now says which is which.

## 2.42 The L*a*b* table is reproducible, and it is a float that decides it

P7-T03 declined to reproduce OpenCV's fixed-point colour tables, reasoning
that they are a precision compromise for speed and a port working in double
is the better of the two to keep. That reasoning is sound for a library and
wrong for this tree: the goldens for `ocv_enhancer` and `ocv_color_correction`
are recordings of cv2, compared at a tolerance of zero, so those two files
cannot come off cv2 until the conversion is identical rather than better.

`rgb_to_lab` is now identical, on **all 16777216 8-bit triples**. The domain
is small enough to check completely, so this is not a sample.

The shape of OpenCV's path is a gamma table on the input byte, a fixed-point
matrix, a cube-root table, and 12- and 15-bit rounding shifts. Reconstructing
it took four wrong answers, each informative:

* The **truncated constants** print in most references -- 0.008856, 7.787,
  0.137931 -- and OpenCV uses the rationals they came from, 216/24389,
  841/108 and 4/29. This changes nothing measurable, because the branch they
  guard covers 18 of 3072 entries, but it is the arithmetic OpenCV does.
* The cube-root table's **argument is a float**. Computed in double the
  conversion differs from cv2 on 1671 triples; the argument rounded to float
  first, on none in the prototype.
* The cube root itself is **not a correctly rounded one**. Two entries sit a
  ten-thousandth above a rounding tie -- 9454.500194 and 37088.500396 -- and
  OpenCV rounds them down where arithmetic rounds them up. Only index 49 is
  reachable, and that single entry moved 17645 triples. It is pinned as data,
  with the measurement in the comment, because deriving it would mean
  reproducing OpenCV's cube root.
* The prototype agreed with cv2 and the C++ did not, on the same inputs: the
  prototype had run under the install's **cv2 5.0.0** and the C++ check under
  the system's **4.12.0**, whose tables differ. The recordings name the
  install they were taken in, so 5.0.0 is the contract. A cross-version
  difference in a fixed-point table is worth knowing about: it means "matches
  OpenCV" is a statement about a version.

The inverse is exact too, and it took a different route to get there.

`cv::cvtColor`'s 8-bit Lab-to-RGB is **a separate implementation from its own
float path**, not that path rounded: cv2's float answer, rounded to a byte,
disagrees with cv2's 8-bit answer by a count on 2.8% of triples and by two on
0.001%. Reverse-engineering it the way the forward was done would have been
several hours, so OpenCV's source was read instead -- it is Apache-2.0 and
public, and reading the implementation of a dependency is cheaper than
inferring it. The prototype was then exact on all 16777216 triples first try.

The parts worth knowing, because none of them is guessable:

* One table per L holds **both y and f(y)**, built in float -- `i * 100 *
  16384` passes 2^24 before L reaches 3, so that rounding is part of the
  table, not an artifact.
* a and b are divided by 500 and 200 by **reciprocal multiplication**, and the
  b one carries a `+1` that is OpenCV's and not a transcription slip.
* f is inverted through a table spanning every value f(x) and f(z) can reach,
  biased by -8145 so a negative index works. Its integer divisions
  **truncate toward zero over a negative range**, so they are not floors.
* The sRGB transfer is a plain 4096-entry table whose entries are already
  0..255, so the gamma and the quantisation to a byte are one lookup.

The float path was reproduced first, before the source was read, and is worth
recording even though it turned out to be the wrong target: OpenCV's float
inverse gamma is a **natural cubic spline on 1024 knots**, matched to 1.8e-07,
with its last interval continuing linearly past the final knot. That is what a
float conversion needs, and it is not what the 8-bit one uses.

CLAHE went exact in the same sitting, and for a reason worth recording
separately: **two things had to change together.** OpenCV scales the
cumulative histogram in float, and computes the bilinear blend in float from
`x * (1/tileWidth) - 0.5f` grouped across-then-down; and it rounds with
`saturate_cast`, which is half to even, where this rounded half away from
zero. Either one left in double or rounded the other way and tens of pixels in
a frame come out a count off -- which is exactly the size of gap a tolerance
of 1 absorbs without anyone choosing to. 192 configurations now agree exactly,
over eight shapes including ones that do not divide by their tile grid, clip
limits from 0 to 40, and grids from 1 by 1 to 8 by 8.

That leaves the two colour files short of portable, but for a different
reason than before -- see 2.44. `ocv_enhancer` also still needs
`fastNlMeansDenoisingColored`, which does not exist here at all and is a real
algorithm rather than a table.

## 2.43 A recording of the replacement is not a recording of what it replaced

`ocv_convert_color`'s `rgb_to_lab` and `lab_to_rgb` goldens are, bit for bit,
what the real-valued formula produced -- not what OpenCV produced. They were
added by the commit that did the port, so they captured the port's own output.
With both conversions now identical to cv2 the recordings are the only thing
that disagrees, which is how it was noticed: six of the eight recorded pairs
went exact and these two went from passing to still passing at max 1. On the
recorded fixture the filter's output is identical to cv2 and the recording is
a count away from both.

The tolerance hid it. `(1.0, 0.5)` was justified in a comment describing a
difference in the opposite direction -- OpenCV's fixed point against our
double -- and a tolerance of 1 admits both stories, so nothing ever had to
choose between them. This is the same shape as `rgb_to_gray` being a version
behind under a tolerance of exactly 1.

Two things came out of it. The C++ recordings in
`tests/golden/image_kernels/opencv.json` **are** cv2's, and they print their
own margin; 34 of the 60 were never approaching their tolerance, so 28 came
down to what they actually achieve. With the inverse and CLAHE exact as well,
**48 of the 60 are now held to the byte where 23 were.** The five left are the
float geometry ones -- `remap_wave`, the two `warp_perspective`,
`warp_affine_rotate`, `match_ncc` -- where shaving the headroom off an
interpolating path buys a brittle test rather than a contract. `lab_to_rgb`
came down to exact once the integer inverse landed, making four of the colour
cases exact where none was.

The Python side needs a re-record, not a tightening, and that is left for a
decision: the value to record is already known to be right over the whole
input domain, but replacing a recording is a change to a contract.


## 2.44 Float accumulation order, which is not a table and cannot be tabulated

With both L*a*b* conversions and CLAHE exact, `ocv_color_correction` looked
clear: it uses no denoising, and every call left in it had a replacement. It is
not clear, and the reason is worth stating because it is a different *kind* of
obstacle from the tables and it applies to any port of this shape.

Three of its steps agree with cv2 only to within a few float32 ULP:

| step | agreement with cv2 |
| --- | --- |
| `GaussianBlur` on a float image, sigma given | max 4.6e-05 on values to 255 |
| `normalize` NORM_MINMAX on float32 | max 1.5e-05 |
| `exp` on float32 against `np.exp` | max 3.8e-06, on 40% of values |

The blur one is the instructive case. The obvious explanation is that this
accumulates in double where OpenCV accumulates in float, so a separable pass
was written **in float32, in tap order, with OpenCV's own kernel from
`getGaussianKernel`** and compared: it differs from `cv2.GaussianBlur` by the
same 4.6e-05. The gap is therefore not a precision choice that can be matched
by making ours less precise -- it is the *order* in which OpenCV's vectorised
filter accumulates its taps, which is a property of its SIMD structure rather
than of the filter. Nothing short of reproducing that structure reproduces the
number, and it is not something OpenCV guarantees across its own dispatch
paths.

Also worth noting while here: `normalize` on **uint16** is half a count out,
and `for_both_pixel_types` binds only uint8 and float, so the uint16 call is
going somewhere unintended rather than being refused. Worth a look
independently of this file.

None of this matters while the chain stays in float. It matters because the
chain ends in a byte: a difference of 4.6e-05 flips a rounded byte only when
the value sits that close to a boundary, which is rare -- and the recording is
compared at a tolerance of exactly zero, so rare is still a failure.

So the decision in front of `ocv_color_correction` is not an implementation
one. Either it carries a tolerance the way `ocv_optical_flow` already does --
`(1.0, 0.001)`, for this same class of reason -- or it stays on cv2. That is a
contract question, and it is recorded rather than answered here. What should
*not* happen is the port landing with a tolerance of 1 quietly attached, which
is how 2.43 came about.