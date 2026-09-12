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
  minimum (`area <= minarea`), and `image_ops::min_area_rect` kept the first
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
  golden fail and not enough to make anyone suspect it. `image_ops` does
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
* **`image_ops::resize` had OpenCV's sample grid but not its arithmetic**,
  and for an 8-bit image those are different answers. OpenCV resizes bytes in
  fixed point -- eleven-bit coefficients, and a vertical pass whose 8-bit
  specialisation shifts twice rather than casting once -- where `image_ops`
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
  `image_ops::resize_by_scale` is the one that takes a scale now, and all
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
  tree -- `viame.types` is legitimate -- but worth knowing when a VIAME
  script fails in the standard library for no reason.
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
not go in `image_ops`, which is the code that replaces OpenCV and so must not
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
* **An export set is all-or-nothing.** `viame_image_ops` is an INTERFACE
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

## 2. Open questions

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
