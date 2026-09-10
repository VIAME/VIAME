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

## 2. Open questions

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
