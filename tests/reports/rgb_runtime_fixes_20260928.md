# RGB and runtime fixes — 2026-09-28

## Changes

Lite tools now retain RGB when exporting depth point colors and drawing registration contact sheets. Calibration documentation describes RGB inputs. The HSV byte conversion now reproduces main's SIMD body and scalar row-tail rounding.

Shared fixes, implemented first in an isolated main checkout:

- Cached calibration dimensions serialize as Python integers. JSON is written atomically, preserving an existing calibration if serialization fails.
- Training help/list commands return success.
- Two augmentation pipelines chain supported two-input mergers; the default linking pipeline uses the existing default tracker.
- Unfinished pipelines are excluded from installation/packaging. Wheel selection follows includes, embedded pipelines and training templates, preserves subdirectories, ignores comments, and rejects missing dependencies. Default adaptive training configuration and templates ship without bundling optional pretrained weights.
- The wheel launcher supplies its installation prefix for pipeline environment references. `run` uses `viame runner` directly.
- Sparse 3D reconstruction exports through COLMAP without requiring Open3D. The viewer propagates its child exit status and diagnoses a missing GUI executable.
- Main pipeline validation checks process construction and supports `--no-resolve`.
- Main's OpenCV container adapter checks the actual image's channel count before deciding whether to convert RGB to BGR. This fixes the drawing pipeline's background color swap.
- Multimodal registration is discoverable, declares valid homography ports, emits all outputs on failed alignment, and clips thermal normalization without byte wraparound or division by zero.
- Query shutdown sends completion on every output. Main's feedback preference score is exposed to Python.

Additional Lite fixes:

- Port main's current descriptor-request and query processes, result preference score, and Python adapter result conversion.
- Initial/zero/forward video seeks avoid building the timestamp index; closing resets the cursor.
- Exact Euclidean distance returns the OpenCV-compatible sentinel when the mask has no background.
- K-means uses bounded assignment workspaces, retains nearest distances during seeding, and avoids a duplicate final distance computation.

## Validation

An isolated test wheel, `viame==0.23.3+audit.rgb`, was built with the CMake install manifest and installed into a temporary Python 3.10 environment. Existing dependency packages were reused locally; this was not a fresh network dependency-resolution test.

| Check | Result |
| --- | --- |
| Advertised applet help/startup | 32/32 passed |
| Functional applet suite on installed wheel | 119 passed, 5 optional cases skipped |
| Packaged pipeline construction, excluding templates | 97/97 passed |
| Installed-wheel runtime regressions | 8 passed |
| Frame-reader, image utilities, clustering, multimodal tests | 89 passed |
| Native color tests, including recorded golden outputs | 17 passed |
| Main runtime/packaging regression tests | 19 passed |
| Main Python preference-score binding | Compiled and exercised |
| Cached stereo calibration export | Succeeded on main and Lite |
| Depth point cloud | Main and Lite PLY files byte-identical |
| Drawing and debayer/enhancement pipelines | Main fixed build and installed Lite wheel pixel-identical |
| GFIT groups v3, CPU, Tules video | 18 detection rows and 8 track rows exactly match the previously recorded main package run |
| K-means, 65,536 RGB samples, 16 centers, 10 iterations | 0.78 s → 0.31 s; identical labels, centers and compactness |

The query process was also exercised through an embedded pipeline: receive results through Python, send end-of-input, receive completion, and wait for shutdown. Both native branches were checked.

Pipeline construction is not full inference coverage. The GFIT check covers this video and CPU configuration; CUDA was unavailable. External GUI applications, real training jobs, and all optional models were not exercised. The published PyPI packages were not changed.

## Integration state

The original main checkout and both original Git metadata directories are read-only in this session. Lite source edits are present in the shared working directory. Commits were created in isolated checkouts under `/tmp/viame-runtime-fixes-20260928`; the original branch refs were not updated, merged or pushed.

- Main: `56eb422a7` in `main/`.
- Main KWIVER submodule: `199ff7bd6`, `c385a21c5` in `main/packages/kwiver/`.
- Lite: `6037afde9` in `lite/`.
- Transfer artifacts: `main-fixes.bundle`, `kwiver-fixes.bundle`, `lite-fixes.bundle`, and `lite-fixes.patch` in that temporary directory.

The other thread's requirement, dependency, and test-recorder work was left intact and excluded from these isolated commits. Detailed build logs, wheel, environment descriptions and output comparisons are in the same temporary directory. Earlier audit fixtures/results remain under `/tmp/viame-audit-20260928`.
