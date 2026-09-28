# Main and Lite integration — 2026-09-28

## Published work

- Main integrates the runtime fixes from isolated commit `56eb422a7`, the pending initializer and patch notes, the two local class-ranking commits, and upstream main through `c490b6ec7`. Published main: `519412bba`.
- KWIVER's `viame/main` includes `199ff7bd6` and `c385a21c5`; the main gitlink points to the latter.
- DIVE uses upstream commit `e4efa3a7`, which contains the previously local DIVE pointer's work.
- Lite committed the shared RGB/runtime fixes without overwriting the other thread's `tests/reference` reorganization, then merged main.
- All 47 modified Python submodule files were verified against VIAME's patch overlays. The two Darknet documentation differences were incorporated into the stored overlay. The submodules were then restored to their recorded commits; reproducing their changes requires no new fork or private commit.
- The obsolete, clean pymotmetrics checkout and generated foundation-stereo egg-info remain on disk, locally excluded from Git status. Initial patches and untracked-file backups remain under `/tmp/viame-integration-20260928`.

## Merge resolutions

Main's new VLM service, Ollama algorithms, model-card utility, and tests now use Lite's library layout, native namespaces, lazy algorithm declarations, and test registration. Main-only KWIVER and retired plugin paths remain removed. Lite's runtime fixes and pipeline registry checks are retained.

Training evaluation uses `viame runner`. Sample-frame drawing uses Lite's image/video interfaces and image kernels, preserving green truth boxes and red computed boxes in RGB. It does not introduce an OpenCV C++ dependency. Drawing errors are reported without discarding a trained model.

Lite's dependency installation no longer patches installed PyPI packages. The kwplot fix is proposed upstream in https://github.com/Kitware/kwplot/pull/1. The kwimage OpenCV compatibility fix is already in upstream tag `v0.12.0`; no duplicate PR was opened. No dependency release pins were changed in this integration.

## Validation

| Check | Result |
| --- | --- |
| Main runtime and wheel regression suite, main runtime environment | 19 passed |
| Lite VLM, query service, Vertex AI, model-wrap, multimodal, runtime, wheel and fork-guard unit tests | 100 passed, 11 subtests passed |
| Lite pipeline/runner, video seeks, clustering and image utilities | 98 passed |
| Native Lite applet and model-card/class-ranking test builds | Passed |
| Native model-card tests | 4 passed |
| Native class-ranking tests | 3 passed |
| Compiled evaluation-drawing smoke test using merged helper code | Image/video reads, RGB colors, grayscale expansion, bounds and JPEG output passed |
| Patched-source cv2 guard | 2,203 runtime Python files across 6 initialized submodules passed |
| Rebuilt train applet help | Passed |

The Python source checks used the existing audit wheel's native modules with source package paths overlaid; they are not a fresh wheel-install test. The rebuilt applet and native tests used the existing Lite build. Full training, remote Ollama inference, GPU execution, and a fresh dependency resolution were not run. No PyPI package was published.
