# VIAME lite: install size audit and reduction candidates

Measured 2026-09-08 on `build/install` of `main` @ 637e275ab (Linux,
CUDA build, VIVIA and PostgreSQL enabled, default model downloads).
Total: 14 GB. This is a working list; decisions are recorded in
STATUS.md when taken. Sizes are on-disk, symlinks not double counted.

## 1. Where the 14 GB goes

| Area | Size |
|---|---:|
| python site-packages | 8.4 GB |
| downloaded models under `configs/` | 3.4 GB |
| DIVE (Electron) | 0.8 GB |
| native libs and binaries (`lib`, `lib64`, `bin`) | 1.0 GB |
| `include/` | 0.2 GB |

### Native binaries by dependency

| Dependency | Size | Notes |
|---|---:|---|
| FFmpeg + x264/x265 | 267 MB | 242 MB static `.a`; runtime `.so` about 25 MB |
| Qt + VTK + vivia tools | 250 MB | qmake 38 MB, vpView 11 MB, viqui 12 MB, qgltf 8 MB, libxml2; only with VIVIA |
| kwiver | 111 MB | mvg 19, arrows/core 12, klv 12, vital_types 10; klv and most of mvg unused |
| OpenCV | 64 MB | imgproc 28, core 15 |
| VXL | 58 MB | 39 MB static, 17 MB `kwiver_algo_vxl.so` |
| viame | 34 MB | |
| darknet | 21 MB | |
| postgres server + cppdb | 10 MB | `bin/postgres` 8 MB should not ship |
| boost, gdal/proj/geotiff, sqlite, geographiclib/kml, png/jpeg/tiff, zlib, tinyxml, svm | 22 MB | svm and tinyxml 0.1 MB each |

All `.so` files checked are stripped. Total static archives in `lib/`: 434 MB.

### Python packages (largest)

| Package | Size | Why present |
|---|---:|---|
| nvidia/* | 3605 MB | torch wheel deps: cudnn 1005, cublas 573, cusparselt 431, nccl 377, cusparse 280, cufft 268, cusolver 232, nvshmem 195 |
| torch | 1628 MB | |
| triton | 697 MB | torch `Requires-Dist` on Linux; see §2 |
| onnxruntime-gpu | 494 MB | CUDA provider 416 MB; 27 pipelines use `onnx` |
| llvmlite + numba | 195 MB | transitive (ensemble_boxes, kwcoco, sam3, pandas optional) |
| pycolmap | 166 MB | COLMAP option |
| castxml | 147 MB | build-time only (kwiver python algo bindings) |
| mmcv | 145 MB | mmdet and netharn detectors |
| kwiver (python bindings) | 131 MB | 100+ separate `.so` modules |
| scipy (+libs) | 130 MB | |
| wandb | 86 MB | optional logger in trainers, unused |
| transformers | 80 MB | huggingface detector, sam3 text |
| pandas | 58 MB | |
| sympy | 49 MB | torch dependency |
| sklearn | 41 MB | 3 VIAME files |
| matplotlib + fontTools | 60 MB | plots, 16 VIAME files |
| cuda (bindings/toolkit python) | 36 MB | torch dependency |
| onnx | 35 MB | export |
| skimage | 35 MB | 6 VIAME files |
| detectron2, mmdeploy, decord, numba | ~100 MB | optional backends |

## 2. Triton

697 MB = `libtriton.so` 448 MB (bundled LLVM/MLIR), NVIDIA backend 210 MB
(`ptxas` 31 MB, `ptxas-blackwell` 40 MB, CUPTI/nvperf static libs 74 MB),
`libproton.so` 23 MB. Installed only because torch's Linux wheel requires it.

Users:

- `torch.compile` / inductor. VIAME never enables it: `sam3_utilities.py`
  passes `compile=False` at all four call sites, `rf_detr_detector.py` calls
  `optimize_for_inference(compile=False)`, litdet's `compile_model` defaults
  to False, sam2 `vos_optimized` is never requested, rf-detr training
  `model_config.compile` defaults to False.
- sam3 optional kernels: `perflib/nms.py` and `perflib/connected_components.py`
  catch the import error and fall back to CPU; `model/edt.py` guards the
  import. `train/loss/sigmoid_focal_loss.py` imports triton unconditionally:
  **verify whether `sam3_trainer.py` reaches it before dropping triton.**
- `ultralytics/utils/triton.py` is the Triton Inference Server client,
  unrelated.

Removal: omit from the lock file (installs are `--no-deps` from P1-T07).
Cost: `torch.compile` unavailable if a user opts in; sam3 video tracking
uses CPU NMS/connected components. Saves 697 MB.

## 3. Reduction candidates

Ordered by confidence, then size. "Now" means it can be done on `main`
without waiting for the lite port.

| # | Item | Saves | Confidence | When | Action |
|---|---|---:|---|---|---|
| 1 | Static `.a` archives in `lib/` | 434 MB | certain | now | Install blacklist for `*.a` (ffmpeg, vxl, libxml2, vv*); lite removes their sources anyway |
| 2 | triton | 697 MB | certain after the sam3 trainer check | now | Drop from python deps; install torch `--no-deps` |
| 3 | Qt, VTK, vivia binaries | 250 MB | certain | now | Only present with `VIAME_ENABLE_VIVIA`; lite drops VIVIA |
| 4 | `include/` | 219 MB | certain for binary releases | now | Skip in release packaging unless out-of-tree C++ plugin dev is supported (ties to open decision 10) |
| 5 | castxml | 147 MB | certain | now | Never install it; P8 removes the need |
| 6 | wandb | 86 MB | certain | now | Drop from deps; trainers only use it if configured |
| 7 | `bin/postgres`, sqlite CLI, GeographicLib/libtiff/Qt tools | ~15 MB | certain | now | The old `installation_blacklist.cmake` list, applied |
| 8 | numba + llvmlite | 195 MB | likely | now | No VIAME code imports them. Verify `ensemble_boxes` (used by detection fusion), `kwcoco`, `sam3` degrade gracefully without numba |
| 9 | pycolmap + open3d | 166 MB+ | certain | now | Move to the COLMAP extra (`requirements/colmap.in`); only `reconstruction.py` and `reconstruct_3d` need them |
| 10 | kwiver python bindings | ~120 MB | certain | P8 | One `viame._core` module instead of 100+ `.so` files |
| 11 | decord | 26 MB | likely | now | Only sam2/sam3/pytorchvideo video readers; VIAME feeds frames itself |
| 12 | detectron2, mmdeploy | 54 MB | per option | P9 | Already behind `VIAME_ENABLE_PYTORCH-*`; make sure their wheels are not in the base lock |
| 13 | mmcv + mmdet | 160 MB+ | medium term | later | Needed while netharn/mmdet models ship; goes when those models are ONNX |
| 14 | nccl + nvshmem | 572 MB | needs custom torch wheel | later | Multi-GPU/distributed only, but `libtorch_cuda` links them; requires building torch without `USE_NCCL`/nvshmem (infra exists: `VIAME_BUILD_PYTORCH_FROM_SOURCE`) |
| 15 | cusparselt | 431 MB | needs test | later | Sparse ops only; check whether torch 2.12 dlopens it lazily |
| 16 | onnxruntime-gpu CUDA provider | 416 MB | keep | - | 27 pipelines run ONNX on GPU |
| 17 | cudnn | 1005 MB | keep | - | Precompiled engines are the conv performance path |
| 18 | Models under `configs/` | 3351 MB | policy | now | Already shippable as add-on packs; base install could carry only the default fish and generic packs |
| 19 | DIVE | 835 MB | keep | - | Electron; not a dependency in this sense |

Sum of rows 1 to 9 (no code changes, after two verifications): about
2.2 GB, or 1.9 GB on builds that already have VIVIA off. Rows 14 and 15
could recover another 1 GB with a custom torch wheel.

## 4. Verification needed before acting

- sam3 training path and `sigmoid_focal_loss.py` (row 2).
- `ensemble_boxes`, `kwcoco`, `sam3` behaviour without numba (row 8).
- torch import with `nvidia-cusparselt` absent (row 15).
- Which pipelines and tools import `decord` (row 11).

## 5. Follow-ups for the lite tasks

- P1-T07: build the lock files without triton, wandb, castxml, decord;
  colmap and learn extras separate.
- P1-T08 / P10-T05: install blacklist for `*.a`, tool binaries, `include/`
  in release packaging.
- P8-T01/T02: single `viame._core` extension (row 10).
- P9-T01: consider a `torch` wheel built without NCCL/nvshmem for the
  desktop variant (rows 14, 15).
