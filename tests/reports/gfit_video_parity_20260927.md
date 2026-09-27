# GFIT package video parity — 2026-09-27

All 24 pairwise CSV comparisons passed exactly across main, Lite CPU, and Lite
CUDA preprocessing. These are local test wheels of the current branches, not
an audit of releases already published on PyPI.

## Runs

Each tracker includes the GFIT detector and classifiers and writes both
`detections.csv` and `tracks.csv`. The same pipeline files, video bytes and
model files were used for every backend. Main selected CPU preprocessing;
Lite was tested with `cpu` and with `auto`, which selected CUDA for both motion
preparation and classifier preprocessing. Inference used CUDA throughout.

| Clip | Pipeline | Sampled frames | Detections | Track states | Tracks |
|---|---|---:|---:|---:|---:|
| seamap | `tracker_gfit_groups_v3.pipe` | 41 | 709 | 369 | 25 |
| seamap | `tracker_gfit_species_v3.pipe` | 41 | 709 | 369 | 25 |
| tules | `tracker_gfit_groups_v3.pipe` | 41 | 18 | 8 | 1 |
| tules | `tracker_gfit_species_v3.pipe` | 41 | 18 | 8 | 1 |

The clips are excerpts of `FishTrack23-Latest/Sample/SEFSC-SEAMAP-761901231-Cam2.mp4`
(first eight seconds requested) and `CDFW-2021-July-Tules1.mp4` (eight seconds
requested from 20 seconds). They were stream-copied with FFmpeg, including the
codec's delayed end frames. Each run passes the 30-frame motion warmup.

Comparison preserves row order and compares every CSV data field, including
track/detection IDs, timestamps, frame numbers, bounding boxes, confidence,
class names and class scores. Only comment lines containing export time and
execution metadata are ignored. Equality is at the CSV writer's precision;
this does not assert equality of unexported internal floating-point values.

## Bugs found and corrected

The first SEAMAP groups run produced 709 detections and 369 track states on
main, versus 697 and 367 on Lite. Lite CPU and CUDA already agreed with each
other. The differences came from the ported video reader:

- `vidl_ffmpeg` was an alias of the FFmpeg-arrow reader, although VXL uses
  different byte YUV-to-RGB arithmetic. Full-range YUVJ uses VXL's swscale
  RGB24 fallback instead. Both paths now reproduce the corresponding main
  behavior for these videos.
- Rounding timestamps instead of truncating them changed the frames selected
  at 5 Hz. VXL also assigns nominal successive timestamps to delayed frames
  drained at EOF. Lite now preserves both behaviors for its PyAV VIDL reader.

All 242 decoded frame timestamps match main for each excerpt. The corrected
outputs match in all 12 pipeline runs. No CUDA kernel or pipeline configuration
changes were needed for these fixes; the production changes are Lite-specific.

Regression coverage includes main-recorded synthetic video pixels, full-range
conversion, timestamp truncation at a sampling boundary, and decoder flushing.
The FFmpeg-arrow tests keep their own oracle rather than incorrectly requiring
VIDL to match that different reader. The reader test run passed 50 tests;
writer and throughput tests were excluded because those paths were unchanged.

## Environment and limits

Main source: `c0f004f40`. Lite native/package base: `ba2828b7c`, with the reader
corrections in this change. The test wheels were assembled from existing native
builds plus the current Python modules and installed into separate environments.
They were `0.23.3+parity.main` and `0.23.3+parity.lite4`.

Both used Python 3.10, Torch 2.12.0+cu126, torchvision 0.27.0+cu126, NumPy 2.0.2,
PyAV 17.1.0, kwimage 0.11.6, kwarray 0.7.2, kwcoco 0.9.0, and VIAME's
RF-DETR 1.8.0.dev0 fork. Third-party Python dependencies were shared from the
existing installation to hold their versions constant. The GPU was a Quadro
RTX 5000. Seeds were fixed and cuDNN benchmarking disabled.

VIAME and KWIVER native libraries/plugins came from the test wheels. Some main
codec/image dependencies (including libx264, libx265, libjpeg, libpng and
libswresample) resolved from the existing build installation. This comparison
therefore does **not** establish clean-machine wheel portability. It also does
not test Windows, every video format, entire source videos, or the no-PyAV
CLI fallback through the full GFIT pipeline. CLI color conversion has separate
regression checks; the end-to-end matrix used PyAV.

The machine-readable [results](gfit_video_parity_20260927.json) include model,
video and wheel hashes, all comparisons, selected backends and run durations.
Durations include model loading and shared-machine contention and should not
be treated as throughput benchmarks. Logs, wheels, inputs, output CSVs and
reproduction scripts are retained locally under `/tmp/gfit-video-parity/`.
