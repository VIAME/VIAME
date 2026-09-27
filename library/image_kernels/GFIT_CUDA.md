# GFIT CUDA preprocessing

Build with `-DVIAME_ENABLE_CUDA_KERNELS=ON` and install the GFIT add-on and its
models as usual. CUDA remains optional and off by default. Select one of:

- `detector_gfit_groups_v3_cuda.pipe`
- `detector_gfit_species_v3_cuda.pipe`
- `tracker_gfit_groups_v3_cuda.pipe`
- `tracker_gfit_species_v3_cuda.pipe`

These variants replace motion-image preparation and Netharn classifier chip
preprocessing with CUDA operations. The existing GFIT pipelines keep their CPU
preprocessing. RF-DETR and classifier inference already support CUDA; ByteTrack
association, Kalman filtering and track averaging continue on the CPU.

The motion filter uses device 0 by default. To use another GPU, change
`detector_grey:filter:gfit_motion_cuda:device` and the inference stages' existing
`xpu` settings together. Explicit CUDA selection reports an error when the
optional backend or device is unavailable.

## Shared C++ and Python operations

`context::gfit_motion(image)` / `ctx.gfit_motion(image)` fuse greyscale conversion,
5/30-frame variance, scaling and channel merging. Input is uint8 with 1–4
channels; output is uint8 RGB ordered `[variance5 * .5, grey, variance30 * .5]`.
One context holds one ordered sequence. Shape/channel changes reset history;
`reset_gfit_motion()` resets it explicitly. `out=` reuses output storage.
The kernel preserves the existing CPU temporal update and rounding behavior,
including its subtraction of the previous frame when the window is full.

`context::resize_letterbox(image, width, height)` / `ctx.resize_letterbox(...)`
accept uint8 images with 1–4 channels. They use area reduction, Lanczos4
enlargement, nearest-even embedded dimensions/offsets, and black padding to
match classifier preprocessing. Outputs own their device storage. Lanczos
coefficients derive from OpenCV's resize implementation; its license ships
with the optional backend. No OpenCV CUDA runtime is required.

The Netharn refiner's `preprocess_backend=cuda` uploads CPU chip views, resizes
on the GPU, and passes results into PyTorch through the CUDA array interface.
Normalization preserves the CPU float64 division followed by float32 conversion.
CUDA work runs in the calling process, outside DataLoader workers. The default
is `preprocess_backend=cpu`. See [CUDA.md](CUDA.md) for ownership and external
stream synchronization requirements.

## Validation and timing

Tests live under `tests/library/{image_kernels,classifiers,image_processing}`.
They compare motion sequences against the registered CPU filter chain and
letterbox pixels/tensors against existing classifier preprocessing. Set
`VIAME_GFIT_CLASSIFIER_MODEL` to a deployed GFIT classifier ZIP to also compare
model probability vectors; the groups EfficientNetV2-S checkpoint produced
bit-identical vectors for four test chips.

On a Quadro RTX 5000 with CUDA 12.6, 960×1728 RGB motion preparation measured:

| Path | Median time |
|---|---:|
| Existing CPU filter chain | 122.1 ms |
| CUDA with resident input/output | 0.388 ms |
| CUDA including upload and download | 15.8 ms |

Reproduce with `tests/library/image_kernels/benchmark_gfit_cuda.py` in an
installed build. Measurements include completed work after warmup and depend
on hardware and system load. The pipeline adapter still transfers motion images
back to the CPU for existing pipeline ports. Full tracking throughput and
Windows runtime execution have not been measured for this change.
