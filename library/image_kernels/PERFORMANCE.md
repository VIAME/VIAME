# Kernel performance controls

## Worker budget and CPU dispatch

Set `VIAME_NUM_THREADS` before the first kernel call. The default is the smaller
of four and the machine's reported hardware concurrency; `1` disables internal
parallelism. Large Gaussian filters, SIFT descriptors, stereo stripes, denoising
stripes, and smoother passes share one process-wide worker pool. Small jobs and
nested calls execute on the calling thread. This budget limits internal workers;
it does not limit application threads or other libraries such as BLAS.

On POSIX, a child forked after the budget was initialized runs kernels serially
so it cannot wait for workers that existed only in its parent. A spawned process
initializes its own budget and pool.

AVX2/FMA Gaussian filtering and AVX2 stereo matching costs are selected at
runtime on supported x86 GCC/Clang builds. Other CPUs and compilers use the
portable path. Set `VIAME_DISABLE_SIMD=1` before the first call to force that
path for comparison. SIMD keeps the existing accumulation order, saturation,
and tie rules; the four stereo stripes stay fixed regardless of worker count.

## Reuse across frames

Python callers can retain scratch buffers explicitly:

```python
from viame import image_kernels as kernels

gaussian = kernels.GaussianWorkspace()
stereo = kernels.StereoWorkspace()

blurred = kernels.gaussian_blur(float_image, 21, 2.5, workspace=gaussian)
disparity = kernels.stereo_sgbm(
    left, right, num_disparities=128, mode="sgbm_3way", workspace=stereo)
```

Gaussian storage is reused for float32 filtering. Stereo storage retains rolling
cost rows, not an image-sized disparity cost volume. Workspaces resize for new
shapes and settings and invalidate image-dependent cached rows on every call.
Results own their memory and survive later calls. Python serializes concurrent
calls sharing a workspace; use separate workspaces for independent camera streams.
Deleting a workspace releases its retained buffers.

C++ callers pass `gaussian_workspace*` as the final argument to `gaussian_blur`,
or `stereo_workspace*` to `stereo_sgbm`. These C++ workspaces require external
synchronization if shared. Both arguments default to `nullptr`, which uses local
scratch. SIFT reuses Gaussian scratch across pyramid levels automatically.

## Local measurements

The after column contains five-run warm medians on the development machine,
using the default four-worker budget; the before column records the earlier
local baselines. These are representative measurements, not CI timing assertions.

| Operation | Before this change | After |
|---|---:|---:|
| Float Gaussian, 1024×1024, size 21 | 139 ms | 9.6 ms; 4.5 ms with workspace |
| SIFT, 512×512 | 888 ms | 173 ms |
| Three-way SGBM, 640×480, 128 disparities | 693 ms | 210 ms |
| Denoising, 256×256, patch 7/search 21 | 119 ms | 49 ms |

Stereo workspace reuse showed no meaningful latency improvement in this test;
it avoids repeated cost-buffer allocation. Concurrent stripe buffers increased
additional stereo peak RSS from the earlier roughly 3.25 MiB measurement to
8.5 MiB, still well below the former full-volume implementation's 396 MiB.
SIFT and stereo remain slower than OpenCV on the measured inputs.

Validation includes scalar/single-thread and SIMD/four-worker runs, bit-identical
before/after comparisons, workspace resizing, concurrent Python callers, nested
C++ calls, exception propagation, fork behavior, and address/undefined-behavior
sanitizer checks. The independent OpenCV comparisons skip when it is unavailable.
