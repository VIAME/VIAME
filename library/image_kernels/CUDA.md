# Optional CUDA image kernels

Enable with `-DVIAME_ENABLE_CUDA_KERNELS=ON`. The default is **OFF**.
This option is independent of the inference backends' `VIAME_ENABLE_CUDA`.
It requires a 64-bit host, CMake 3.18+, a CUDA compiler supporting C++17, and the CUDA runtime.
Set `CMAKE_CUDA_ARCHITECTURES` for the GPUs you distribute to, for example
`-DCMAKE_CUDA_ARCHITECTURES="75;86"`. No OpenCV, NPP or cuDNN is required.

`filter.cu`, `denoise.cu`, `temporal.cu` and `resample.cu` sit beside the CPU headers. They build into
`viame_image_kernels_cuda`, a separate shared library. Neither `libviame` nor
the ordinary Python image-kernel extension links CUDA. CPU calls and pipeline
configuration remain unchanged. A build with the option off never enables the
CUDA language or looks for the CUDA toolkit through this backend.

## Python

```python
from viame.image_kernels import cuda

if cuda.available():
    ctx = cuda.Context(device=0)
    image = ctx.upload(array)  # float32 or uint8, HxW or HxWxC
    blurred = ctx.gaussian_blur(image, 21, sigma=3)
    # Keep intermediates on the GPU; no transfer between these operations.
    blurred = ctx.gaussian_blur(blurred, 5, out=blurred)
    result = ctx.download(blurred)
else:
    print(cuda.availability_error())
```

For uint8 input use `ctx.denoise_non_local_means(image, strength=9, patch=7,
window=21)`. Gaussian currently accepts float32 only. Upload does not silently
cast unsupported dtypes. Noncontiguous NumPy arrays are packed on upload;
one-channel images download as HxW arrays, including inputs shaped HxWx1.

`out=image` reuses an existing allocation; it must match shape, dtype and device.
It also works with `upload`. Without `out`, returned images own independent
storage. Contexts cache filter coefficients, NLM weights and scratch allocations
up to the largest operation seen. Release the context to free that scratch;
release all references to an image to free its device allocation. Images can
outlive their context. No explicit close is needed.

Importing `cuda` is lazy. `available()` returns false if the extension, runtime
or device is missing, and `availability_error()` explains why. Explicit CUDA
operations report errors; they do not silently run on the CPU.

## C++

Link `viame::image_kernels_cuda` (also provided by the installed VIAME CMake
package), then include `<image_kernels/cuda.h>`. The public header needs no
CUDA toolkit headers.

```cpp
namespace gpu = viame::image_kernels::cuda;
gpu::context ctx;
auto image = ctx.allocate(width, height, 1, gpu::pixel_type::float32);
ctx.upload(image, host_pixels); // interleaved pixels; optional row stride in bytes
auto result = ctx.gaussian_blur(image, 21, 3.0);
ctx.download(result, output_pixels);
```

Each context owns a nonblocking CUDA stream. Calls synchronize that stream
before returning, including on exceptions, so host buffers need only remain
valid during the call. A mutex serializes calls on one context; Python releases
the GIL before taking it. Different contexts can run concurrently. Callers must
coordinate concurrent access when sharing writable images between contexts.
The API does not expose asynchronous operations. `image::device_data()` provides
a borrowed device pointer, and Python images provide `__cuda_array_interface__`
for `torch.as_tensor(image, device="cuda:0")`. Keep the image alive while using
a borrowed pointer. Synchronize external writes (for example,
`torch.cuda.synchronize()`) before passing the image back into this API;
external libraries may use different streams.

## Supported operations and numerical behavior

- Gaussian: float32, 1–4 channels, reflect-101 borders, odd size 1–255,
  finite nonnegative sigma (zero derives it). Shares CPU coefficients and
  follows the CPU float path's summation order, including explicit fused
  multiply-add operations. GPU tests allow `rtol=3e-6, atol=3e-7`.
- Non-local means: uint8, 1–3 channels, reflect-101 borders, finite nonnegative
  strength, patch/search sizes 1–63, forced odd as in the CPU API. Shares the
  CPU integer weight table and requires exact output equality. Separable
  distance sums avoid a patch-area loop per candidate; scratch use does not
  scale with the number of search displacements. This is the plain operation,
  not `denoise_colour` and its Lab conversion.
- Images must be nonempty, with at most `INT_MAX/8` elements. Unsupported
  types, sizes, devices and mismatched output buffers are rejected.

## Validation and performance

Validated on Linux with CUDA 12.6 and a Quadro RTX 5000 (compute capability 7.5).
The backend and extension also compile through the full VIAME CMake project;
CPU-only configuration succeeds without CUDA compiler discovery. Python tests
cover CPU parity, border/degenerate shapes, strides, in-place operations,
allocation reuse, ownership, parameter validation and concurrent calls. C++
tests exercise the public API and padded host rows. GPU tests skip when CUDA
is unavailable. The shared CPU helper extraction passed 117 before/after
comparisons with identical output. The combined Python suite passed 414 tests;
16 wheel tests and two C++ tests passed. CUDA Compute Sanitizer ran all 65
CUDA Python tests with zero memory errors. With devices hidden, the optional
import test passed and the 64 GPU cases skipped. An installed C++ consumer
built and ran without CUDA headers in its include path.

Run `tests/library/image_kernels/benchmark_cuda.py --repeats 10` from an installed build. It warms up
both paths, verifies results, and reports median wall time including stream
completion. The transfer-inclusive case reuses device allocations but includes
NumPy output allocation. Example measurements on this shared workstation:

| Operation | CPU | CUDA, images on device | CUDA, upload + compute + download |
|---|---:|---:|---:|
| Gaussian 1024×1024 float32, size 21 | 10.2 ms | 0.32 ms | 18.2 ms |
| NLM 256×256 uint8, patch 7/search 21 | 59.6 ms | 9.87 ms | 10.5 ms |

Transfers make Gaussian slower end to end here. This is why CUDA selection is
explicit and intermediate images can stay on-device. These timings depend on
hardware, driver and system load; measure the actual workload before selecting
CUDA. NLM still launches two kernels per displacement and could benefit from
launch batching for small images. Windows and CUDA 13 runtime execution have
not been tested in this change.

## GFIT tracking and classification

See [GFIT CUDA preprocessing](GFIT_CUDA.md) for the optional pipeline variants,
public motion/letterbox operations, parity checks and stage timings.
