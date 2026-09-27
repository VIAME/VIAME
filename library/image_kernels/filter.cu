/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include "cuda_internal.h"
namespace viame {
namespace image_kernels {
namespace cuda {
namespace detail {
namespace {
__device__ float sample(float const *src, int x, int y, int c, int w, int h,
                        int channels) {
  return src[(std::size_t(reflect101(y, h)) * w + reflect101(x, w)) * channels +
             c];
}
__global__ void horizontal(float const *src, float *dst, int w, int h,
                           int channels, float const *tap, int n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(w) * h * channels)
    return;
  int c = i % channels, x = (i / channels) % w,
      y = i / (std::size_t(w) * channels);
  int half = n / 2;
  float sum;
  if (n == 3)
    sum = fmaf(sample(src, x, y, c, w, h, channels), tap[1],
               (sample(src, x - 1, y, c, w, h, channels) +
                sample(src, x + 1, y, c, w, h, channels)) *
                   tap[2]);
  else if (n == 5)
    sum = fmaf(sample(src, x + 2, y, c, w, h, channels) +
                   sample(src, x - 2, y, c, w, h, channels),
               tap[4],
               fmaf(sample(src, x, y, c, w, h, channels), tap[2],
                    (sample(src, x - 1, y, c, w, h, channels) +
                     sample(src, x + 1, y, c, w, h, channels)) *
                        tap[3]));
  else {
    sum = sample(src, x - half, y, c, w, h, channels) * tap[0];
    for (int k = 1; k < n; ++k)
      sum = fmaf(sample(src, x + k - half, y, c, w, h, channels), tap[k], sum);
  }
  dst[i] = sum;
}
__global__ void vertical(float const *src, float *dst, int w, int h,
                         int channels, float const *tap, int n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(w) * h * channels)
    return;
  int c = i % channels, x = (i / channels) % w,
      y = i / (std::size_t(w) * channels);
  int half = n / 2;
  float sum = fmaf(src[i], tap[half], 0.f);
  for (int k = 1; k <= half; ++k)
    sum = fmaf(sample(src, x, y + k, c, w, h, channels) +
                   sample(src, x, y - k, c, w, h, channels),
               tap[half + k], sum);
  dst[i] = sum;
}
} // namespace
void gaussian(float const *src, float *dst, float *scratch, int w, int h, int c,
              float const *tap, int n, cudaStream_t stream) {
  unsigned blocks = static_cast<unsigned>((std::size_t(w) * h * c + 255) / 256);
  horizontal<<<blocks, 256, 0, stream>>>(src, scratch, w, h, c, tap, n);
  check(cudaGetLastError());
  vertical<<<blocks, 256, 0, stream>>>(scratch, dst, w, h, c, tap, n);
  check(cudaGetLastError());
}
} // namespace detail
} // namespace cuda
} // namespace image_kernels
} // namespace viame
