/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include "cuda_internal.h"
namespace viame {
namespace image_kernels {
namespace cuda {
namespace detail {
namespace {
__device__ int sample(unsigned char const *src, int x, int y, int c, int w,
                      int h, int channels) {
  return src[(std::size_t(reflect101(y, h)) * w + reflect101(x, w)) * channels +
             c];
}
// Separable patch distances: O(patch) work rather than O(patch squared).
// Include the vertical halo here: reflecting an already computed distance
// would reflect the displacement too, which gives incorrect edge weights.
__global__ void distance_rows(unsigned char const *src, std::uint64_t *rows,
                              int w, int h, int c, int half, int dx, int dy) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(w) * (h + 2 * half))
    return;
  int x = i % w, y = i / w - half;
  std::uint64_t sum = 0;
  for (int k = -half; k <= half; ++k)
    for (int plane = 0; plane < c; ++plane) {
      int d = sample(src, x + k, y, plane, w, h, c) -
              sample(src, x + k + dx, y + dy, plane, w, h, c);
      sum += d * d;
    }
  rows[i] = sum;
}
__global__ void accumulate(unsigned char const *src, std::uint64_t const *rows,
                           std::int64_t *sums, int w, int h, int c, int half,
                           int dx, int dy, std::int64_t const *weights,
                           int shift, int levels) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(w) * h)
    return;
  int x = i % w, y = i / w;
  std::uint64_t distance = 0;
  for (int k = 0; k <= 2 * half; ++k)
    distance += rows[i + std::size_t(k) * w];
  auto level = distance >> shift;
  if (level >= static_cast<std::uint64_t>(levels))
    level = levels - 1;
  auto weight = weights[level];
  auto offset = i * (c + 1);
  sums[offset + c] += weight;
  for (int plane = 0; plane < c; ++plane)
    sums[offset + plane] +=
        weight * sample(src, x + dx, y + dy, plane, w, h, c);
}
__global__ void finish(unsigned char *dst, std::int64_t const *sums,
                       std::size_t pixels, int c) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels)
    return;
  auto offset = i * (c + 1);
  auto total = sums[offset + c];
  for (int plane = 0; plane < c; ++plane)
    dst[i * c + plane] =
        static_cast<unsigned char>((sums[offset + plane] + total / 2) / total);
}
} // namespace
void nlm(unsigned char const *src, unsigned char *dst, int w, int h, int c,
         int patch, int window, std::int64_t const *weights, int shift,
         int levels, std::uint64_t *rows, std::int64_t *sums,
         cudaStream_t stream) {
  auto pixels = std::size_t(w) * h;
  check(cudaMemsetAsync(sums, 0, pixels * (c + 1) * sizeof(std::int64_t),
                        stream));
  unsigned blocks = static_cast<unsigned>((pixels + 255) / 256);
  unsigned row_blocks = static_cast<unsigned>(
      (std::size_t(w) * (h + 2 * (patch / 2)) + 255) / 256);
  for (int dy = -window / 2; dy <= window / 2; ++dy)
    for (int dx = -window / 2; dx <= window / 2; ++dx) {
      distance_rows<<<row_blocks, 256, 0, stream>>>(src, rows, w, h, c,
                                                    patch / 2, dx, dy);
      check(cudaGetLastError());
      accumulate<<<blocks, 256, 0, stream>>>(
          src, rows, sums, w, h, c, patch / 2, dx, dy, weights, shift, levels);
      check(cudaGetLastError());
    }
  finish<<<blocks, 256, 0, stream>>>(dst, sums, pixels, c);
  check(cudaGetLastError());
}
} // namespace detail
} // namespace cuda
} // namespace image_kernels
} // namespace viame
