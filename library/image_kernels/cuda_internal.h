/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#ifndef VIAME_IMAGE_KERNELS_CUDA_INTERNAL_H
#define VIAME_IMAGE_KERNELS_CUDA_INTERNAL_H
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace viame {
namespace image_kernels {
namespace cuda {
namespace detail {
inline void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string("VIAME CUDA: ") +
                             cudaGetErrorString(result));
}
// Operations are explicitly fused where the CPU uses fma. Disable implicit
// contraction for other arithmetic in the CUDA compile options.
void gaussian(float const *, float *, float *, int, int, int, float const *,
              int, cudaStream_t);
void nlm(unsigned char const *, unsigned char *, int, int, int, int, int,
         std::int64_t const *, int, int, std::uint64_t *, std::int64_t *,
         cudaStream_t);
void gfit_motion(unsigned char const *, unsigned char *, double *, double *,
                 unsigned char *, std::size_t, int, int, cudaStream_t);
struct resize_entry {
  int index;
  float weight;
  int fixed;
};
void letterbox(unsigned char const *, unsigned char *, int, int, int, int, int,
               int, int, int, int, bool, int const *, int const *,
               resize_entry const *, resize_entry const *, cudaStream_t);
#ifdef __CUDACC__
__device__ inline int reflect101(int x, int n) {
  if (n == 1)
    return 0;
  int period = 2 * (n - 1);
  x %= period;
  if (x < 0)
    x += period;
  return x < n ? x : period - x;
}
#endif
} // namespace detail
} // namespace cuda
} // namespace image_kernels
} // namespace viame
#endif
