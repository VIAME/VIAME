/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include "cuda_internal.h"
namespace viame {
namespace image_kernels {
namespace cuda {
namespace detail {
namespace {
__device__ unsigned char variance_channel(double value, double &mean,
                                          unsigned char previous, int count,
                                          int window) {
  if (count == 0) {
    mean = value;
    return 0;
  }
  double before = fabs(value - mean);
  if (count < window) {
    double weight = 1.0 / (count + 1);
    mean = (1.0 - weight) * mean + weight * value;
  } else {
    mean += (1.0 / window) * (value - double(previous));
  }
  // Preserve frame_averager<uint8_t>'s truncation before the second distance.
  double variance =
      before * fabs(value - double(static_cast<unsigned char>(mean)));
  return variance <= 510.0 ? static_cast<unsigned char>(variance * 0.5 + 0.5)
                           : 255;
}
__global__ void motion(unsigned char const *src, unsigned char *dst,
                       double *short_mean, double *long_mean,
                       unsigned char *previous, std::size_t pixels,
                       int channels, int count) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels)
    return;
  double value = 0;
  if (channels == 3)
    value = double(src[i * 3]) * 0.2125 + double(src[i * 3 + 1]) * 0.7154 +
            double(src[i * 3 + 2]) * 0.0721;
  else {
    unsigned char total = 0;
    for (int c = 0; c < channels; ++c)
      total = static_cast<unsigned char>(total + src[i * channels + c]);
    value = int(total) / channels;
  }
  unsigned char grey = static_cast<unsigned char>(value + 0.5);
  auto old = count ? previous[i] : 0;
  dst[i * 3] = variance_channel(grey, short_mean[i], old, count, 5);
  dst[i * 3 + 1] = grey;
  dst[i * 3 + 2] = variance_channel(grey, long_mean[i], old, count, 30);
  previous[i] = grey;
}
} // namespace
void gfit_motion(unsigned char const *src, unsigned char *dst,
                 double *short_mean, double *long_mean, unsigned char *previous,
                 std::size_t pixels, int channels, int count,
                 cudaStream_t stream) {
  motion<<<static_cast<unsigned>((pixels + 255) / 256), 256, 0, stream>>>(
      src, dst, short_mean, long_mean, previous, pixels, channels, count);
  check(cudaGetLastError());
}
} // namespace detail
} // namespace cuda
} // namespace image_kernels
} // namespace viame
