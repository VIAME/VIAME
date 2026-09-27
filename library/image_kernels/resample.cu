/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include "cuda_internal.h"
namespace viame {
namespace image_kernels {
namespace cuda {
namespace detail {
namespace {
__global__ void resize_kernel(unsigned char const *src, unsigned char *dst,
                              int sw, int sh, int channels, int dw, int dh,
                              int ew, int eh, int left, int top, bool area,
                              int const *xoff, int const *yoff,
                              resize_entry const *xtab,
                              resize_entry const *ytab) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(dw) * dh * channels)
    return;
  int c = i % channels, x = (i / channels) % dw - left,
      y = i / (std::size_t(dw) * channels) - top;
  if (x < 0 || y < 0 || x >= ew || y >= eh) {
    dst[i] = 0;
    return;
  }
  if (sw == ew && sh == eh) {
    dst[i] = src[(std::size_t(y) * sw + x) * channels + c];
    return;
  }
  int result;
  if (area && sw % ew == 0 && sh % eh == 0) {
    int sx = sw / ew, sy = sh / eh;
    unsigned long long sum = 0;
    for (int j = 0; j < sy; ++j)
      for (int k = 0; k < sx; ++k)
        sum += src[(std::size_t(y * sy + j) * sw + x * sx + k) * channels + c];
    if (sx == 2 && sy == 2 && (channels == 1 || channels == 3 || channels == 4))
      result = (sum + 2) >> 2;
    else
      result = __float2int_rn(float(sum) * float(1.0 / (double(sx) * sy)));
  } else if (area) {
    float sum = 0;
    for (int j = yoff[y]; j < yoff[y + 1]; ++j) {
      float row = 0;
      for (int k = xoff[x]; k < xoff[x + 1]; ++k)
        row += float(src[(std::size_t(ytab[j].index) * sw + xtab[k].index) *
                             channels +
                         c]) *
               xtab[k].weight;
      sum += row * ytab[j].weight;
    }
    result = __float2int_rn(sum);
  } else {
    long long sum = 0;
    for (int j = yoff[y]; j < yoff[y + 1]; ++j) {
      int row = 0;
      for (int k = xoff[x]; k < xoff[x + 1]; ++k)
        row += int(src[(std::size_t(ytab[j].index) * sw + xtab[k].index) *
                           channels +
                       c]) *
               xtab[k].fixed;
      sum += static_cast<long long>(row) * ytab[j].fixed;
    }
    result = int((sum + (1 << 21)) >> 22);
  }
  dst[i] = static_cast<unsigned char>(min(255, max(0, result)));
}
} // namespace
void letterbox(unsigned char const *src, unsigned char *dst, int sw, int sh,
               int c, int dw, int dh, int ew, int eh, int left, int top,
               bool area, int const *xoff, int const *yoff,
               resize_entry const *xtab, resize_entry const *ytab,
               cudaStream_t stream) {
  resize_kernel<<<static_cast<unsigned>((std::size_t(dw) * dh * c + 255) / 256),
                  256, 0, stream>>>(src, dst, sw, sh, c, dw, dh, ew, eh, left,
                                    top, area, xoff, yoff, xtab, ytab);
  check(cudaGetLastError());
}
} // namespace detail
} // namespace cuda
} // namespace image_kernels
} // namespace viame
