/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_GAUSSIAN_KERNEL_H
#define VIAME_IMAGE_KERNELS_GAUSSIAN_KERNEL_H
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace viame {
namespace image_kernels {
// ----------------------------------------------------------------------------
/// A Gaussian of \p size samples with standard deviation \p sigma.
///
/// `cv::getGaussianKernel`, whose three surprises are all in here:
///
/// * when \p sigma is not positive it does **not** simply derive one from the
///   size for a small kernel. For an odd size of **nine** or less it returns a
///   fixed table, and the table's rows are not samples of any Gaussian --
///   (0.25, 0.5, 0.25) is a binomial -- so deriving the sigma and evaluating
///   gives a visibly different blur at size three, which is the size the
///   pipelines use most. Size nine is easy to stop one short of: the table
///   used to end at seven, and a 16 bit image is where the extra row shows,
///   because at Q16.16 the two kernels differ by more than the rounding;
/// * the derived sigma is a **fused** `size * 0.15 + 0.35`. Written as
///   `0.3 * ((size - 1) * 0.5 - 1) + 0.8` -- which is what OpenCV's own
///   comment still says -- it is the same number in exact arithmetic and a
///   different double at sizes 15, 19 and 51, again enough to move a Q16.16
///   tap;
/// * the samples are `exp( x * x * -0.125 / sigma^2 )` over the **doubled**
///   offset `x = 2i - (size - 1)`, so the argument of the exponential is an
///   exact integer times a constant, and the normalising sum is built from
///   half the kernel doubled plus the centre's one rather than by adding up
///   every sample. Both are followed here, since the association is what the
///   last bit of each tap depends on.
///
/// The result sums to one either way.
inline std::vector<double> gaussian_kernel_1d(size_t size, double sigma = 0.0) {
  if (size == 0 || size % 2 == 0) {
    throw std::invalid_argument(
        "gaussian_kernel_1d: the size has to be odd and positive");
  }

  if (sigma <= 0.0 && size <= 9) {
    // cv::getGaussianKernelBitExact's tables, indexed by (size - 1) / 2
    static std::vector<double> const table[] = {
        {1.0},
        {0.25, 0.5, 0.25},
        {0.0625, 0.25, 0.375, 0.25, 0.0625},
        {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125},
        {4.0 / 256.0, 13.0 / 256.0, 30.0 / 256.0, 51.0 / 256.0, 60.0 / 256.0,
         51.0 / 256.0, 30.0 / 256.0, 13.0 / 256.0, 4.0 / 256.0},
    };

    return table[(size - 1) / 2];
  }

  if (sigma <= 0.0) {
    sigma = std::fma(static_cast<double>(size), 0.15, 0.35);
  }

  auto const scale = -0.125 / (sigma * sigma);
  auto const half = (size - 1) / 2;

  std::vector<double> values(half);
  auto total = 0.0;

  for (size_t i = 0; i < half; ++i) {
    auto const x =
        2.0 * static_cast<double>(i) - (static_cast<double>(size) - 1.0);

    values[i] = std::exp(x * x * scale);
    total += values[i];
  }

  total = total * 2.0 + 1.0;

  auto const inverse = 1.0 / total;
  std::vector<double> out(size);

  for (size_t i = 0; i < half; ++i) {
    out[i] = out[size - 1 - i] = values[i] * inverse;
  }

  out[half] = inverse;

  return out;
}

} // namespace image_kernels
} // namespace viame
#endif
