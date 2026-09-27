/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_DENOISE_WEIGHTS_H
#define VIAME_IMAGE_KERNELS_DENOISE_WEIGHTS_H
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
namespace viame {
namespace image_kernels {
namespace detail {
/// The smallest power of two at or above \p value, as an exponent.
inline int nearest_power_of_two(int value) {
  auto power = 0;

  while ((1 << power) < value) {
    ++power;
  }

  return power;
}

struct nlm_weights {
  int shift;
  std::vector<int64_t> values;
};
inline nlm_weights make_nlm_weights(double strength, int planes, int patch_size,
                                    int window_size) {
  // The fixed-point scale is whatever keeps the weighted sum inside an int.
  auto const ceiling = static_cast<int64_t>(window_size) * window_size * 255;
  auto const scale =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max() / ceiling);

  auto const area = patch_size * patch_size;
  auto const shift = nearest_power_of_two(area);
  auto const step = static_cast<double>(1 << shift) / area;
  auto const furthest = 255 * 255 * planes;
  auto const levels = static_cast<int>(furthest / step) + 1;

  std::vector<int64_t> weight(static_cast<size_t>(levels));

  for (int level = 0; level < levels; ++level) {
    auto const distance = level * step;
    auto value = (strength == 0.0)
                     ? (distance == 0.0 ? 1.0 : 0.0)
                     : std::exp(-distance / (strength * strength * planes));

    if (std::isnan(value)) {
      value = 1.0;
    }

    auto found = static_cast<int64_t>(
        std::nearbyint(static_cast<double>(scale) * value));

    if (static_cast<double>(found) < 0.001 * static_cast<double>(scale)) {
      found = 0;
    }

    weight[static_cast<size_t>(level)] = found;
  }

  return {shift, std::move(weight)};
}
} // namespace detail
} // namespace image_kernels
} // namespace viame
#endif
