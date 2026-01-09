/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shot break detection utility functions
 */

#ifndef VIAME_CORE_DETECT_SHOT_BREAKS_H
#define VIAME_CORE_DETECT_SHOT_BREAKS_H

#include "viame_core_export.h"

#include <vital/types/image_container.h>
#include <vital/types/descriptor.h>

#include <vector>

namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

// =============================================================================
// Utility functions for shot break detection
// =============================================================================

/**
 * \brief Compute a normalized color histogram for an image
 *
 * Computes a histogram for each color channel with the specified number of bins.
 * Uses sparse sampling for large images to improve performance.
 *
 * \param image Input image
 * \param histogram_bins Number of bins per channel (typically 32)
 * \return Normalized histogram vector (size = histogram_bins * depth)
 */
VIAME_CORE_EXPORT std::vector< double > compute_image_histogram(
  const kv::image_container_sptr& image,
  unsigned histogram_bins );

/**
 * \brief Compute the difference between two image histograms
 *
 * Computes histograms for both images and returns the histogram intersection
 * distance (0 = identical, 1 = completely different).
 *
 * \param image1 First image
 * \param image2 Second image
 * \param histogram_bins Number of bins per channel
 * \return Histogram difference in range [0, 1]
 */
VIAME_CORE_EXPORT double compute_histogram_difference(
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  unsigned histogram_bins );

/**
 * \brief Compute the mean absolute pixel difference between two images
 *
 * Uses sparse sampling for large images to improve performance.
 * Returns 1.0 if images have different dimensions.
 *
 * \param image1 First image
 * \param image2 Second image
 * \return Normalized pixel difference in range [0, 1]
 */
VIAME_CORE_EXPORT double compute_pixel_difference(
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2 );

/**
 * \brief Compute normalized distance between two descriptors
 *
 * Computes the L2 distance between descriptor vectors, normalized to [0, 1].
 *
 * \param desc1 First descriptor
 * \param desc2 Second descriptor
 * \return Normalized distance in range [0, 1], or 1.0 if descriptors are incompatible
 */
VIAME_CORE_EXPORT double compute_descriptor_distance(
  const kv::descriptor_sptr& desc1,
  const kv::descriptor_sptr& desc2 );

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_DETECT_SHOT_BREAKS_H
