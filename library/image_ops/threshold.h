/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_THRESHOLD_H
#define VIAME_IMAGE_OPS_THRESHOLD_H

#include <image_ops/statistics.h>

#include <viame/core_types/image.h>

#include <cstddef>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Mask of the pixels at or above \p value.
///
/// Inclusive, matching `vil_threshold_above`: a pixel exactly equal to the
/// threshold is kept.
template < typename T >
kwiver::vital::image_of< bool >
threshold_above( kwiver::vital::image_of< T > const& image, T value )
{
  kwiver::vital::image_of< bool > result( image.width(), image.height(),
                                          image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        result( i, j, plane ) = image( i, j, plane ) >= value;
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Mask of the pixels at or above the value at \p fraction of the
/// distribution.
///
/// \param fraction       Where to cut, in [0, 1]; 0.95 keeps the brightest
///                       twentieth.
/// \param sampling_points How many samples the percentile is taken from.
template < typename T >
kwiver::vital::image_of< bool >
threshold_percentile( kwiver::vital::image_of< T > const& image,
                      double fraction,
                      size_t sampling_points = 1000 )
{
  auto const values =
    percentiles( image, std::vector< double >{ fraction }, sampling_points );

  return threshold_above( image, values.empty() ? T{ 0 } : values[ 0 ] );
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_THRESHOLD_H
