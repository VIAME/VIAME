/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_STATISTICS_H
#define VIAME_IMAGE_OPS_STATISTICS_H

#include <vital/types/image.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Sample the image on a fixed stride and return the samples sorted.
///
/// The stride, and the order it walks the image in, are exactly what VXL's
/// `sample_and_sort_image` does, because the percentiles taken from this are
/// what several filters threshold and stretch against, and pipelines are
/// tuned to those numbers. Two details of that walk are deliberate rather
/// than accidental-looking:
///
/// * the sample count is capped at one plane's worth of pixels, while the
///   stride is computed over all planes, so a colour image is sampled more
///   sparsely than a grey one of the same size;
/// * the walking position is not reset between planes, so each plane starts
///   where the previous one left off and the samples of a colour image come
///   from a different grid in each plane.
///
/// \param image        Image to sample.
/// \param sampling_points Requested number of samples per plane.
/// \param remove_extremes Drop values equal to zero or to the type maximum,
///                     which is how saturated backgrounds are kept from
///                     dominating a percentile.
template < typename T >
std::vector< T >
sample_and_sort( kwiver::vital::image_of< T > const& image,
                 size_t sampling_points,
                 bool remove_extremes = false )
{
  auto const width = image.width();
  auto const height = image.height();
  auto const depth = image.depth();

  std::vector< T > samples;

  if( width == 0 || height == 0 || depth == 0 )
  {
    return samples;
  }

  if( width * height < sampling_points )
  {
    sampling_points = width * height;
  }

  if( sampling_points == 0 )
  {
    return samples;
  }

  auto const scanning_area = width * height * depth;
  auto const pixel_step = scanning_area / sampling_points;

  samples.reserve( sampling_points * depth );

  size_t position = 0;

  for( size_t plane = 0; plane < depth; ++plane )
  {
    for( size_t sample = 0; sample < sampling_points;
         ++sample, position += pixel_step )
    {
      auto const i = position % width;
      auto const j = ( position / width ) % height;
      samples.push_back( image( i, j, plane ) );
    }
  }

  std::sort( samples.begin(), samples.end() );

  if( remove_extremes )
  {
    constexpr T low = T{ 0 };
    constexpr T high = std::numeric_limits< T >::max();

    auto const first =
      std::find_if( samples.begin(), samples.end(),
                    []( T value ){ return value != low; } );
    samples.erase( samples.begin(), first );

    auto const last =
      std::find_if( samples.rbegin(), samples.rend(),
                    []( T value ){ return value != high; } );
    samples.erase( last.base(), samples.end() );
  }

  return samples;
}

// ----------------------------------------------------------------------------
/// Value at each of \p fractions in the sampled distribution of \p image.
///
/// \param fractions Values in [0, 1]; 0.5 is the median.
template < typename T >
std::vector< T >
percentiles( kwiver::vital::image_of< T > const& image,
             std::vector< double > const& fractions,
             size_t sampling_points,
             bool remove_extremes = false )
{
  auto const samples =
    sample_and_sort( image, sampling_points, remove_extremes );

  std::vector< T > values( fractions.size(), T{ 0 } );

  if( samples.empty() )
  {
    return values;
  }

  // The -1 is because a percentile is the fraction of values that fall
  // below, and the +0.5 rounds rather than truncates the index
  auto const last_index = static_cast< double >( samples.size() - 1 );

  for( size_t index = 0; index < fractions.size(); ++index )
  {
    auto const offset =
      static_cast< size_t >( last_index * fractions[ index ] + 0.5 );
    values[ index ] = samples[ std::min( offset, samples.size() - 1 ) ];
  }

  return values;
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_STATISTICS_H
