/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shot break detection utility functions implementation
 */

#include "detect_shot_breaks.h"

#include <cmath>
#include <algorithm>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
std::vector< double >
compute_image_histogram(
  const kv::image_container_sptr& img,
  unsigned histogram_bins )
{
  if( !img )
  {
    return std::vector< double >();
  }

  const kv::image& image = img->get_image();
  size_t width = image.width();
  size_t height = image.height();
  size_t depth = image.depth();

  // Create histogram bins for each channel
  size_t total_bins = histogram_bins * depth;
  std::vector< double > histogram( total_bins, 0.0 );

  // Sample pixels (use stride for large images to improve performance)
  size_t stride = std::max( size_t( 1 ), std::min( width, height ) / 100 );
  size_t sample_count = 0;

  for( size_t y = 0; y < height; y += stride )
  {
    for( size_t x = 0; x < width; x += stride )
    {
      for( size_t c = 0; c < depth; ++c )
      {
        uint8_t pixel_val = image.at< uint8_t >( x, y, c );
        size_t bin = static_cast< size_t >( pixel_val ) * histogram_bins / 256;
        bin = std::min( bin, static_cast< size_t >( histogram_bins - 1 ) );
        histogram[c * histogram_bins + bin] += 1.0;
      }
      sample_count++;
    }
  }

  // Normalize histogram
  if( sample_count > 0 )
  {
    for( auto& val : histogram )
    {
      val /= static_cast< double >( sample_count );
    }
  }

  return histogram;
}

// -----------------------------------------------------------------------------
double
compute_histogram_difference(
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  unsigned histogram_bins )
{
  auto hist1 = compute_image_histogram( image1, histogram_bins );
  auto hist2 = compute_image_histogram( image2, histogram_bins );

  if( hist1.empty() || hist2.empty() || hist1.size() != hist2.size() )
  {
    return 0.0;
  }

  // Compute histogram intersection (similarity measure)
  // Intersection = sum of min values, result is 0-1 where 1 = identical
  double intersection = 0.0;
  for( size_t i = 0; i < hist1.size(); ++i )
  {
    intersection += std::min( hist1[i], hist2[i] );
  }

  // Convert to difference (0 = identical, 1 = completely different)
  return 1.0 - intersection;
}

// -----------------------------------------------------------------------------
double
compute_pixel_difference(
  const kv::image_container_sptr& img1,
  const kv::image_container_sptr& img2 )
{
  if( !img1 || !img2 )
  {
    return 0.0;
  }

  const kv::image& image1 = img1->get_image();
  const kv::image& image2 = img2->get_image();

  // Check dimensions match
  if( image1.width() != image2.width() ||
      image1.height() != image2.height() ||
      image1.depth() != image2.depth() )
  {
    // Different dimensions = scene change
    return 1.0;
  }

  size_t width = image1.width();
  size_t height = image1.height();
  size_t depth = image1.depth();

  // Sample pixels (use stride for large images to improve performance)
  size_t stride = std::max( size_t( 1 ), std::min( width, height ) / 100 );
  double total_diff = 0.0;
  size_t sample_count = 0;

  for( size_t y = 0; y < height; y += stride )
  {
    for( size_t x = 0; x < width; x += stride )
    {
      for( size_t c = 0; c < depth; ++c )
      {
        int val1 = static_cast< int >( image1.at< uint8_t >( x, y, c ) );
        int val2 = static_cast< int >( image2.at< uint8_t >( x, y, c ) );
        total_diff += std::abs( val1 - val2 );
      }
      sample_count++;
    }
  }

  // Normalize to 0-1 range (max difference is 255 per channel)
  if( sample_count > 0 && depth > 0 )
  {
    return total_diff / ( sample_count * depth * 255.0 );
  }

  return 0.0;
}

// -----------------------------------------------------------------------------
double
compute_descriptor_distance(
  const kv::descriptor_sptr& desc1,
  const kv::descriptor_sptr& desc2 )
{
  if( !desc1 || !desc2 )
  {
    return 1.0; // Maximum distance if either descriptor is null
  }

  // Get descriptor data as doubles
  std::vector< double > vec1 = desc1->as_double();
  std::vector< double > vec2 = desc2->as_double();

  if( vec1.empty() || vec2.empty() || vec1.size() != vec2.size() )
  {
    return 1.0; // Maximum distance if sizes don't match
  }

  // Compute L2 (Euclidean) distance
  double sum_sq = 0.0;
  double norm1_sq = 0.0;
  double norm2_sq = 0.0;

  for( size_t i = 0; i < vec1.size(); ++i )
  {
    double diff = vec1[i] - vec2[i];
    sum_sq += diff * diff;
    norm1_sq += vec1[i] * vec1[i];
    norm2_sq += vec2[i] * vec2[i];
  }

  // Normalize distance to 0-1 range
  // distance = sqrt(sum_sq) / sqrt(norm1_sq + norm2_sq)
  double max_dist = std::sqrt( norm1_sq + norm2_sq );
  if( max_dist > 0 )
  {
    return std::sqrt( sum_sq ) / max_dist;
  }

  return 0.0;
}

} // end namespace core

} // end namespace viame
