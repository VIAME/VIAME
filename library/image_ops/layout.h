/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Putting images beside each other
///
/// What `cv::hconcat` and `cv::vconcat` did, which is how `merge_images`
/// builds a side-by-side stereo frame and how the training augmentations
/// stack their channels.
///
/// `channels.h` has the plane-wise split and merge; this is the spatial one.

#ifndef VIAME_IMAGE_OPS_LAYOUT_H
#define VIAME_IMAGE_OPS_LAYOUT_H

#include <viame/core_types/image.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// The images laid left to right, which is `cv::hconcat`.
///
/// They have to agree in height and plane count; a caller with images of
/// different heights resizes or pads first, because which of those it wanted
/// is not something this can guess.
template < typename T >
kwiver::vital::image_of< T >
horizontal_concat( std::vector< kwiver::vital::image_of< T > > const& images )
{
  if( images.empty() )
  {
    throw std::invalid_argument( "horizontal_concat: nothing to join" );
  }

  auto const height = images[ 0 ].height();
  auto const depth = images[ 0 ].depth();
  size_t width = 0;

  for( auto const& image : images )
  {
    if( image.height() != height || image.depth() != depth )
    {
      throw std::invalid_argument(
        "horizontal_concat: the images differ in height or plane count" );
    }

    width += image.width();
  }

  kwiver::vital::image_of< T > out( width, height, depth );

  size_t offset = 0;

  for( auto const& image : images )
  {
    for( size_t plane = 0; plane < depth; ++plane )
    {
      for( size_t j = 0; j < height; ++j )
      {
        for( size_t i = 0; i < image.width(); ++i )
        {
          out( offset + i, j, plane ) = image( i, j, plane );
        }
      }
    }

    offset += image.width();
  }

  return out;
}

// ----------------------------------------------------------------------------
/// The images laid top to bottom, which is `cv::vconcat`.
template < typename T >
kwiver::vital::image_of< T >
vertical_concat( std::vector< kwiver::vital::image_of< T > > const& images )
{
  if( images.empty() )
  {
    throw std::invalid_argument( "vertical_concat: nothing to join" );
  }

  auto const width = images[ 0 ].width();
  auto const depth = images[ 0 ].depth();
  size_t height = 0;

  for( auto const& image : images )
  {
    if( image.width() != width || image.depth() != depth )
    {
      throw std::invalid_argument(
        "vertical_concat: the images differ in width or plane count" );
    }

    height += image.height();
  }

  kwiver::vital::image_of< T > out( width, height, depth );

  size_t offset = 0;

  for( auto const& image : images )
  {
    for( size_t plane = 0; plane < depth; ++plane )
    {
      for( size_t j = 0; j < image.height(); ++j )
      {
        for( size_t i = 0; i < width; ++i )
        {
          out( i, offset + j, plane ) = image( i, j, plane );
        }
      }
    }

    offset += image.height();
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
