/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_MORPHOLOGY_H
#define VIAME_IMAGE_OPS_MORPHOLOGY_H

#include <viame/core_types/image.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace viame {
namespace image_ops {

/// Offsets, in (i, j), of the pixels a morphological operation looks at.
typedef std::vector< std::pair< int, int > > structuring_element;

// ----------------------------------------------------------------------------
/// Every offset strictly inside a circle of \p radius.
///
/// The comparison is strict, which is why a radius of exactly 1 gives a single
/// pixel and 2 gives a 3x3 square rather than a plus: at radius 2 the offset
/// (2, 0) is at distance 4, not less than 4, and drops out. VXL's
/// `vil_structuring_element::set_to_disk` does the same, and the pipelines are
/// set up around the sizes it produces.
inline structuring_element
disk_element( double radius )
{
  structuring_element element;

  auto const limit = static_cast< int >( radius );
  auto const squared = radius * radius;

  for( int j = -limit; j <= limit; ++j )
  {
    for( int i = -limit; i <= limit; ++i )
    {
      if( static_cast< double >( i * i + j * j ) < squared )
      {
        element.emplace_back( i, j );
      }
    }
  }

  return element;
}

// ----------------------------------------------------------------------------
/// A horizontal run from -\p radius to \p radius.
inline structuring_element
line_i_element( double radius )
{
  structuring_element element;
  auto const limit = static_cast< int >( radius );

  for( int i = -limit; i <= limit; ++i )
  {
    element.emplace_back( i, 0 );
  }

  return element;
}

// ----------------------------------------------------------------------------
/// A vertical run from -\p radius to \p radius.
inline structuring_element
line_j_element( double radius )
{
  structuring_element element;
  auto const limit = static_cast< int >( radius );

  for( int j = -limit; j <= limit; ++j )
  {
    element.emplace_back( 0, j );
  }

  return element;
}

namespace detail {

// ----------------------------------------------------------------------------
/// Apply \p element at every pixel, combining with `and` (erode) or `or`.
///
/// Offsets that fall outside the image are skipped rather than treated as
/// set or unset, so the element is effectively clipped to the border. That
/// means eroding an all-true image leaves it all true, which is what VXL
/// does and what the pipelines that erode near a frame edge expect.
inline kwiver::vital::image_of< bool >
apply( kwiver::vital::image_of< bool > const& image,
       structuring_element const& element,
       bool erode )
{
  auto const width = static_cast< int >( image.width() );
  auto const height = static_cast< int >( image.height() );

  kwiver::vital::image_of< bool > result( image.width(), image.height(),
                                          image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( int j = 0; j < height; ++j )
    {
      for( int i = 0; i < width; ++i )
      {
        bool value = erode;

        for( auto const& offset : element )
        {
          auto const si = i + offset.first;
          auto const sj = j + offset.second;

          if( si < 0 || si >= width || sj < 0 || sj >= height )
          {
            continue;
          }

          bool const sample = image( si, sj, plane );

          if( erode )
          {
            if( !sample ) { value = false; break; }
          }
          else if( sample )
          {
            value = true;
            break;
          }
        }

        result( i, j, plane ) = value;
      }
    }
  }

  return result;
}

} // namespace detail

// ----------------------------------------------------------------------------
inline kwiver::vital::image_of< bool >
erode( kwiver::vital::image_of< bool > const& image,
       structuring_element const& element )
{
  return detail::apply( image, element, true );
}

// ----------------------------------------------------------------------------
inline kwiver::vital::image_of< bool >
dilate( kwiver::vital::image_of< bool > const& image,
        structuring_element const& element )
{
  return detail::apply( image, element, false );
}

// ----------------------------------------------------------------------------
/// Erode then dilate: removes specks smaller than the element.
inline kwiver::vital::image_of< bool >
opening( kwiver::vital::image_of< bool > const& image,
         structuring_element const& element )
{
  return dilate( erode( image, element ), element );
}

// ----------------------------------------------------------------------------
/// Dilate then erode: fills holes smaller than the element.
inline kwiver::vital::image_of< bool >
closing( kwiver::vital::image_of< bool > const& image,
         structuring_element const& element )
{
  return erode( dilate( image, element ), element );
}

// ----------------------------------------------------------------------------
/// Collapse every plane into one with `or` (union) or `and` (intersection).
inline kwiver::vital::image_of< bool >
combine_planes( kwiver::vital::image_of< bool > const& image, bool use_union )
{
  kwiver::vital::image_of< bool > result( image.width(), image.height(), 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      bool value = image( i, j, 0 );

      for( size_t plane = 1; plane < image.depth(); ++plane )
      {
        value = use_union ? ( value || image( i, j, plane ) )
                          : ( value && image( i, j, plane ) );
      }

      result( i, j, 0 ) = value;
    }
  }

  return result;
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_MORPHOLOGY_H
