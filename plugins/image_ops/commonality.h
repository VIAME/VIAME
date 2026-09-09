/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_COMMONALITY_H
#define VIAME_IMAGE_OPS_COMMONALITY_H

#include <vital/types/image.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Whether \p value is a power of two, which the histogram indexing needs.
inline bool
is_power_of_two( unsigned value )
{
  return value > 0 && ( value & ( value - 1 ) ) == 0;
}

// ----------------------------------------------------------------------------
/// Floor of log2, by the shift-until-one rule VXL uses.
template < typename T >
unsigned
integer_log2( T value )
{
  unsigned result = 0;

  while( ( value >> result ) > 1 )
  {
    ++result;
  }

  return result;
}

// ----------------------------------------------------------------------------
/// How common each pixel's colour is across a region of the image.
///
/// Bins every pixel into a `bins` per channel histogram, then replaces each
/// pixel with how often its own bin occurred, scaled so that a bin holding
/// every pixel comes out at `scale` (or at the type maximum when `scale` is
/// 0) and saturating there. Rare colours come out dark, common ones bright.
///
/// The bin index drops the low bits of each channel rather than dividing, so
/// `bins` must be a power of two; the caller checks that.
///
/// \param image  Integral pixels, any plane count.
/// \param bins   Bins per channel.
/// \param scale  Output value for a bin holding every pixel; 0 means the
///               type maximum.
template < typename T >
kwiver::vital::image_of< T >
color_commonality( kwiver::vital::image_of< T > const& image,
                   unsigned bins, unsigned scale )
{
  kwiver::vital::image_of< T > result( image.width(), image.height(), 1 );

  auto const width = image.width();
  auto const height = image.height();
  auto const depth = image.depth();

  if( width == 0 || height == 0 || depth == 0 || !is_power_of_two( bins ) )
  {
    return result;
  }

  constexpr auto type_max = std::numeric_limits< T >::max();
  auto const ceiling = static_cast< unsigned >( type_max );
  auto const factor = ( scale == 0 ) ? ceiling : scale;

  // Only a three channel image gets a full colour cube; anything else is
  // binned as if it were a single channel, which is what arrows/vxl sizes
  // its histogram for
  size_t size = bins;
  if( depth == 3 )
  {
    size = static_cast< size_t >( bins ) * bins * bins;
  }

  std::vector< size_t > steps( depth, 1 );
  for( size_t plane = 1; plane < depth; ++plane )
  {
    steps[ plane ] = steps[ plane - 1 ] * bins;
  }

  auto const shift =
    integer_log2( type_max ) + 1 - integer_log2( bins );

  std::vector< uint64_t > histogram( size, 0 );

  auto bin_of =
    [ & ]( size_t i, size_t j )
    {
      size_t index = 0;

      for( size_t plane = 0; plane < depth; ++plane )
      {
        index += steps[ plane ] *
                 static_cast< size_t >( image( i, j, plane ) >> shift );
      }

      return index < size ? index : size - 1;
    };

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      ++histogram[ bin_of( i, j ) ];
    }
  }

  uint64_t total = 0;
  for( auto count : histogram )
  {
    total += count;
  }

  if( total == 0 )
  {
    return result;
  }

  for( auto& count : histogram )
  {
    auto const value = ( static_cast< uint64_t >( factor ) * count ) / total;
    count = ( value > ceiling ) ? ceiling : value;
  }

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      result( i, j, 0 ) = static_cast< T >( histogram[ bin_of( i, j ) ] );
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// `color_commonality` computed independently over a grid of tiles.
///
/// Each tile gets its own histogram, so a colour that is common in one corner
/// and rare in another reads differently in each. Tile edges are the same
/// integer splits arrows/vxl uses, so tiles tile the image exactly.
///
/// arrows/vxl's grid path builds its regions with the corners in the wrong
/// order and leaves most of the output unwritten; there is nothing to be bit
/// compatible with, so this computes what the option says it computes. No
/// shipped pipeline uses it. See tests/golden/README.md.
template < typename T >
kwiver::vital::image_of< T >
color_commonality_grid( kwiver::vital::image_of< T > const& image,
                        unsigned bins, unsigned scale,
                        unsigned columns, unsigned rows )
{
  kwiver::vital::image_of< T > result( image.width(), image.height(), 1 );

  if( columns == 0 || rows == 0 )
  {
    return result;
  }

  auto const width = image.width();
  auto const height = image.height();

  for( unsigned row = 0; row < rows; ++row )
  {
    auto const top = ( row * height ) / rows;
    auto const bottom = ( ( row + 1 ) * height ) / rows;

    for( unsigned column = 0; column < columns; ++column )
    {
      auto const left = ( column * width ) / columns;
      auto const right = ( ( column + 1 ) * width ) / columns;

      if( right <= left || bottom <= top )
      {
        continue;
      }

      kwiver::vital::image_of< T > tile( right - left, bottom - top,
                                         image.depth() );

      for( size_t plane = 0; plane < image.depth(); ++plane )
      {
        for( size_t j = top; j < bottom; ++j )
        {
          for( size_t i = left; i < right; ++i )
          {
            tile( i - left, j - top, plane ) = image( i, j, plane );
          }
        }
      }

      auto const filtered = color_commonality( tile, bins, scale );

      for( size_t j = top; j < bottom; ++j )
      {
        for( size_t i = left; i < right; ++i )
        {
          result( i, j, 0 ) = filtered( i - left, j - top, 0 );
        }
      }
    }
  }

  return result;
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_COMMONALITY_H
