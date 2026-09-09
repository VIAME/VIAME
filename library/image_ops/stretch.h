/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_STRETCH_H
#define VIAME_IMAGE_OPS_STRETCH_H

#include <vital/types/image.h>

#include <cstddef>
#include <limits>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Smallest and largest value in the image.
template < typename T >
void
value_range( kwiver::vital::image_of< T > const& image, T& low, T& high )
{
  low = T{ 0 };
  high = T{ 0 };

  if( image.width() == 0 || image.height() == 0 || image.depth() == 0 )
  {
    return;
  }

  low = image( 0, 0, 0 );
  high = low;

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const value = image( i, j, plane );

        if( value < low )  { low = value; }
        if( value > high ) { high = value; }
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// Map the image's own range onto [\p low, \p high], as doubles.
///
/// A flat image maps to \p low everywhere rather than dividing by zero.
/// Reproduces `vil_convert_stretch_range`.
template < typename T >
kwiver::vital::image_of< double >
stretch_range( kwiver::vital::image_of< T > const& image,
               double low, double high )
{
  T source_low;
  T source_high;
  value_range( image, source_low, source_high );

  double scale = 0.0;

  if( source_high > source_low )
  {
    scale = ( high - low ) /
            static_cast< double >( source_high - source_low );
  }

  auto const offset = -1.0 * static_cast< double >( source_low ) * scale + low;

  kwiver::vital::image_of< double > result( image.width(), image.height(),
                                            image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        result( i, j, plane ) =
          scale * static_cast< double >( image( i, j, plane ) ) + offset;
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Map [\p source_low, \p source_high] onto [\p low, \p high], clamping.
///
/// Values at or below the source low come out at \p low and those at or above
/// the source high at \p high. Reproduces
/// `vil_convert_stretch_range_limited`.
template < typename T >
kwiver::vital::image_of< double >
stretch_range_limited( kwiver::vital::image_of< T > const& image,
                       T source_low, T source_high,
                       double low, double high )
{
  auto const scale = ( high - low ) /
                     static_cast< double >( source_high - source_low );

  kwiver::vital::image_of< double > result( image.width(), image.height(),
                                            image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const value = image( i, j, plane );

        result( i, j, plane ) =
          ( value <= source_low ) ? low
          : ( value >= source_high ) ? high
          : low + scale * static_cast< double >( value - source_low );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Map the image's own range onto the whole of an 8 bit output.
///
/// The dedicated byte path of `vil_convert_stretch_range`, which scales
/// straight into the byte rather than through a double image, so the top
/// value lands on 255 only when it divides exactly.
template < typename T >
kwiver::vital::image_of< uint8_t >
stretch_to_byte( kwiver::vital::image_of< T > const& image )
{
  T source_low;
  T source_high;
  value_range( image, source_low, source_high );

  auto const offset = -1.0 * static_cast< double >( source_low );
  double scale = 0.0;

  if( source_high > source_low )
  {
    scale = 255.0 / static_cast< double >( source_high - source_low );
  }

  kwiver::vital::image_of< uint8_t > result( image.width(), image.height(),
                                             image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        result( i, j, plane ) = static_cast< uint8_t >(
          scale * ( static_cast< double >( image( i, j, plane ) ) + offset ) );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// The output range a stretch targets for pixel type \p T.
///
/// The maximum is extended by almost one so that the top of the source range
/// still truncates to the type maximum rather than falling a step short,
/// which is what spreads the values evenly across the range. Real types
/// target [0, 1] instead.
template < typename T >
void
stretch_target( double& low, double& high )
{
  if( std::numeric_limits< T >::is_integer )
  {
    constexpr double almost_one = 1 - 1e-6;
    low = static_cast< double >( std::numeric_limits< T >::min() );
    high = static_cast< double >( std::numeric_limits< T >::max() ) +
           almost_one;
  }
  else
  {
    low = 0.0;
    high = 1.0;
  }
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_STRETCH_H
