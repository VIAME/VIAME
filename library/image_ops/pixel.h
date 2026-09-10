/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_PIXEL_H
#define VIAME_IMAGE_OPS_PIXEL_H

#include <cstddef>
#include <limits>
#include <type_traits>

namespace viame {
namespace image_ops {

/// How one pixel value becomes another type.
///
/// These reproduce VXL's conversion rules, which the pipelines have been
/// tuned against for years and which are not always what a fresh
/// implementation would choose:
///
/// * `cast` truncates toward zero and does not clamp, so a float 300.7
///   reaching a byte is whatever the C++ conversion gives. That is
///   `vil_convert_cast`.
/// * `round` adds a half away from zero and then truncates, still without
///   clamping. That is `vil_convert_round_pixel`.
///
/// Both are kept honest by the recordings under `tests/golden/vxl`.

// ----------------------------------------------------------------------------
/// Convert without rounding or clamping, as a C++ cast would.
template < typename Out, typename In >
Out
cast_pixel( In value )
{
  return static_cast< Out >( value );
}

// ----------------------------------------------------------------------------
/// Convert a real value to \p Out, rounding half away from zero.
///
/// Integral \p Out only; a real \p Out is just a cast.
template < typename Out >
Out
round_pixel( double value )
{
  if constexpr( std::is_floating_point< Out >::value )
  {
    return static_cast< Out >( value );
  }
  else
  {
    return static_cast< Out >( value > 0.0 ? value + 0.5 : value - 0.5 );
  }
}

// ----------------------------------------------------------------------------
/// Convert a real value to \p Out, rounding half away from zero and clamping.
///
/// The saturating counterpart of `round_pixel`, which is what OpenCV's
/// `saturate_cast` does and what a colour conversion needs: L*a*b* and HSV
/// both have coordinates outside the RGB gamut, so a round trip through
/// either produces values below zero and above the top of the type, and
/// wrapping them is a black pixel where a white one belongs.
///
/// `round_pixel` stays unclamped because the VXL recordings depend on it.
template < typename Out >
Out
saturate_pixel( double value )
{
  if constexpr( std::is_floating_point< Out >::value )
  {
    return static_cast< Out >( value );
  }
  else
  {
    constexpr auto low =
      static_cast< double >( std::numeric_limits< Out >::lowest() );
    constexpr auto high =
      static_cast< double >( std::numeric_limits< Out >::max() );

    value = value > 0.0 ? value + 0.5 : value - 0.5;

    if( value <= low ) { return std::numeric_limits< Out >::lowest(); }
    if( value >= high ) { return std::numeric_limits< Out >::max(); }

    return static_cast< Out >( value );
  }
}

// ----------------------------------------------------------------------------
/// The largest value \p T holds, as a double.
template < typename T >
constexpr double
pixel_max()
{
  return static_cast< double >( std::numeric_limits< T >::max() );
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_PIXEL_H
