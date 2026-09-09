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
