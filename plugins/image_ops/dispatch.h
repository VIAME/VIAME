/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_DISPATCH_H
#define VIAME_IMAGE_OPS_DISPATCH_H

#include <vital/types/image.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Call \p functor with the image typed as whatever it actually holds.
///
/// A `vital::image` carries its pixel type as traits rather than in the C++
/// type, so anything that works pixel by pixel has to recover the type
/// first. \p functor is a generic lambda taking `image_of< T > const&`; every
/// pixel type vital can hold is instantiated for it.
///
/// \throws std::runtime_error if the image holds a type not listed here.
template < typename Functor >
auto
dispatch_pixel_type( kwiver::vital::image const& image, Functor&& functor )
  -> decltype( functor( kwiver::vital::image_of< uint8_t >() ) )
{
  auto const& traits = image.pixel_traits();

#define VIAME_DISPATCH_CASE( KIND, BYTES, PIXEL )                     \
  if( traits.type == kwiver::vital::image_pixel_traits::KIND &&       \
      traits.num_bytes == BYTES )                                     \
  {                                                                   \
    return functor( kwiver::vital::image_of< PIXEL >( image ) );      \
  }

  VIAME_DISPATCH_CASE( BOOL, sizeof( bool ), bool )
  VIAME_DISPATCH_CASE( UNSIGNED, 1, uint8_t )
  VIAME_DISPATCH_CASE( UNSIGNED, 2, uint16_t )
  VIAME_DISPATCH_CASE( UNSIGNED, 4, uint32_t )
  VIAME_DISPATCH_CASE( UNSIGNED, 8, uint64_t )
  VIAME_DISPATCH_CASE( SIGNED, 1, int8_t )
  VIAME_DISPATCH_CASE( SIGNED, 2, int16_t )
  VIAME_DISPATCH_CASE( SIGNED, 4, int32_t )
  VIAME_DISPATCH_CASE( SIGNED, 8, int64_t )
  VIAME_DISPATCH_CASE( FLOAT, 4, float )
  VIAME_DISPATCH_CASE( FLOAT, 8, double )

#undef VIAME_DISPATCH_CASE

  std::ostringstream message;
  message << "unsupported pixel type " << traits;
  throw std::runtime_error( message.str() );
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_DISPATCH_H
