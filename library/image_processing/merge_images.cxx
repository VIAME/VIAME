// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Stack two images into one, plane after plane
///
/// Was `cv::split` on each and `cv::merge` on the concatenation; since
/// P7-T04 the planes are copied straight across, which is the same thing
/// once the OpenCV bridge is not interleaving and de-interleaving them on
/// the way.

#include "merge_images.h"

#include <image_ops/dispatch.h>

#include <viame/core_types/image_container.h>

#include <stdexcept>

using namespace kwiver::vital;

namespace io = viame::image_ops;

namespace kwiver {

namespace arrows {

namespace ocv {

/// Merge images
kwiver::vital::image_container_sptr
merge_images
::merge(
  kwiver::vital::image_container_sptr image1,
  kwiver::vital::image_container_sptr image2 ) const
{
  if( !image1 || !image2 )
  {
    return nullptr;
  }

  auto const first = image1->get_image();
  auto const second = image2->get_image();

  if( first.width() != second.width() || first.height() != second.height() )
  {
    throw std::runtime_error(
      "merge_images: the two images differ in size" );
  }

  auto const merged = io::dispatch_pixel_type(
    first,
    [ & ]( auto const& typed ) -> vital::image
    {
      using pixel_t = io::pixel_type_t< decltype( typed ) >;

      vital::image_of< pixel_t > other( second );

      vital::image_of< pixel_t > out( typed.width(), typed.height(),
                                      typed.depth() + other.depth() );

      for( size_t j = 0; j < typed.height(); ++j )
      {
        for( size_t i = 0; i < typed.width(); ++i )
        {
          size_t plane = 0;

          for( size_t p = 0; p < typed.depth(); ++p, ++plane )
          {
            out( i, j, plane ) = typed( i, j, p );
          }

          for( size_t p = 0; p < other.depth(); ++p, ++plane )
          {
            out( i, j, plane ) = other( i, j, p );
          }
        }
      }

      return vital::image( out );
    } );

  return std::make_shared< vital::simple_image_container >( merged );
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver
