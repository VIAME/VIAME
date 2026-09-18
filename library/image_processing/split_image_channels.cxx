// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Split an image into one image per plane
///
/// Was `cv::split`; since P7-T04 it copies the planes out directly, which on
/// a `viame::image` is what `cv::split` was doing anyway once the bridge had
/// interleaved them on the way in and the way out again.

#include "split_image_channels.h"

#include <image_kernels/dispatch.h>

#include <viame/core_types/image_container.h>

using namespace viame;

namespace io = viame::image_kernels;

namespace viame {

namespace ocv {

/// Destructor
split_image_channels
::~split_image_channels()
{}

/// Split image into its channel planes
std::vector< viame::image_container_sptr >
split_image_channels
::split( viame::image_container_sptr image ) const
{
  std::vector< viame::image_container_sptr > output;

  if( !image )
  {
    return output;
  }

  auto const source = image->get_image();

  for( size_t plane = 0; plane < source.depth(); ++plane )
  {
    auto const single = io::dispatch_pixel_type(
      source,
      [ & ]( auto const& typed ) -> viame::image
      {
        using pixel_t = io::pixel_type_t< decltype( typed ) >;

        viame::image_of< pixel_t > out( typed.width(), typed.height(), 1 );

        for( size_t j = 0; j < typed.height(); ++j )
        {
          for( size_t i = 0; i < typed.width(); ++i )
          {
            out( i, j, 0 ) = typed( i, j, plane );
          }
        }

        return viame::image( out );
      } );

    output.push_back(
      std::make_shared< viame::simple_image_container >( single ) );
  }

  return output;
}

} // namespace ocv

} // namespace viame
