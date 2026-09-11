// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of habcam split image horizontally algorithm
 *
 * Was a `cv::Mat` region of interest and a clone; since P7-T04b it is
 * `image_ops::crop`. The image passes straight through when it is not wide
 * enough to be a side-by-side pair, which is what makes this habcam's
 * rather than the plain horizontal split.
 */

#include "split_image_habcam.h"

#include <image_ops/dispatch.h>
#include <image_ops/resample.h>

#include <viame/core_types/image_container.h>

namespace io = viame::image_ops;

namespace kv = kwiver::vital;

namespace viame {

/// Split image
std::vector< kwiver::vital::image_container_sptr >
split_image_habcam
::split( kwiver::vital::image_container_sptr image ) const
{
  std::vector< kwiver::vital::image_container_sptr > output;

  if( image->width() >= c_required_width_factor * image->height() )
  {
    auto const source = image->get_image();
    auto const half = source.width() / 2;

    for( size_t piece = 0; piece < 2; ++piece )
    {
      auto const cropped = io::dispatch_pixel_type(
        source,
        [ & ]( auto const& typed ) -> kv::image
        {
          return kv::image(
            io::crop( typed, piece * half, 0, half, source.height() ) );
        } );

      output.push_back(
        std::make_shared< kv::simple_image_container >( cropped ) );
    }
  }
  else
  {
    output.push_back( image );
  }

  return output;
}

} // end namespace viame
