/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of split image horizontally algorithm
 *
 * Was a `cv::Mat` region of interest and a clone; since P7-T04b it is
 * `image_ops::crop`, which does the same thing on a `vital::image`. The
 * bridge was asked for an `RGB_COLOR` mat both ways, so it never swapped a
 * channel and there was nothing here for it to do but copy.
 */

#include "split_image_horizontally.h"

#include <image_ops/dispatch.h>
#include <image_ops/resample.h>

#include <viame/core_types/image_container.h>

namespace io = viame::image_ops;

namespace viame {

namespace kv = kwiver::vital;

/// Split image
std::vector< kv::image_container_sptr >
split_image_horizontally
::split( kv::image_container_sptr image ) const
{
  std::vector< kv::image_container_sptr > output;

  auto const source = image->get_image();

  // An odd width loses its middle column, as the integer halving did.
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

  return output;
}

} // end namespace viame
