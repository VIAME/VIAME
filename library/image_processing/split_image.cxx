// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Split an image in half, left and right
///
/// Was `cv::Mat`'s region-of-interest and a clone; since P7-T04 it is
/// `image_kernels::crop`, which does the same thing on a `viame::image` and
/// leaves nothing for the OpenCV bridge to convert.

#include "split_image.h"

#include <image_kernels/dispatch.h>
#include <image_kernels/resample.h>

#include <viame/core_types/image_container.h>

using namespace viame;

namespace io = viame::image_kernels;

namespace viame {

namespace ocv {

/// Destructor
split_image
::~split_image()
{}

/// Split image
std::vector< viame::image_container_sptr >
split_image
::split( viame::image_container_sptr image ) const
{
  std::vector< viame::image_container_sptr > output;

  if( !image )
  {
    return output;
  }

  auto const source = image->get_image();

  // An odd width loses its middle column, which is what the integer halving
  // did before: both halves are `width / 2` wide.
  auto const half = source.width() / 2;

  for( size_t piece = 0; piece < 2; ++piece )
  {
    auto const cropped = io::dispatch_pixel_type(
      source,
      [ & ]( auto const& typed ) -> viame::image
      {
        return viame::image(
          io::crop( typed, piece * half, 0, half, source.height() ) );
      } );

    output.push_back(
      std::make_shared< viame::simple_image_container >( cropped ) );
  }

  return output;
}

} // namespace ocv

} // namespace viame
