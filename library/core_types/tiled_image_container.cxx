// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Implementation of simple tiled image container.

#include <viame/core_types/tiled_image_container.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
size_t
tiled_image_container
::width() const
{
  return tile_width() * tile_grid_width();
}

// ----------------------------------------------------------------------------
size_t
tiled_image_container
::height() const
{
  return tile_height() * tile_grid_height();
}

// ----------------------------------------------------------------------------
size_t
tiled_image_container
::size() const
{
  size_t result = 0;
  size_t x, y;
  bool first = true;
  while( next_tile( x, y, first ) )
  {
    result += get_tile( x, y )->size();
    first = false;
  }
  return result;
}

// ----------------------------------------------------------------------------
size_t
tiled_image_container
::tile_grid_size() const
{
  return tile_grid_width() * tile_grid_height();
}

} // namespace vital

} // namespace kwiver
