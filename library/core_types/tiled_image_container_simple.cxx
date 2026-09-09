// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Simple implementation of a tiled image container.

#include <viame/core_types/tiled_image_container_simple.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
simple_tiled_image_container
::simple_tiled_image_container(
  size_t tile_width, size_t tile_height,
  size_t grid_width, size_t grid_height, size_t depth,
  image_pixel_traits const& pixel_traits )
  : m_tile_width{ tile_width },
    m_tile_height{ tile_height },
    m_grid_width{ grid_width },
    m_grid_height{ grid_height },
    m_depth{ depth },
    m_pixel_traits{ pixel_traits },
    m_tiles{}
{
  if( !m_tile_width || !m_tile_height ||
      !m_grid_width || !m_grid_height || !m_depth )
  {
    throw std::runtime_error( "Tiled image dimensions cannot be zero" );
  }
}

// ----------------------------------------------------------------------------
size_t
simple_tiled_image_container
::depth() const
{
  return m_depth;
}

// ----------------------------------------------------------------------------
image
simple_tiled_image_container
::get_image() const
{
  // Return empty image, but still assign pixel traits. This is the only way to
  // access pixel traits externally.
  return image{ m_pixel_traits };
}

// ----------------------------------------------------------------------------
bool
simple_tiled_image_container
::has_tile( size_t x, size_t y ) const
{
  if( x >= m_grid_width || y >= m_grid_height )
  {
    throw std::runtime_error( "Tile coordinates out of bounds" );
  }

  return m_tiles.count( std::make_pair( y, x ) );
}

// ----------------------------------------------------------------------------
image_container_sptr
simple_tiled_image_container
::get_tile( size_t x, size_t y ) const
{
  if( x >= m_grid_width || y >= m_grid_height )
  {
    throw std::runtime_error( "Tile coordinates out of bounds" );
  }

  if( auto const it = m_tiles.find( std::make_pair( y, x ) );
      it != m_tiles.end() )
  {
    return it->second;
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
void
simple_tiled_image_container
::set_tile( size_t x, size_t y, image_container_sptr const& tile )
{
  if( x >= m_grid_width || y >= m_grid_height )
  {
    throw std::runtime_error( "Tile coordinates out of bounds" );
  }

  if( !tile )
  {
    m_tiles.erase( std::make_pair( y, x ) );
    return;
  }

  if( m_tile_width != tile->width() || m_tile_height != tile->height() ||
      m_depth != tile->depth() )
  {
    throw std::runtime_error( "Tile has incorrect dimensions" );
  }

  if( m_pixel_traits != tile->get_image().pixel_traits() )
  {
    throw std::runtime_error( "Tile has incorrect pixel traits" );
  }

  auto const [ it, inserted ] =
    m_tiles.try_emplace( std::make_pair( y, x ), tile );
  if( !inserted )
  {
    it->second = tile;
  }
}

// ----------------------------------------------------------------------------
bool
simple_tiled_image_container
::next_tile( size_t& x, size_t& y, bool first ) const
{
  if( first )
  {
    if( m_tiles.empty() )
    {
      return false;
    }

    auto const it = m_tiles.begin();
    y = it->first.first;
    x = it->first.second;
    return true;
  }

  if( auto const it = m_tiles.upper_bound( std::make_pair( y, x ) );
      it != m_tiles.end() )
  {
    y = it->first.first;
    x = it->first.second;
    return true;
  }

  return false;
}

// ----------------------------------------------------------------------------
size_t
simple_tiled_image_container
::tile_grid_width() const
{
  return m_grid_width;
}

// ----------------------------------------------------------------------------
size_t
simple_tiled_image_container
::tile_grid_height() const
{
  return m_grid_height;
}

// ----------------------------------------------------------------------------
size_t
simple_tiled_image_container
::tile_count() const
{
  return m_tiles.size();
}

// ----------------------------------------------------------------------------
size_t
simple_tiled_image_container
::tile_height() const
{
  return m_tile_height;
}

// ----------------------------------------------------------------------------
size_t
simple_tiled_image_container
::tile_width() const
{
  return m_tile_width;
}

} // namespace vital

} // namespace kwiver
