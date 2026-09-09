// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Simple implementation of a tiled image container.

#include <viame/core_types/tiled_image_container.h>
#include <viame/core_types/vital_types_export.h>

#include <map>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// A sparse tiled image container that holds existing tiles in memory.
class VITAL_TYPES_EXPORT simple_tiled_image_container
  : public tiled_image_container
{
public:
  /// Construct an empty container.
  ///
  /// \param tile_width Width of each tile in pixels.
  /// \param tile_height Height of each tile in pixels.
  /// \param grid_width Number of possible tiles in the X direction.
  /// \param grid_height Number of possible tiles in the Y direction.
  /// \param depth Number of channels in each pixel.
  /// \param pixel_traits Data format of each pixel channel.
  simple_tiled_image_container(
    size_t tile_width, size_t tile_height,
    size_t grid_width, size_t grid_height, size_t depth,
    image_pixel_traits const& pixel_traits = image_pixel_traits() );

  size_t depth() const override;

  image get_image() const override;

  bool has_tile( size_t x, size_t y ) const override;
  image_container_sptr get_tile( size_t x, size_t y ) const override;
  void set_tile( size_t x, size_t y, image_container_sptr const& tile );
  bool next_tile( size_t& x, size_t& y, bool first = false ) const override;

  size_t tile_grid_width() const override;
  size_t tile_grid_height() const override;
  size_t tile_count() const override;
  size_t tile_height() const override;
  size_t tile_width() const override;

private:
  size_t m_tile_width;
  size_t m_tile_height;
  size_t m_grid_width;
  size_t m_grid_height;
  size_t m_depth;
  image_pixel_traits m_pixel_traits;
  std::map< std::pair< size_t, size_t >, image_container_sptr > m_tiles;
};

} // namespace vital

} // namespace kwiver
