// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Container interface for tiled images.

#ifndef VITAL_TILED_IMAGE_CONTAINER_H_
#define VITAL_TILED_IMAGE_CONTAINER_H_

#include <viame/core_types/image_container.h>
#include <viame/core_types/vital_types_export.h>

#include <optional>
#include <vector>

#include <cstdint>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Container for an image segmented into a regular grid of tiles.
///
/// This class holds a grid of sub-images (tiles) aligned to a grid, which
/// together form the full image. The full image may be sparse, i.e. any
/// particular location in the grid may contain no pixel data. All tiles must
/// have the same dimensions and pixel format.
///
/// This class is intended to allow handling of very large images by not
/// necessarily requiring all image data to be loaded into memory at once, or
/// indeed to exist at all. The image is accessed on a tile-by-tile basis,
/// allowing implementations to load images piecemeal from disk, over the
/// network, or on-demand from some other image construction method.
///
/// As the most common use case for \c tiled_image_container is to hold images
/// too large to fit into memory all at once, the \c get_image() function of any
/// implementation classes may return an empty image (with appropriate pixel
/// traits set, but no pixel data).
class VITAL_TYPES_EXPORT tiled_image_container
  : public image_container
{
public:
  /// Return the total image width in pixels.
  ///
  /// The returned value will always be an exact multiple of the tile width. It
  /// does not depend on whether any tiles actually exist, merely indicating the
  /// capacity of the image.
  size_t width() const override final;

  /// Return the total image height in pixels.
  ///
  /// The returned value will always be an exact multiple of the tile height. It
  /// does not depend on whether any tiles actually exist, merely indicating the
  /// capacity of the image.
  size_t height() const override final;

  /// Return the combined size of all existing tiles in bytes.
  size_t size() const override;

  /// Return \c true if a tile exists at grid location \p x, \p y.
  virtual bool has_tile( size_t x, size_t y ) const = 0;

  /// Return the tile at grid location \p x, \p y.
  ///
  /// This function returns \c nullptr if no such tile exists.
  virtual vital::image_container_sptr get_tile( size_t x, size_t y ) const = 0;

  /// Update \p x, \p y to the location of the next existing tile.
  ///
  /// \note No particular order of iteration is guaranteed.
  ///
  /// \param[in, out] x
  ///   X coordinate of the current tile, set to the X coordinate of the next
  ///   tile if this function returns \c true. The passed value is ignored if
  ///   \p first is \c true.
  /// \param[in, out] y
  ///   Y coordinate of the current tile, set to the Y coordinate of the next
  ///   tile if this function returns \c true. The passed value is ignored if
  ///   \p first is \c true.
  /// \param first
  ///   If \c true, \p x and \p y will be set to the location of the first tile.
  ///
  /// \return
  ///   \c true if a next tile has been found, \c false if there are no more
  ///   tiles, in which case \p x and \p y are not modified.
  virtual bool next_tile( size_t& x, size_t& y, bool first = false ) const = 0;

  /// Return the number of possibly-existing tiles in the X direction.
  virtual size_t tile_grid_width() const = 0;

  /// Return the number of possibly-existing tiles in the Y direction.
  virtual size_t tile_grid_height() const = 0;

  /// Return the total number of possibly-existing tiles.
  size_t tile_grid_size() const;

  /// Return the total number of actually-existing tiles.
  virtual size_t tile_count() const = 0;

  /// Return the height of a tile in pixels.
  virtual size_t tile_height() const = 0;

  /// Return the width of a tile in pixels.
  virtual size_t tile_width() const = 0;
};

using tiled_image_container_sptr = std::shared_ptr< tiled_image_container >;

} // namespace vital

} // namespace kwiver

#endif
