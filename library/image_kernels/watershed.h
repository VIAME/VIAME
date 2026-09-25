/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Marker-based watershed segmentation
///
/// What `cv::watershed` did. Meyer's flooding algorithm over a hierarchical
/// queue: every pixel that borders a labelled region is filed under the
/// colour distance across that border, and the queue is drained from the
/// smallest distance up, so a region grows through flat ground before it
/// climbs an edge. Where two regions meet, the pixel becomes a watershed
/// line rather than joining either.
///
/// The details are OpenCV's, because the interactive segmenter's output is
/// held to what it produced:
///
/// * the distance between two pixels is the **largest** of the three
///   per-channel absolute differences, not their sum or their norm;
/// * the queue has 256 buckets, one per distance, and a pixel enters at the
///   *smallest* distance to any labelled neighbour;
/// * neighbours are the four sharing an edge, never the diagonals;
/// * the one pixel border of the image is watershed line by definition, and
///   a negative marker on input is cleared to zero rather than honoured.

#ifndef VIAME_IMAGE_KERNELS_WATERSHED_H
#define VIAME_IMAGE_KERNELS_WATERSHED_H

#include <viame/core_types/image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

/// A pixel waiting in the flooding queue.
constexpr int32_t watershed_in_queue = -2;

/// A pixel the flooding decided is a boundary between two regions.
constexpr int32_t watershed_line = -1;

namespace detail {

/// One bucket of the hierarchical queue, as a singly linked span of indices.
struct watershed_bucket
{
  std::vector< size_t > entries;
  size_t next = 0;

  bool empty() const { return next >= entries.size(); }
};

/// The colour distance `cv::watershed` uses: the largest per-channel step.
template < typename T >
inline int
watershed_distance( viame::image_of< T > const& image,
                    size_t ax, size_t ay, size_t bx, size_t by )
{
  int worst = 0;

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    auto const a = static_cast< int >( image( ax, ay, plane ) );
    auto const b = static_cast< int >( image( bx, by, plane ) );
    worst = std::max( worst, std::abs( a - b ) );
  }

  return std::min( worst, 255 );
}

} // namespace detail

// ----------------------------------------------------------------------------
/// Flood \p markers over \p image, in place.
///
/// \p markers is one plane of `int32`: a positive value seeds a region, zero
/// is ground to be claimed, and on return the pixels where two regions met
/// carry `watershed_line`. Every seed label is preserved; nothing is
/// renumbered.
///
/// @param image the image whose edges stop the flooding, one or three planes
/// @param markers the seeds, modified in place
template < typename T >
void
watershed( viame::image_of< T > const& image,
           viame::image_of< int32_t >& markers )
{
  if( markers.depth() != 1 )
  {
    throw std::invalid_argument( "watershed markers are a single plane" );
  }

  if( markers.width() != image.width() ||
      markers.height() != image.height() )
  {
    throw std::invalid_argument(
      "watershed wants markers the size of the image" );
  }

  auto const width = image.width();
  auto const height = image.height();

  if( width < 3 || height < 3 )
  {
    return;
  }

  std::vector< detail::watershed_bucket > queue( 256 );

  auto const index_of = [ & ]( size_t x, size_t y ) { return y * width + x; };

  auto const push = [ & ]( int level, size_t x, size_t y )
  {
    queue[ static_cast< size_t >( level ) ].entries.push_back(
      index_of( x, y ) );
  };

  // The border is watershed line by definition, which is also what keeps the
  // four-neighbour reads below inside the image without a bounds test.
  for( size_t x = 0; x < width; ++x )
  {
    markers( x, 0, 0 ) = watershed_line;
    markers( x, height - 1, 0 ) = watershed_line;
  }

  for( size_t y = 0; y < height; ++y )
  {
    markers( 0, y, 0 ) = watershed_line;
    markers( width - 1, y, 0 ) = watershed_line;
  }

  // Seed the queue with every unlabelled pixel that touches a label, filed
  // under its smallest distance to one.
  for( size_t y = 1; y + 1 < height; ++y )
  {
    for( size_t x = 1; x + 1 < width; ++x )
    {
      auto& here = markers( x, y, 0 );

      if( here < 0 )
      {
        here = 0;
      }

      if( here != 0 )
      {
        continue;
      }

      int smallest = 256;

      auto const consider = [ & ]( size_t nx, size_t ny )
      {
        if( markers( nx, ny, 0 ) > 0 )
        {
          smallest = std::min(
            smallest, detail::watershed_distance( image, x, y, nx, ny ) );
        }
      };

      consider( x - 1, y );
      consider( x + 1, y );
      consider( x, y - 1 );
      consider( x, y + 1 );

      if( smallest < 256 )
      {
        push( smallest, x, y );
        here = watershed_in_queue;
      }
    }
  }

  // Drain from the smallest distance up. The level is allowed to move
  // **backwards**: a pixel pushed below the current level is nearer than
  // anything waiting, and OpenCV serves it first rather than finishing the
  // level it is on. Draining strictly forwards instead disagrees with
  // `cv::watershed` on about one pixel in eighty, all of them on a
  // boundary.
  size_t level = 0;

  while( level < queue.size() )
  {
    if( queue[ level ].empty() )
    {
      ++level;
      continue;
    }

    auto const at = queue[ level ].entries[ queue[ level ].next++ ];
    auto const x = at % width;
    auto const y = at / width;

    int32_t label = 0;
    bool boundary = false;

    auto const look = [ & ]( size_t nx, size_t ny )
    {
      auto const neighbour = markers( nx, ny, 0 );

      if( neighbour <= 0 )
      {
        return;
      }

      if( label == 0 )
      {
        label = neighbour;
      }
      else if( label != neighbour )
      {
        boundary = true;
      }
    };

    look( x - 1, y );
    look( x + 1, y );
    look( x, y - 1 );
    look( x, y + 1 );

    if( boundary || label == 0 )
    {
      markers( x, y, 0 ) = watershed_line;
      continue;
    }

    markers( x, y, 0 ) = label;

    // Anything still unclaimed beside it now borders a label
    auto const spread = [ & ]( size_t nx, size_t ny )
    {
      if( nx == 0 || ny == 0 || nx + 1 >= width || ny + 1 >= height )
      {
        return;
      }

      if( markers( nx, ny, 0 ) != 0 )
      {
        return;
      }

      auto const step = detail::watershed_distance( image, x, y, nx, ny );
      push( step, nx, ny );
      level = std::min( level, static_cast< size_t >( step ) );
      markers( nx, ny, 0 ) = watershed_in_queue;
    };

    spread( x - 1, y );
    spread( x + 1, y );
    spread( x, y - 1 );
    spread( x, y + 1 );
  }

  // Anything that never came out of the queue never reached a label
  for( size_t y = 0; y < height; ++y )
  {
    for( size_t x = 0; x < width; ++x )
    {
      if( markers( x, y, 0 ) == watershed_in_queue )
      {
        markers( x, y, 0 ) = watershed_line;
      }
    }
  }
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_WATERSHED_H
