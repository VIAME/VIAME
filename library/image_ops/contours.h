/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Connected components, contour tracing, and what is measured on them
///
/// What `cv::connectedComponents`, `cv::findContours`, `cv::contourArea`,
/// `cv::boundingRect`, `cv::convexHull` and `cv::minAreaRect` did. These are
/// what turns a mask into detections, which is `detect_heat_map`,
/// `add_keypoints_from_mask`, the calibration target detector and the
/// ellipse proposal.
///
/// The contour is traced with Suzuki and Abe's border following -- the
/// algorithm `findContours` implements -- restricted to outer borders in
/// the `RETR_EXTERNAL` sense, which is the only mode VIAME asks for.
///
/// `morphology.h` already has the erode and dilate the masks are built with,
/// and `polygon.h` the rasterisation; neither traces a boundary.

#ifndef VIAME_IMAGE_OPS_CONTOURS_H
#define VIAME_IMAGE_OPS_CONTOURS_H

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// A pixel position, in the (i, j) order the images use.
struct point
{
  long i = 0;
  long j = 0;

  bool operator==( point const& other ) const
  {
    return i == other.i && j == other.j;
  }
};

/// An axis-aligned rectangle, half open: \p right and \p bottom are one past.
struct rect
{
  long left = 0;
  long top = 0;
  long right = 0;
  long bottom = 0;

  long width() const { return right - left; }
  long height() const { return bottom - top; }
  long area() const { return width() * height(); }

  bool empty() const { return width() <= 0 || height() <= 0; }
};

/// A rectangle at an angle, which is what `cv::minAreaRect` returns.
struct rotated_rect
{
  double centre_i = 0.0;
  double centre_j = 0.0;
  double width = 0.0;
  double height = 0.0;
  /// Counter-clockwise from the i axis, in degrees.
  double angle = 0.0;

  double area() const { return width * height; }
};

// ----------------------------------------------------------------------------
/// How a component's neighbours are counted.
///
/// `FOUR` joins pixels sharing an edge, `EIGHT` also those sharing a corner.
/// `cv::connectedComponents` defaults to eight; `cv::findContours` traces
/// eight-connected outer borders of four-connected regions, which is the
/// pairing this keeps.
enum class connectivity
{
  FOUR,
  EIGHT,
};

// ----------------------------------------------------------------------------
/// Label each connected run of non-zero pixels, which is
/// `cv::connectedComponents`.
///
/// Background is 0 and components are numbered from 1 in raster order of
/// their first pixel, which is what OpenCV's default (SAUF) labelling gives.
///
/// @param mask one plane; anything non-zero is foreground
/// @param how which neighbours join
/// @param[out] count how many components were found, background excluded
template < typename T >
kwiver::vital::image_of< int32_t >
label_components( kwiver::vital::image_of< T > const& mask,
                  connectivity how, size_t& count )
{
  if( mask.depth() != 1 )
  {
    throw std::invalid_argument( "label_components takes a single plane" );
  }

  auto const width = mask.width();
  auto const height = mask.height();

  kwiver::vital::image_of< int32_t > labels( width, height, 1 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      labels( i, j, 0 ) = 0;
    }
  }

  count = 0;

  // A union-find over provisional labels, resolved in a second pass. Two
  // passes rather than one flood fill per component: a flood fill recurses
  // as deep as a component is long, and a mask of a horizon is one component
  // the width of the frame.
  std::vector< int32_t > parent{ 0 };

  auto find = [ &parent ]( int32_t label )
  {
    while( parent[ label ] != label )
    {
      parent[ label ] = parent[ parent[ label ] ];
      label = parent[ label ];
    }
    return label;
  };

  auto join = [ &parent, &find ]( int32_t a, int32_t b )
  {
    a = find( a );
    b = find( b );

    if( a != b )
    {
      parent[ std::max( a, b ) ] = std::min( a, b );
    }
  };

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      if( mask( i, j, 0 ) == T{} )
      {
        continue;
      }

      // The already-labelled neighbours: west, north, and for eight
      // connectivity the two northern diagonals
      std::vector< int32_t > around;

      auto consider = [ & ]( long di, long dj )
      {
        auto const x = static_cast< long >( i ) + di;
        auto const y = static_cast< long >( j ) + dj;

        if( x < 0 || y < 0 || x >= static_cast< long >( width ) ||
            y >= static_cast< long >( height ) )
        {
          return;
        }

        auto const label =
          labels( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 );

        if( label != 0 )
        {
          around.push_back( label );
        }
      };

      consider( -1, 0 );
      consider( 0, -1 );

      if( how == connectivity::EIGHT )
      {
        consider( -1, -1 );
        consider( 1, -1 );
      }

      if( around.empty() )
      {
        auto const fresh = static_cast< int32_t >( parent.size() );
        parent.push_back( fresh );
        labels( i, j, 0 ) = fresh;
      }
      else
      {
        auto const smallest = *std::min_element( around.begin(),
                                                 around.end() );
        labels( i, j, 0 ) = smallest;

        for( auto const other : around )
        {
          join( smallest, other );
        }
      }
    }
  }

  // Resolve, and renumber so the labels run 1..count in raster order
  std::vector< int32_t > final_label( parent.size(), 0 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      auto const provisional = labels( i, j, 0 );

      if( provisional == 0 )
      {
        continue;
      }

      auto const root = find( provisional );

      if( final_label[ root ] == 0 )
      {
        final_label[ root ] = static_cast< int32_t >( ++count );
      }

      labels( i, j, 0 ) = final_label[ root ];
    }
  }

  return labels;
}

// ----------------------------------------------------------------------------
/// The outer boundary of every component, traced as `cv::findContours` with
/// `RETR_EXTERNAL` and `CHAIN_APPROX_NONE` traces it.
///
/// Every boundary pixel, in order, counter-clockwise in image coordinates --
/// which looks clockwise on screen because j runs downwards. A single pixel
/// component is a contour of one point.
///
/// The contours come out in the order their first pixel is met in a raster
/// scan, which is OpenCV's order too.
template < typename T >
std::vector< std::vector< point > >
find_contours( kwiver::vital::image_of< T > const& mask )
{
  if( mask.depth() != 1 )
  {
    throw std::invalid_argument( "find_contours takes a single plane" );
  }

  // One outer boundary per connected component, which is what
  // `RETR_EXTERNAL` means. Starting a trace wherever a foreground pixel has
  // a background pixel to its west -- the textual Suzuki and Abe condition
  // for an outer border -- also starts one on the inside of a ring, because
  // the pixel to the right of the hole satisfies it too. Distinguishing the
  // two needs the nesting level Suzuki and Abe carry; going through the
  // components instead gets `RETR_EXTERNAL` directly, since a component has
  // exactly one outer boundary.
  size_t count = 0;
  auto const labels = label_components( mask, connectivity::EIGHT, count );

  auto const width = static_cast< long >( mask.width() );
  auto const height = static_cast< long >( mask.height() );

  auto in_component = [ & ]( long i, long j, int32_t label )
  {
    if( i < 0 || j < 0 || i >= width || j >= height )
    {
      return false;
    }

    return labels( static_cast< size_t >( i ), static_cast< size_t >( j ), 0 )
           == label;
  };

  // The eight neighbours, counter-clockwise from due east
  static constexpr long around_i[ 8 ] = { 1, 1, 0, -1, -1, -1, 0, 1 };
  static constexpr long around_j[ 8 ] = { 0, 1, 1, 1, 0, -1, -1, -1 };

  // The first pixel of each component in raster order, which is the topmost
  // of its leftmost -- so its western neighbour is outside the component and
  // the trace can start heading east
  std::vector< point > starts( count + 1, point{ -1, -1 } );

  for( long j = 0; j < height; ++j )
  {
    for( long i = 0; i < width; ++i )
    {
      auto const label =
        labels( static_cast< size_t >( i ), static_cast< size_t >( j ), 0 );

      if( label != 0 && starts[ label ].i < 0 )
      {
        starts[ label ] = point{ i, j };
      }
    }
  }

  std::vector< std::vector< point > > contours;

  for( size_t label = 1; label <= count; ++label )
  {
    auto const start = starts[ label ];

    if( start.i < 0 )
    {
      continue;
    }

    std::vector< point > contour;
    point here = start;

    int from = 4;   // west, which is outside the component at the start
    point second{ -1, -1 };
    bool have_second = false;

    for( ;; )
    {
      contour.push_back( here );

      int direction = -1;

      for( int step = 1; step <= 8; ++step )
      {
        auto const candidate = ( from + step ) % 8;

        if( in_component( here.i + around_i[ candidate ],
                          here.j + around_j[ candidate ],
                          static_cast< int32_t >( label ) ) )
        {
          direction = candidate;
          break;
        }
      }

      if( direction < 0 )
      {
        break;   // an isolated pixel
      }

      point next{ here.i + around_i[ direction ],
                  here.j + around_j[ direction ] };

      if( !have_second )
      {
        second = next;
        have_second = true;
      }
      else if( here == start && next == second )
      {
        // Back at the beginning heading the same way: the loop is closed.
        // Jacob's stopping criterion; stopping merely on reaching the start
        // again cuts a contour short wherever it touches itself.
        break;
      }

      // Came from the opposite side of the step just taken
      from = ( direction + 4 ) % 8;
      here = next;
    }

    contours.push_back( std::move( contour ) );
  }

  return contours;
}

// ----------------------------------------------------------------------------
/// The smallest rectangle containing \p contour, which is `cv::boundingRect`.
inline rect
bounding_rect( std::vector< point > const& contour )
{
  rect out;

  if( contour.empty() )
  {
    return out;
  }

  out.left = contour[ 0 ].i;
  out.top = contour[ 0 ].j;
  out.right = contour[ 0 ].i + 1;
  out.bottom = contour[ 0 ].j + 1;

  for( auto const& at : contour )
  {
    out.left = std::min( out.left, at.i );
    out.top = std::min( out.top, at.j );
    out.right = std::max( out.right, at.i + 1 );
    out.bottom = std::max( out.bottom, at.j + 1 );
  }

  return out;
}

// ----------------------------------------------------------------------------
/// The area \p contour encloses, which is `cv::contourArea`.
///
/// The shoelace formula over the boundary vertices, so it is the area of the
/// polygon through the pixel centres rather than the count of pixels inside
/// it. Those differ: a three by three square traced at its centres is two by
/// two. That is `cv::contourArea`'s answer too, and the difference is why
/// `detect_heat_map` filtering on area and a caller counting mask pixels do
/// not agree.
///
/// Unsigned, as OpenCV's is without `oriented`.
inline double
contour_area( std::vector< point > const& contour )
{
  if( contour.size() < 3 )
  {
    return 0.0;
  }

  double twice = 0.0;

  for( size_t at = 0; at < contour.size(); ++at )
  {
    auto const& a = contour[ at ];
    auto const& b = contour[ ( at + 1 ) % contour.size() ];

    twice += static_cast< double >( a.i ) * static_cast< double >( b.j ) -
             static_cast< double >( b.i ) * static_cast< double >( a.j );
  }

  return std::abs( twice ) / 2.0;
}

// ----------------------------------------------------------------------------
/// The convex hull of \p points, counter-clockwise, which is
/// `cv::convexHull`.
///
/// Andrew's monotone chain. Collinear points are dropped, as OpenCV's is
/// without `returnPoints` false.
inline std::vector< point >
convex_hull( std::vector< point > points )
{
  if( points.size() < 3 )
  {
    std::sort( points.begin(), points.end(),
               []( point const& a, point const& b )
               { return a.i != b.i ? a.i < b.i : a.j < b.j; } );
    points.erase( std::unique( points.begin(), points.end() ),
                  points.end() );
    return points;
  }

  std::sort( points.begin(), points.end(),
             []( point const& a, point const& b )
             { return a.i != b.i ? a.i < b.i : a.j < b.j; } );
  points.erase( std::unique( points.begin(), points.end() ), points.end() );

  auto turn = []( point const& o, point const& a, point const& b )
  {
    return ( a.i - o.i ) * ( b.j - o.j ) - ( a.j - o.j ) * ( b.i - o.i );
  };

  std::vector< point > hull( 2 * points.size() );
  size_t at = 0;

  for( auto const& p : points )
  {
    while( at >= 2 && turn( hull[ at - 2 ], hull[ at - 1 ], p ) <= 0 )
    {
      --at;
    }
    hull[ at++ ] = p;
  }

  auto const lower = at + 1;

  for( size_t index = points.size() - 1; index > 0; --index )
  {
    auto const& p = points[ index - 1 ];

    while( at >= lower && turn( hull[ at - 2 ], hull[ at - 1 ], p ) <= 0 )
    {
      --at;
    }
    hull[ at++ ] = p;
  }

  hull.resize( at > 0 ? at - 1 : 0 );
  return hull;
}

// ----------------------------------------------------------------------------
/// The smallest rectangle of any orientation containing \p points, which is
/// `cv::minAreaRect`.
///
/// Rotating calipers over the convex hull: the minimum-area rectangle has a
/// side flush with a hull edge, so trying every edge is enough.
inline rotated_rect
min_area_rect( std::vector< point > const& points )
{
  rotated_rect out;

  auto const hull = convex_hull( points );

  if( hull.empty() )
  {
    return out;
  }

  if( hull.size() == 1 )
  {
    out.centre_i = static_cast< double >( hull[ 0 ].i );
    out.centre_j = static_cast< double >( hull[ 0 ].j );
    return out;
  }

  double best_area = -1.0;

  for( size_t at = 0; at < hull.size(); ++at )
  {
    auto const& a = hull[ at ];
    auto const& b = hull[ ( at + 1 ) % hull.size() ];

    auto const edge_i = static_cast< double >( b.i - a.i );
    auto const edge_j = static_cast< double >( b.j - a.j );
    auto const length = std::hypot( edge_i, edge_j );

    if( length == 0.0 )
    {
      continue;
    }

    auto const ui = edge_i / length;
    auto const uj = edge_j / length;

    double min_along = 0.0;
    double max_along = 0.0;
    double min_across = 0.0;
    double max_across = 0.0;
    bool first = true;

    for( auto const& p : hull )
    {
      auto const di = static_cast< double >( p.i - a.i );
      auto const dj = static_cast< double >( p.j - a.j );

      auto const along = di * ui + dj * uj;
      auto const across = -di * uj + dj * ui;

      if( first )
      {
        min_along = max_along = along;
        min_across = max_across = across;
        first = false;
      }
      else
      {
        min_along = std::min( min_along, along );
        max_along = std::max( max_along, along );
        min_across = std::min( min_across, across );
        max_across = std::max( max_across, across );
      }
    }

    auto const w = max_along - min_along;
    auto const h = max_across - min_across;
    auto const area = w * h;

    if( best_area < 0.0 || area < best_area )
    {
      best_area = area;

      auto const centre_along = ( min_along + max_along ) / 2.0;
      auto const centre_across = ( min_across + max_across ) / 2.0;

      out.centre_i = static_cast< double >( a.i ) + centre_along * ui -
                     centre_across * uj;
      out.centre_j = static_cast< double >( a.j ) + centre_along * uj +
                     centre_across * ui;
      out.width = w;
      out.height = h;
      out.angle = std::atan2( uj, ui ) * 180.0 / 3.14159265358979323846;
    }
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
