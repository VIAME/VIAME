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
#include <array>
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

  /// The four corners, in perimeter order.
  ///
  /// `cv::RotatedRect::points`' formula, applied to these parameters. It
  /// gives the same four points OpenCV's does, since the rectangle is the
  /// same rectangle, but it may start at a different corner or go the other
  /// way round: OpenCV normalises its angle into [0, 90) and swaps width
  /// and height to suit, and `min_area_rect` here keeps the edge it found.
  ///
  /// Adjacent entries are adjacent corners, which is what every caller
  /// needs -- "which pair of opposite edges is the longer" and "the edge
  /// midpoint furthest left" are both answered from any perimeter order.
  std::array< std::pair< double, double >, 4 > corners() const
  {
    auto const radians = angle * 3.14159265358979323846 / 180.0;
    auto const b = std::cos( radians ) * 0.5;
    auto const a = std::sin( radians ) * 0.5;

    std::array< std::pair< double, double >, 4 > out;

    out[ 0 ] = { centre_i - a * height - b * width,
                 centre_j + b * height - a * width };
    out[ 1 ] = { centre_i + a * height - b * width,
                 centre_j - b * height - a * width };
    out[ 2 ] = { 2.0 * centre_i - out[ 0 ].first,
                 2.0 * centre_j - out[ 0 ].second };
    out[ 3 ] = { 2.0 * centre_i - out[ 1 ].first,
                 2.0 * centre_j - out[ 1 ].second };

    return out;
  }
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
/// The contours come out in **reverse** raster order of their first pixel,
/// which is the order `cv::findContours` returns them in. It is worth
/// matching rather than sorting afterwards: a caller that takes the first
/// few, or that writes them out in order, sees a different answer otherwise,
/// and `detect_heat_map` is one.
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

  // Reverse raster order, as `cv::findContours` gives
  std::reverse( contours.begin(), contours.end() );

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

    // `<=`, not `<`: the **last** edge achieving the minimum wins, which is
    // what `cv::rotatingCalipers` does. It decides nothing when there is one
    // minimum and everything when there are several, and several is the
    // normal case -- an ellipse's minimum-area rectangle is achieved at four
    // orientations exactly. Keeping the first instead put the keypoints of
    // `add_keypoints_from_mask` six pixels away from where OpenCV put them.
    if( best_area < 0.0 || area <= best_area )
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

// ----------------------------------------------------------------------------
/// One traced border and where it sits in the nesting.
struct border
{
  std::vector< point > points;

  /// True for a hole -- an inner border, traced on the foreground pixels
  /// that surround a background region.
  bool is_hole = false;
};

// ----------------------------------------------------------------------------
/// Every border of every component, outer and hole, which is
/// `cv::findContours` under `RETR_CCOMP` and `CHAIN_APPROX_NONE`.
///
/// Suzuki and Abe's border following proper, rather than the
/// `RETR_EXTERNAL` shortcut `find_contours` takes: the raster scan starts an
/// **outer** border at a foreground pixel whose western neighbour is
/// background and a **hole** border at a foreground pixel whose eastern
/// neighbour is background, and the already-traced pixels are marked so that
/// neither is started twice.
///
/// The order is OpenCV's, and it is not the scan order. OpenCV builds a tree
/// whose siblings are **prepended** and then flattens it depth first, so a
/// top-level border comes out after the ones found later in the scan, and
/// each border's holes follow it immediately. `blobs` in
/// `tests/plugins/core/mask_polygon_csv.txt` is what pins the first half of
/// that and `ring` the second.
template < typename T >
std::vector< border >
find_borders( kwiver::vital::image_of< T > const& mask )
{
  if( mask.depth() != 1 )
  {
    throw std::invalid_argument( "find_borders takes a single plane" );
  }

  auto const width = static_cast< long >( mask.width() );
  auto const height = static_cast< long >( mask.height() );

  // Suzuki and Abe's labelled image: 0 background, 1 an untouched foreground
  // pixel, and otherwise plus or minus the border number that claimed it.
  std::vector< int > labels(
    static_cast< size_t >( width * height ), 0 );

  auto at = [ & ]( long i, long j ) -> int&
  {
    return labels[ static_cast< size_t >( j * width + i ) ];
  };

  auto value = [ & ]( long i, long j ) -> int
  {
    if( i < 0 || j < 0 || i >= width || j >= height )
    {
      return 0;
    }

    return at( i, j );
  };

  for( long j = 0; j < height; ++j )
  {
    for( long i = 0; i < width; ++i )
    {
      at( i, j ) =
        mask( static_cast< size_t >( i ), static_cast< size_t >( j ), 0 )
          ? 1 : 0;
    }
  }

  // The eight neighbours, counter-clockwise from due east
  static constexpr long around_i[ 8 ] = { 1, 1, 0, -1, -1, -1, 0, 1 };
  static constexpr long around_j[ 8 ] = { 0, 1, 1, 1, 0, -1, -1, -1 };

  // The two-level tree. `parent` is -1 for a top-level border; `children`
  // keeps each parent's holes in the order the flattening wants.
  std::vector< border > found;
  std::vector< int > parent_of;
  std::vector< std::vector< int > > children;

  // `border_of[ label ]` is the index in `found` of the border that wrote
  // `label`, so a hole can find the outer border it belongs to. Indexed by
  // the label itself, so the two entries that are not borders -- 0 for
  // background and 1 for untouched foreground -- have to be there.
  std::vector< int > border_of( 2, -1 );

  int next_label = 1;

  for( long j = 0; j < height; ++j )
  {
    // The last border number seen on this scan line, which is how Suzuki and
    // Abe decide a new border's parent.
    int last_label = 1;

    for( long i = 0; i < width; ++i )
    {
      int const here = at( i, j );

      if( here == 0 )
      {
        last_label = 1;
        continue;
      }

      bool outer = false;
      bool hole = false;

      if( here == 1 && value( i - 1, j ) == 0 )
      {
        outer = true;
      }
      else if( here >= 1 && value( i + 1, j ) == 0 )
      {
        hole = true;
      }

      if( !outer && !hole )
      {
        if( here != 1 )
        {
          last_label = here > 0 ? here : -here;
        }
        continue;
      }

      ++next_label;

      // The parent, by Suzuki and Abe's table: an outer border's parent is
      // the hole border that last claimed a pixel on this line, and a hole
      // border's parent is the outer border that did.
      int const previous = last_label;
      int previous_index =
        ( previous >= 2 && previous < static_cast< int >( border_of.size() ) )
          ? border_of[ static_cast< size_t >( previous ) ] : -1;

      int my_parent = -1;

      if( previous_index >= 0 )
      {
        if( found[ static_cast< size_t >( previous_index ) ].is_hole != hole )
        {
          my_parent = previous_index;
        }
        else
        {
          my_parent = parent_of[ static_cast< size_t >( previous_index ) ];
        }
      }

      // Trace. `from` is the direction of the pixel the border was entered
      // from: west for an outer border, east for a hole.
      long from_i = hole ? i + 1 : i - 1;
      long from_j = j;

      std::vector< point > points;
      point start{ i, j };
      point here_point = start;
      point second{ -1, -1 };
      bool have_second = false;
      bool closed = false;

      auto direction_of = [ & ]( point const& centre, long other_i,
                                 long other_j )
      {
        for( int d = 0; d < 8; ++d )
        {
          if( centre.i + around_i[ d ] == other_i &&
              centre.j + around_j[ d ] == other_j )
          {
            return d;
          }
        }

        return 4;
      };

      int from = direction_of( here_point, from_i, from_j );

      for( ;; )
      {
        points.push_back( here_point );

        int direction = -1;
        int examined_east = 0;

        // **Clockwise**, which is the direction Suzuki and Abe specify and
        // the direction `cv::findContours` comes out in.
        // `find_contours` above searches the other way; it predates this and
        // its callers are held to what it gives.
        for( int step = 1; step <= 8; ++step )
        {
          int const candidate = ( from + 8 - step ) % 8;
          long const ni = here_point.i + around_i[ candidate ];
          long const nj = here_point.j + around_j[ candidate ];

          if( value( ni, nj ) != 0 )
          {
            direction = candidate;
            break;
          }

          // Suzuki and Abe mark a pixel negative when the search passed over
          // its eastern neighbour and found background there: that is what
          // stops the raster scan restarting the same border from inside.
          if( candidate == 0 )
          {
            examined_east = 1;
          }
        }

        int& label_here = at( here_point.i, here_point.j );

        if( examined_east )
        {
          label_here = -next_label;
        }
        else if( label_here == 1 )
        {
          label_here = next_label;
        }

        if( direction < 0 )
        {
          break;   // an isolated pixel
        }

        point const next{ here_point.i + around_i[ direction ],
                          here_point.j + around_j[ direction ] };

        if( !have_second )
        {
          second = next;
          have_second = true;
        }
        else if( here_point == start && next == second )
        {
          closed = true;
          break;
        }

        from = ( direction + 4 ) % 8;
        here_point = next;
      }

      // Jacob's criterion fires one step late: the start has already been
      // pushed a second time by the time it is checked. Dropping it is not
      // cosmetic -- it is the difference between the start being the middle
      // of a straight run, which `simplify_chain` removes, and being a
      // corner, which it keeps.
      if( closed && points.size() > 1 )
      {
        points.pop_back();
      }

      found.push_back( border{ std::move( points ), hole } );
      parent_of.push_back( my_parent );
      children.emplace_back();
      border_of.push_back( static_cast< int >( found.size() ) - 1 );

      if( my_parent >= 0 )
      {
        children[ static_cast< size_t >( my_parent ) ].insert(
          children[ static_cast< size_t >( my_parent ) ].begin(),
          static_cast< int >( found.size() ) - 1 );
      }

      last_label = next_label;
    }
  }

  // Flatten: top-level borders in reverse discovery order, each followed by
  // its children, which were prepended as they were found.
  std::vector< int > roots;

  for( size_t index = 0; index < found.size(); ++index )
  {
    if( parent_of[ index ] < 0 )
    {
      roots.insert( roots.begin(), static_cast< int >( index ) );
    }
  }

  std::vector< border > out;

  for( auto const root : roots )
  {
    out.push_back( found[ static_cast< size_t >( root ) ] );

    for( auto const child : children[ static_cast< size_t >( root ) ] )
    {
      out.push_back( found[ static_cast< size_t >( child ) ] );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Drop the points inside a straight run, which is `CHAIN_APPROX_SIMPLE`.
///
/// A traced border has every pixel; OpenCV's simple chain code keeps only
/// the ends of each horizontal, vertical or diagonal segment. The closed
/// contour is treated as such: the first point is kept when the direction
/// into it differs from the direction out of it.
inline std::vector< point >
simplify_chain( std::vector< point > const& contour )
{
  if( contour.size() < 3 )
  {
    return contour;
  }

  auto const step = []( point const& from, point const& to )
  {
    return point{ to.i - from.i, to.j - from.j };
  };

  std::vector< point > out;

  for( size_t index = 0; index < contour.size(); ++index )
  {
    auto const& previous = contour[ ( index + contour.size() - 1 ) %
                                    contour.size() ];
    auto const& current = contour[ index ];
    auto const& next = contour[ ( index + 1 ) % contour.size() ];

    if( !( step( previous, current ) == step( current, next ) ) )
    {
      out.push_back( current );
    }
  }

  return out.empty() ? contour : out;
}

// ----------------------------------------------------------------------------
/// Douglas-Peucker over a closed contour, which is `cv::approxPolyDP` with
/// `closed` true.
///
/// OpenCV starts from the two points furthest apart along the contour --
/// the one furthest from point zero, and then the one furthest from that --
/// and recurses on each half.
inline std::vector< point >
approx_poly( std::vector< point > const& contour, double epsilon )
{
  auto const count = contour.size();

  if( count < 3 )
  {
    return contour;
  }

  auto const distance_squared = []( point const& a, point const& b )
  {
    double const di = static_cast< double >( a.i - b.i );
    double const dj = static_cast< double >( a.j - b.j );
    return di * di + dj * dj;
  };

  // The two starting vertices, as OpenCV picks them
  size_t first = 0;
  double worst = -1.0;

  for( size_t index = 0; index < count; ++index )
  {
    double const d = distance_squared( contour[ 0 ], contour[ index ] );

    if( d > worst )
    {
      worst = d;
      first = index;
    }
  }

  size_t second = 0;
  worst = -1.0;

  for( size_t index = 0; index < count; ++index )
  {
    double const d = distance_squared( contour[ first ], contour[ index ] );

    if( d > worst )
    {
      worst = d;
      second = index;
    }
  }

  // The perpendicular distance of a point from the line through two others,
  // times the length of that line -- which is what OpenCV compares against
  // `epsilon * length`, so the square root is taken once at the end.
  auto const deviation = []( point const& p, point const& a, point const& b )
  {
    double const di = static_cast< double >( b.i - a.i );
    double const dj = static_cast< double >( b.j - a.j );
    double const pi = static_cast< double >( p.i - a.i );
    double const pj = static_cast< double >( p.j - a.j );

    return std::abs( pi * dj - pj * di );
  };

  std::vector< bool > keep( count, false );
  keep[ first ] = true;
  keep[ second ] = true;

  // Each arc between two kept vertices, walked forward around the loop
  std::vector< std::pair< size_t, size_t > > pending;
  pending.emplace_back( first, second );
  pending.emplace_back( second, first );

  while( !pending.empty() )
  {
    auto const arc = pending.back();
    pending.pop_back();

    auto const& a = contour[ arc.first ];
    auto const& b = contour[ arc.second ];

    double const length = std::sqrt( distance_squared( a, b ) );
    double const limit = epsilon * ( length > 0.0 ? length : 1.0 );

    size_t furthest = arc.first;
    double most = -1.0;

    for( size_t offset = 1;; ++offset )
    {
      size_t const index = ( arc.first + offset ) % count;

      if( index == arc.second )
      {
        break;
      }

      double const d = ( length > 0.0 )
        ? deviation( contour[ index ], a, b )
        : std::sqrt( distance_squared( contour[ index ], a ) );

      if( d > most )
      {
        most = d;
        furthest = index;
      }
    }

    if( most > limit )
    {
      keep[ furthest ] = true;
      pending.emplace_back( arc.first, furthest );
      pending.emplace_back( furthest, arc.second );
    }
  }

  std::vector< point > out;

  for( size_t index = 0; index < count; ++index )
  {
    if( keep[ index ] )
    {
      out.push_back( contour[ index ] );
    }
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
