/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Keypoints from a detection's mask, on image_ops
 *
 * Five ways of finding the two ends of a shape: the edge midpoints of the
 * mask's oriented box, the extremes along its principal axis, the two
 * farthest points of its hull, the short-edge midpoints of the hull's
 * oriented box, and the endpoints of its morphological skeleton. Then an
 * option to pull each result back onto the mask's outline.
 *
 * Two things worth knowing before reading them.
 *
 * **"Head" is whichever end has the larger x.** Every method finishes by
 * sorting its two points that way, so the names are about image position
 * and not about the animal: a fish facing left has its head recorded as its
 * tail.
 *
 * **`oriented_bbox` is oriented only when there is a mask.** Without one it
 * falls back to the axis-aligned corners of the bounding box, and the
 * "edge midpoints" are then the midpoints of the box's sides.
 */

#include "add_keypoints_from_mask.h"

#include <image_ops/contours.h>
#include <image_ops/morphology.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace kv = kwiver::vital;
namespace io = viame::image_ops;

namespace viame
{

namespace {

// ----------------------------------------------------------------------------
/// The detection's mask as a byte image, or an empty one.
kv::image_of< uint8_t >
mask_image( kv::detected_object_sptr const& det )
{
  auto const container = det ? det->mask() : nullptr;

  if( !container )
  {
    return {};
  }

  return kv::image_of< uint8_t >( container->get_image() );
}

// ----------------------------------------------------------------------------
/// Every set pixel of \p mask, in the mask's own coordinates.
std::vector< io::point >
set_pixels( kv::image_of< uint8_t > const& mask )
{
  std::vector< io::point > points;

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      if( mask( i, j, 0 ) > 0 )
      {
        points.push_back( { static_cast< long >( i ),
                            static_cast< long >( j ) } );
      }
    }
  }

  return points;
}

// ----------------------------------------------------------------------------
/// Head first, which is whichever point has the larger x.
std::pair< kv::vector_2d, kv::vector_2d >
ordered( kv::vector_2d const& first, kv::vector_2d const& second )
{
  return ( first.x() >= second.x() ) ? std::make_pair( first, second )
                                     : std::make_pair( second, first );
}

// ----------------------------------------------------------------------------
/// The mask thresholded to 0 or 255, which is what the skeleton thins.
kv::image_of< uint8_t >
binary_of( kv::image_of< uint8_t > const& mask )
{
  kv::image_of< uint8_t > out( mask.width(), mask.height(), 1 );

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      out( i, j, 0 ) = mask( i, j, 0 ) > 0 ? 255 : 0;
    }
  }

  return out;
}

} // namespace

// ----------------------------------------------------------------------------
std::vector< kv::vector_2d >
get_mask_points( kv::detected_object_sptr det )
{
  std::vector< kv::vector_2d > points;

  auto const mask = mask_image( det );

  if( !mask.size() )
  {
    return points;
  }

  auto const bbox = det->bounding_box();

  for( auto const& p : set_pixels( mask ) )
  {
    // The truncating int cast the C++ had, which for a positive coordinate
    // is a floor.
    points.emplace_back(
      static_cast< double >(
        static_cast< long >( static_cast< double >( p.i ) + bbox.min_x() ) ),
      static_cast< double >(
        static_cast< long >( static_cast< double >( p.j ) + bbox.min_y() ) ) );
  }

  return points;
}

// ----------------------------------------------------------------------------
std::vector< kv::vector_2d >
compute_box_points( kv::detected_object_sptr det )
{
  auto const bbox = det->bounding_box();

  std::vector< kv::vector_2d > const axis_aligned{
    { bbox.min_x(), bbox.min_y() },
    { bbox.max_x(), bbox.min_y() },
    { bbox.max_x(), bbox.max_y() },
    { bbox.min_x(), bbox.max_y() },
  };

  auto const mask = mask_image( det );

  if( !mask.size() )
  {
    return axis_aligned;
  }

  auto const points = set_pixels( mask );

  if( points.empty() )
  {
    return axis_aligned;
  }

  auto const rect = io::min_area_rect( io::convex_hull( points ) );

  std::vector< kv::vector_2d > out;

  for( auto const& corner : rect.corners() )
  {
    out.emplace_back( corner.first + bbox.min_x(),
                      corner.second + bbox.min_y() );
  }

  return out;
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
center_keypoints( const std::vector< kv::vector_2d >& box_points )
{
  if( box_points.size() < 4 )
  {
    return { { 0.0, 0.0 }, { 0.0, 0.0 } };
  }

  std::vector< kv::vector_2d > centres;

  for( size_t at = 0; at < 4; ++at )
  {
    auto const& a = box_points[ at ];
    auto const& b = box_points[ ( at + 1 ) % 4 ];
    centres.emplace_back( ( a.x() + b.x() ) * 0.5, ( a.y() + b.y() ) * 0.5 );
  }

  auto lowest = centres[ 0 ];
  auto highest = centres[ 0 ];

  for( auto const& centre : centres )
  {
    if( centre.x() < lowest.x() ) { lowest = centre; }
    if( centre.x() > highest.x() ) { highest = centre; }
  }

  // Head is the maximum x, tail the minimum
  return { highest, lowest };
}

// ----------------------------------------------------------------------------
bool
add_keypoints_from_box( kv::detected_object_sptr det )
{
  if( !det )
  {
    return false;
  }

  auto const box = compute_box_points( det );

  if( box.size() < 4 )
  {
    return false;
  }

  auto const keypoints = center_keypoints( box );

  det->add_keypoint( "head",
                     kv::point_2d( keypoints.first.x(),
                                   keypoints.first.y() ) );
  det->add_keypoint( "tail",
                     kv::point_2d( keypoints.second.x(),
                                   keypoints.second.y() ) );

  return true;
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
compute_keypoints_oriented_bbox( kv::detected_object_sptr det )
{
  auto const box = compute_box_points( det );

  if( box.size() < 4 )
  {
    return { { 0.0, 0.0 }, { 0.0, 0.0 } };
  }

  return center_keypoints( box );
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
compute_keypoints_pca( kv::detected_object_sptr det )
{
  auto const points = get_mask_points( det );

  if( points.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  // `cv::PCA` of a two column matrix: the mean, and the eigenvector of the
  // covariance with the larger eigenvalue. Two by two and symmetric, so the
  // eigenvector is analytic rather than iterative. Its **sign** does not
  // matter -- the projections are only used for their extremes, and the
  // result is ordered by x afterwards.
  double mean_x = 0.0;
  double mean_y = 0.0;

  for( auto const& p : points )
  {
    mean_x += p.x();
    mean_y += p.y();
  }

  auto const count = static_cast< double >( points.size() );
  mean_x /= count;
  mean_y /= count;

  double xx = 0.0;
  double xy = 0.0;
  double yy = 0.0;

  for( auto const& p : points )
  {
    auto const dx = p.x() - mean_x;
    auto const dy = p.y() - mean_y;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
  }

  auto const trace = xx + yy;
  auto const difference = xx - yy;
  auto const root = std::sqrt( difference * difference + 4.0 * xy * xy );
  auto const eigenvalue = ( trace + root ) / 2.0;

  double axis_x = 1.0;
  double axis_y = 0.0;

  if( std::abs( xy ) > 1e-12 )
  {
    axis_x = eigenvalue - yy;
    axis_y = xy;
  }
  else if( yy > xx )
  {
    axis_x = 0.0;
    axis_y = 1.0;
  }

  auto const length = std::hypot( axis_x, axis_y );

  if( length > 0.0 )
  {
    axis_x /= length;
    axis_y /= length;
  }

  auto lowest = std::numeric_limits< double >::max();
  auto highest = std::numeric_limits< double >::lowest();
  kv::vector_2d low{ 0.0, 0.0 };
  kv::vector_2d high{ 0.0, 0.0 };

  for( auto const& p : points )
  {
    auto const projection = ( p.x() - mean_x ) * axis_x +
                            ( p.y() - mean_y ) * axis_y;

    if( projection < lowest ) { lowest = projection; low = p; }
    if( projection > highest ) { highest = projection; high = p; }
  }

  return ordered( high, low );
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
compute_keypoints_farthest( kv::detected_object_sptr det )
{
  auto const points = get_mask_points( det );

  if( points.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  std::vector< io::point > integral;

  for( auto const& p : points )
  {
    integral.push_back( { static_cast< long >( p.x() ),
                          static_cast< long >( p.y() ) } );
  }

  auto const hull = io::convex_hull( integral );

  if( hull.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  double furthest = 0.0;
  kv::vector_2d first{ 0.0, 0.0 };
  kv::vector_2d second{ 0.0, 0.0 };

  for( size_t a = 0; a < hull.size(); ++a )
  {
    for( size_t b = a + 1; b < hull.size(); ++b )
    {
      auto const dx = static_cast< double >( hull[ b ].i - hull[ a ].i );
      auto const dy = static_cast< double >( hull[ b ].j - hull[ a ].j );
      auto const distance = dx * dx + dy * dy;

      if( distance > furthest )
      {
        furthest = distance;
        first = { static_cast< double >( hull[ a ].i ),
                  static_cast< double >( hull[ a ].j ) };
        second = { static_cast< double >( hull[ b ].i ),
                   static_cast< double >( hull[ b ].j ) };
      }
    }
  }

  return ordered( first, second );
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
compute_keypoints_hull_extremes( kv::detected_object_sptr det )
{
  auto const points = get_mask_points( det );

  if( points.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  std::vector< io::point > integral;

  for( auto const& p : points )
  {
    integral.push_back( { static_cast< long >( p.x() ),
                          static_cast< long >( p.y() ) } );
  }

  auto const hull = io::convex_hull( integral );

  if( hull.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  auto const corners = io::min_area_rect( hull ).corners();

  auto const edge = [ & ]( size_t a, size_t b )
  {
    return std::hypot( corners[ b ].first - corners[ a ].first,
                       corners[ b ].second - corners[ a ].second );
  };

  auto const midpoint = [ & ]( size_t a, size_t b )
  {
    return kv::vector_2d( ( corners[ a ].first + corners[ b ].first ) * 0.5,
                          ( corners[ a ].second + corners[ b ].second ) * 0.5 );
  };

  // The midpoints of the two **short** edges, which are the pair adjacent
  // to the long ones.
  if( edge( 0, 1 ) > edge( 1, 2 ) )
  {
    return ordered( midpoint( 1, 2 ), midpoint( 3, 0 ) );
  }

  return ordered( midpoint( 0, 1 ), midpoint( 2, 3 ) );
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
compute_keypoints_skeleton( kv::detected_object_sptr det )
{
  auto const mask = mask_image( det );

  if( !mask.size() )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  auto const bbox = det->bounding_box();
  auto image = binary_of( mask );

  // Lantuejoul's skeleton: erode, open, keep the difference, repeat until
  // nothing is left. Not a thinning -- it gives a sparser and more broken
  // set than one, which is why the endpoint search below falls back to the
  // furthest pair of skeleton pixels when it finds no endpoints.
  kv::image_of< uint8_t > skeleton( image.width(), image.height(), 1 );

  for( size_t j = 0; j < skeleton.height(); ++j )
  {
    for( size_t i = 0; i < skeleton.width(); ++i )
    {
      skeleton( i, j, 0 ) = 0;
    }
  }

  auto const element = io::cross_element( 3, 3 );

  for( int iteration = 0; iteration < 1000; ++iteration )
  {
    auto const eroded = io::grey_erode( image, element );
    auto const opened = io::grey_dilate( eroded, element );

    bool any = false;

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const difference = static_cast< int >( image( i, j, 0 ) ) -
                                static_cast< int >( opened( i, j, 0 ) );

        if( difference > 0 )
        {
          skeleton( i, j, 0 ) = 255;
        }

        if( eroded( i, j, 0 ) > 0 )
        {
          any = true;
        }
      }
    }

    image = eroded;

    if( !any )
    {
      break;
    }
  }

  // The interior only, as the C++ scanned: an endpoint on the border would
  // need neighbours outside the mask to be counted.
  std::vector< io::point > filled;
  std::vector< io::point > ends;

  for( long j = 1; j + 1 < static_cast< long >( skeleton.height() ); ++j )
  {
    for( long i = 1; i + 1 < static_cast< long >( skeleton.width() ); ++i )
    {
      if( skeleton( i, j, 0 ) == 0 )
      {
        continue;
      }

      filled.push_back( { i, j } );

      int neighbours = 0;

      for( long dj = -1; dj <= 1; ++dj )
      {
        for( long di = -1; di <= 1; ++di )
        {
          if( di == 0 && dj == 0 )
          {
            continue;
          }

          if( skeleton( i + di, j + dj, 0 ) > 0 )
          {
            ++neighbours;
          }
        }
      }

      if( neighbours == 1 )
      {
        ends.push_back( { i, j } );
      }
    }
  }

  auto const furthest_pair =
    [ & ]( std::vector< io::point > const& candidates, bool& found )
    {
      double furthest = 0.0;
      std::pair< io::point, io::point > best{ candidates.front(),
                                              candidates.front() };
      found = false;

      for( size_t a = 0; a < candidates.size(); ++a )
      {
        for( size_t b = a + 1; b < candidates.size(); ++b )
        {
          auto const dx =
            static_cast< double >( candidates[ b ].i - candidates[ a ].i );
          auto const dy =
            static_cast< double >( candidates[ b ].j - candidates[ a ].j );
          auto const distance = dx * dx + dy * dy;

          if( distance > furthest )
          {
            furthest = distance;
            best = { candidates[ a ], candidates[ b ] };
            found = true;
          }
        }
      }

      return best;
    };

  std::pair< io::point, io::point > chosen;
  bool found = false;

  if( ends.size() >= 2 )
  {
    chosen = furthest_pair( ends, found );

    if( !found )
    {
      // Every endpoint at the same place, which the C++'s uninitialised
      // index pair left as the first two.
      chosen = { ends[ 0 ], ends[ 1 ] };
    }
  }
  else if( filled.size() >= 2 )
  {
    chosen = furthest_pair( filled, found );

    if( !found )
    {
      return compute_keypoints_oriented_bbox( det );
    }
  }
  else
  {
    return compute_keypoints_oriented_bbox( det );
  }

  kv::vector_2d first( static_cast< double >( chosen.first.i ) + bbox.min_x(),
                       static_cast< double >( chosen.first.j ) + bbox.min_y() );
  kv::vector_2d second(
    static_cast< double >( chosen.second.i ) + bbox.min_x(),
    static_cast< double >( chosen.second.j ) + bbox.min_y() );

  return ordered( first, second );
}

// ----------------------------------------------------------------------------
kv::vector_2d
clip_point_to_mask_boundary( const kv::vector_2d& target,
                             kv::detected_object_sptr det )
{
  auto const mask = mask_image( det );

  if( !mask.size() )
  {
    return target;
  }

  auto const contours = io::find_contours( binary_of( mask ) );

  if( contours.empty() )
  {
    return target;
  }

  auto const bbox = det->bounding_box();

  kv::vector_2d const local( target.x() - bbox.min_x(),
                             target.y() - bbox.min_y() );

  auto nearest = std::numeric_limits< double >::max();
  kv::vector_2d best = local;

  for( auto const& contour : contours )
  {
    auto const count = contour.size();

    if( count == 0 )
    {
      continue;
    }

    for( size_t at = 0; at < count; ++at )
    {
      kv::vector_2d const a( static_cast< double >( contour[ at ].i ),
                             static_cast< double >( contour[ at ].j ) );
      auto const& next = contour[ ( at + 1 ) % count ];
      kv::vector_2d const b( static_cast< double >( next.i ),
                             static_cast< double >( next.j ) );

      auto const along = b - a;
      auto const length = along.dot( along );

      kv::vector_2d closest = a;

      if( length >= 1e-12 )
      {
        auto t = ( local - a ).dot( along ) / length;
        t = std::max( 0.0, std::min( 1.0, t ) );
        closest = a + t * along;
      }

      auto const delta = local - closest;
      auto const distance = delta.dot( delta );

      if( distance < nearest )
      {
        nearest = distance;
        best = closest;
      }
    }
  }

  return { best.x() + bbox.min_x(), best.y() + bbox.min_y() };
}

// ----------------------------------------------------------------------------
std::pair< kv::vector_2d, kv::vector_2d >
compute_keypoints( kv::detected_object_sptr det, const std::string& method )
{
  if( method == "pca" )
  {
    return compute_keypoints_pca( det );
  }
  if( method == "farthest" )
  {
    return compute_keypoints_farthest( det );
  }
  if( method == "hull_extremes" )
  {
    return compute_keypoints_hull_extremes( det );
  }
  if( method == "skeleton" )
  {
    return compute_keypoints_skeleton( det );
  }

  return compute_keypoints_oriented_bbox( det );
}

// ----------------------------------------------------------------------------
bool
is_valid_keypoint_method( const std::string& method )
{
  return method == "oriented_bbox" ||
         method == "pca" ||
         method == "farthest" ||
         method == "hull_extremes" ||
         method == "skeleton";
}

// ----------------------------------------------------------------------------
std::string
keypoint_method_description()
{
  return "Method for computing keypoints from polygon/mask. Options:\n"
         "  oriented_bbox - Use midpoints of short edges of oriented bounding box (default)\n"
         "  pca - Use Principal Component Analysis to find major axis extremes\n"
         "  farthest - Find the two farthest points on the polygon\n"
         "  hull_extremes - Use midpoints of short edges of convex hull's oriented bbox\n"
         "  skeleton - Use endpoints of the medial axis/skeleton";
}

// ----------------------------------------------------------------------------
bool
add_keypoints_from_mask
::check_configuration( kv::config_block_sptr config ) const
{
  auto const method =
    config->get_value< std::string >( "method", "oriented_bbox" );

  if( !is_valid_keypoint_method( method ) )
  {
    LOG_ERROR( logger(), "Invalid method: " << method );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
kv::detected_object_set_sptr
add_keypoints_from_mask
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  auto output = std::make_shared< kv::detected_object_set >();

  for( auto det : *detections )
  {
    if( det->mask() )
    {
      auto keypoints = compute_keypoints( det, c_method );

      if( c_clip_to_mask )
      {
        keypoints.first =
          clip_point_to_mask_boundary( keypoints.first, det );
        keypoints.second =
          clip_point_to_mask_boundary( keypoints.second, det );
      }

      det->add_keypoint( "head",
                         kv::point_2d( keypoints.first.x(),
                                       keypoints.first.y() ) );
      det->add_keypoint( "tail",
                         kv::point_2d( keypoints.second.x(),
                                       keypoints.second.y() ) );
    }

    output->add( det );
  }

  return output;
}

} // end namespace viame
