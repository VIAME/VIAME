/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Stereo detection pairing process implementation
 */

#include "stereo_detection_pairing_process.h"
#include "measurement_utilities.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/bounding_box.h>
#include <vital/types/camera_perspective.h>
#include <vital/io/camera_rig_io.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <set>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

// Config traits
create_config_trait( matching_method, std::string, "iou",
  "Matching method to use. Options: 'iou' (bounding box overlap), "
  "'calibration' (uses stereo geometry and reprojection error)" );

create_config_trait( calibration_file, std::string, "",
  "Stereo calibration file (JSON format). Required when matching_method is 'calibration'." );

create_config_trait( iou_threshold, double, "0.1",
  "Minimum IOU (Intersection over Union) threshold for matching detections. "
  "Used with 'iou' method. Pairs with IOU below this value will not be matched." );

create_config_trait( max_reprojection_error, double, "10.0",
  "Maximum reprojection error (in pixels) for valid matches. "
  "Used with 'calibration' method." );

create_config_trait( default_depth, double, "5.0",
  "Default depth (in meters) for projecting points between cameras. "
  "Used with 'calibration' method to estimate initial correspondence." );

create_config_trait( require_class_match, bool, "true",
  "If true, only detections with the same class label can be matched." );

create_config_trait( use_optimal_assignment, bool, "true",
  "If true, use optimal (greedy) assignment to maximize matching quality. "
  "If false, use simple greedy matching in order of left detections." );

create_config_trait( output_unmatched, bool, "true",
  "If true, output unmatched detections as separate tracks with unique IDs. "
  "If false, only output matched detection pairs." );

// Port traits
create_port_trait( detected_object_set1, detected_object_set,
  "Detections from camera 1 (left)" );
create_port_trait( detected_object_set2, detected_object_set,
  "Detections from camera 2 (right)" );
create_port_trait( object_track_set1, object_track_set,
  "Output tracks for camera 1" );
create_port_trait( object_track_set2, object_track_set,
  "Output tracks for camera 2" );

// =============================================================================
// Private implementation class
class stereo_detection_pairing_process::priv
{
public:
  explicit priv( stereo_detection_pairing_process* parent );
  ~priv();

  // Compute IOU between two bounding boxes
  double compute_iou(
    const kv::bounding_box_d& bbox1,
    const kv::bounding_box_d& bbox2 ) const;

  // Get most likely class label from a detection
  std::string get_class_label( const kv::detected_object_sptr& det ) const;

  // Compute reprojection error for a pair of points
  double compute_reprojection_error(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point,
    const kv::vector_2d& right_point ) const;

  // Find matches using IOU method
  std::vector<std::pair<int, int>> find_matches_iou(
    const std::vector<kv::detected_object_sptr>& detections1,
    const std::vector<kv::detected_object_sptr>& detections2 );

  // Find matches using calibration method
  std::vector<std::pair<int, int>> find_matches_calibration(
    const std::vector<kv::detected_object_sptr>& detections1,
    const std::vector<kv::detected_object_sptr>& detections2 );

  // Greedy minimum weight assignment
  std::vector<std::pair<int, int>> greedy_assignment(
    const std::vector<std::vector<double>>& cost_matrix,
    int n1, int n2 );

  // Configuration values
  std::string m_matching_method;
  std::string m_calibration_file;
  double m_iou_threshold;
  double m_max_reprojection_error;
  double m_default_depth;
  bool m_require_class_match;
  bool m_use_optimal_assignment;
  bool m_output_unmatched;

  // Calibration data
  kv::camera_rig_stereo_sptr m_calibration;

  // State
  kv::track_id_t m_next_track_id;

  stereo_detection_pairing_process* parent;
};

// -----------------------------------------------------------------------------
stereo_detection_pairing_process::priv
::priv( stereo_detection_pairing_process* ptr )
  : m_matching_method( "iou" )
  , m_calibration_file( "" )
  , m_iou_threshold( 0.1 )
  , m_max_reprojection_error( 10.0 )
  , m_default_depth( 5.0 )
  , m_require_class_match( true )
  , m_use_optimal_assignment( true )
  , m_output_unmatched( true )
  , m_next_track_id( 0 )
  , parent( ptr )
{
}

// -----------------------------------------------------------------------------
stereo_detection_pairing_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
double
stereo_detection_pairing_process::priv
::compute_iou(
  const kv::bounding_box_d& bbox1,
  const kv::bounding_box_d& bbox2 ) const
{
  if( !bbox1.is_valid() || !bbox2.is_valid() )
  {
    return 0.0;
  }

  // Compute intersection
  double x1 = std::max( bbox1.min_x(), bbox2.min_x() );
  double y1 = std::max( bbox1.min_y(), bbox2.min_y() );
  double x2 = std::min( bbox1.max_x(), bbox2.max_x() );
  double y2 = std::min( bbox1.max_y(), bbox2.max_y() );

  double intersection_width = std::max( 0.0, x2 - x1 );
  double intersection_height = std::max( 0.0, y2 - y1 );
  double intersection_area = intersection_width * intersection_height;

  if( intersection_area <= 0.0 )
  {
    return 0.0;
  }

  // Compute union
  double area1 = bbox1.width() * bbox1.height();
  double area2 = bbox2.width() * bbox2.height();
  double union_area = area1 + area2 - intersection_area;

  if( union_area <= 0.0 )
  {
    return 0.0;
  }

  return intersection_area / union_area;
}

// -----------------------------------------------------------------------------
std::string
stereo_detection_pairing_process::priv
::get_class_label( const kv::detected_object_sptr& det ) const
{
  if( !det )
  {
    return "";
  }

  auto det_type = det->type();
  if( !det_type )
  {
    return "";
  }

  std::string most_likely;
  det_type->get_most_likely( most_likely );
  return most_likely;
}

// -----------------------------------------------------------------------------
double
stereo_detection_pairing_process::priv
::compute_reprojection_error(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point ) const
{
  // Triangulate the point
  kv::vector_3d point_3d = viame::core::triangulate_point(
    left_cam, right_cam, left_point, right_point );

  // Check if point is in front of both cameras (positive Z in camera coordinates)
  kv::vector_3d left_cam_point = left_cam.rotation() * ( point_3d - left_cam.center() );
  kv::vector_3d right_cam_point = right_cam.rotation() * ( point_3d - right_cam.center() );

  if( left_cam_point.z() <= 0 || right_cam_point.z() <= 0 )
  {
    return std::numeric_limits<double>::infinity();
  }

  // Reproject to both cameras
  kv::vector_2d left_reproj = left_cam.project( point_3d );
  kv::vector_2d right_reproj = right_cam.project( point_3d );

  // Compute reprojection error
  double left_err_sq = ( left_reproj - left_point ).squaredNorm();
  double right_err_sq = ( right_reproj - right_point ).squaredNorm();

  // Return RMS error
  return std::sqrt( ( left_err_sq + right_err_sq ) / 2.0 );
}

// -----------------------------------------------------------------------------
std::vector<std::pair<int, int>>
stereo_detection_pairing_process::priv
::greedy_assignment(
  const std::vector<std::vector<double>>& cost_matrix,
  int n1, int n2 )
{
  std::vector<std::pair<int, int>> assignment;
  std::vector<bool> row_used( n1, false );
  std::vector<bool> col_used( n2, false );

  // Collect all valid costs with their indices
  std::vector<std::tuple<double, int, int>> costs;
  for( int i = 0; i < n1; ++i )
  {
    for( int j = 0; j < n2; ++j )
    {
      double cost = cost_matrix[i][j];
      if( std::isfinite( cost ) && cost < 1e9 )
      {
        costs.push_back( std::make_tuple( cost, i, j ) );
      }
    }
  }

  // Sort by cost (ascending - lower is better)
  std::sort( costs.begin(), costs.end() );

  // Greedily assign
  for( const auto& entry : costs )
  {
    int i = std::get<1>( entry );
    int j = std::get<2>( entry );

    if( !row_used[i] && !col_used[j] )
    {
      assignment.push_back( std::make_pair( i, j ) );
      row_used[i] = true;
      col_used[j] = true;
    }
  }

  return assignment;
}

// -----------------------------------------------------------------------------
std::vector<std::pair<int, int>>
stereo_detection_pairing_process::priv
::find_matches_iou(
  const std::vector<kv::detected_object_sptr>& detections1,
  const std::vector<kv::detected_object_sptr>& detections2 )
{
  int n1 = static_cast<int>( detections1.size() );
  int n2 = static_cast<int>( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector<std::pair<int, int>>();
  }

  // Build cost matrix (1 - IOU, so lower is better)
  std::vector<std::vector<double>> cost_matrix( n1, std::vector<double>( n2, 1e10 ) );

  for( int i = 0; i < n1; ++i )
  {
    const auto& det1 = detections1[i];
    std::string class1 = get_class_label( det1 );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( m_require_class_match )
      {
        std::string class2 = get_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Compute IOU
      double iou = compute_iou( det1->bounding_box(), det2->bounding_box() );

      // Check threshold
      if( iou >= m_iou_threshold )
      {
        // Use 1 - IOU as cost (lower is better)
        cost_matrix[i][j] = 1.0 - iou;
      }
    }
  }

  // Find optimal assignment
  if( m_use_optimal_assignment )
  {
    return greedy_assignment( cost_matrix, n1, n2 );
  }
  else
  {
    // Simple sequential matching
    std::vector<std::pair<int, int>> matches;
    std::set<int> used_j;

    for( int i = 0; i < n1; ++i )
    {
      int best_j = -1;
      double best_cost = 1e10;

      for( int j = 0; j < n2; ++j )
      {
        if( used_j.find( j ) != used_j.end() )
        {
          continue;
        }

        if( cost_matrix[i][j] < best_cost )
        {
          best_cost = cost_matrix[i][j];
          best_j = j;
        }
      }

      if( best_j >= 0 && best_cost < 1e9 )
      {
        matches.push_back( std::make_pair( i, best_j ) );
        used_j.insert( best_j );
      }
    }

    return matches;
  }
}

// -----------------------------------------------------------------------------
std::vector<std::pair<int, int>>
stereo_detection_pairing_process::priv
::find_matches_calibration(
  const std::vector<kv::detected_object_sptr>& detections1,
  const std::vector<kv::detected_object_sptr>& detections2 )
{
  int n1 = static_cast<int>( detections1.size() );
  int n2 = static_cast<int>( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector<std::pair<int, int>>();
  }

  if( !m_calibration || !m_calibration->left() || !m_calibration->right() )
  {
    LOG_ERROR( parent->logger(), "Calibration not loaded for calibration matching method" );
    return std::vector<std::pair<int, int>>();
  }

  // Get camera references
  const kv::simple_camera_perspective& left_cam =
    dynamic_cast<const kv::simple_camera_perspective&>( *( m_calibration->left() ) );
  const kv::simple_camera_perspective& right_cam =
    dynamic_cast<const kv::simple_camera_perspective&>( *( m_calibration->right() ) );

  // Build cost matrix using reprojection error
  std::vector<std::vector<double>> cost_matrix( n1, std::vector<double>( n2, 1e10 ) );

  for( int i = 0; i < n1; ++i )
  {
    const auto& det1 = detections1[i];
    std::string class1 = get_class_label( det1 );

    // Get left detection center
    const auto& bbox1 = det1->bounding_box();
    if( !bbox1.is_valid() )
    {
      continue;
    }
    kv::vector_2d left_center = bbox1.center();

    // Project left center to right camera at default depth
    kv::vector_2d expected_right = viame::core::project_left_to_right(
      left_cam, right_cam, left_center, m_default_depth );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( m_require_class_match )
      {
        std::string class2 = get_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Get right detection center
      const auto& bbox2 = det2->bounding_box();
      if( !bbox2.is_valid() )
      {
        continue;
      }
      kv::vector_2d right_center = bbox2.center();

      // Quick check: is the right detection center reasonably close to expected position?
      // This is a heuristic to avoid expensive triangulation for obviously bad matches
      double expected_dist = ( right_center - expected_right ).norm();
      double search_radius = std::max( bbox1.width(), bbox1.height() ) * 2.0;
      if( expected_dist > search_radius )
      {
        continue;
      }

      // Compute reprojection error by triangulating the centers
      double reproj_error = compute_reprojection_error(
        left_cam, right_cam, left_center, right_center );

      // Check threshold
      if( reproj_error < m_max_reprojection_error )
      {
        cost_matrix[i][j] = reproj_error;
      }
    }
  }

  // Find optimal assignment
  if( m_use_optimal_assignment )
  {
    return greedy_assignment( cost_matrix, n1, n2 );
  }
  else
  {
    // Simple sequential matching
    std::vector<std::pair<int, int>> matches;
    std::set<int> used_j;

    for( int i = 0; i < n1; ++i )
    {
      int best_j = -1;
      double best_cost = 1e10;

      for( int j = 0; j < n2; ++j )
      {
        if( used_j.find( j ) != used_j.end() )
        {
          continue;
        }

        if( cost_matrix[i][j] < best_cost )
        {
          best_cost = cost_matrix[i][j];
          best_j = j;
        }
      }

      if( best_j >= 0 && best_cost < 1e9 )
      {
        matches.push_back( std::make_pair( i, best_j ) );
        used_j.insert( best_j );
      }
    }

    return matches;
  }
}

// =============================================================================
stereo_detection_pairing_process
::stereo_detection_pairing_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new stereo_detection_pairing_process::priv( this ) )
{
  make_ports();
  make_config();
}

stereo_detection_pairing_process
::~stereo_detection_pairing_process()
{
}

// -----------------------------------------------------------------------------
void
stereo_detection_pairing_process
::make_ports()
{
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // Input ports
  declare_input_port_using_trait( detected_object_set1, required );
  declare_input_port_using_trait( detected_object_set2, required );
  declare_input_port_using_trait( timestamp, required );

  // Output ports
  declare_output_port_using_trait( object_track_set1, optional );
  declare_output_port_using_trait( object_track_set2, optional );
}

// -----------------------------------------------------------------------------
void
stereo_detection_pairing_process
::make_config()
{
  declare_config_using_trait( matching_method );
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( iou_threshold );
  declare_config_using_trait( max_reprojection_error );
  declare_config_using_trait( default_depth );
  declare_config_using_trait( require_class_match );
  declare_config_using_trait( use_optimal_assignment );
  declare_config_using_trait( output_unmatched );
}

// -----------------------------------------------------------------------------
void
stereo_detection_pairing_process
::_configure()
{
  d->m_matching_method = config_value_using_trait( matching_method );
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_iou_threshold = config_value_using_trait( iou_threshold );
  d->m_max_reprojection_error = config_value_using_trait( max_reprojection_error );
  d->m_default_depth = config_value_using_trait( default_depth );
  d->m_require_class_match = config_value_using_trait( require_class_match );
  d->m_use_optimal_assignment = config_value_using_trait( use_optimal_assignment );
  d->m_output_unmatched = config_value_using_trait( output_unmatched );

  // Validate matching method
  if( d->m_matching_method != "iou" && d->m_matching_method != "calibration" )
  {
    throw std::runtime_error( "Invalid matching_method: '" + d->m_matching_method +
                              "'. Must be 'iou' or 'calibration'." );
  }

  // Load calibration if using calibration method
  if( d->m_matching_method == "calibration" )
  {
    if( d->m_calibration_file.empty() )
    {
      throw std::runtime_error( "calibration_file is required when matching_method is 'calibration'" );
    }

    d->m_calibration = kv::read_stereo_rig( d->m_calibration_file );

    if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
    {
      throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
    }

    LOG_INFO( logger(), "Loaded stereo calibration from: " << d->m_calibration_file );
  }

  LOG_INFO( logger(), "Stereo detection pairing configured:" );
  LOG_INFO( logger(), "  Matching method: " << d->m_matching_method );
  if( d->m_matching_method == "iou" )
  {
    LOG_INFO( logger(), "  IOU threshold: " << d->m_iou_threshold );
  }
  else
  {
    LOG_INFO( logger(), "  Max reprojection error: " << d->m_max_reprojection_error );
    LOG_INFO( logger(), "  Default depth: " << d->m_default_depth );
  }
  LOG_INFO( logger(), "  Require class match: " << ( d->m_require_class_match ? "true" : "false" ) );
  LOG_INFO( logger(), "  Use optimal assignment: " << ( d->m_use_optimal_assignment ? "true" : "false" ) );
  LOG_INFO( logger(), "  Output unmatched: " << ( d->m_output_unmatched ? "true" : "false" ) );
}

// -----------------------------------------------------------------------------
void
stereo_detection_pairing_process
::_step()
{
  // Grab inputs
  auto detection_set1 = grab_from_port_using_trait( detected_object_set1 );
  auto detection_set2 = grab_from_port_using_trait( detected_object_set2 );
  auto timestamp = grab_from_port_using_trait( timestamp );

  // Convert to vectors for indexed access
  std::vector<kv::detected_object_sptr> detections1, detections2;
  for( const auto& det : *detection_set1 )
  {
    detections1.push_back( det );
  }
  for( const auto& det : *detection_set2 )
  {
    detections2.push_back( det );
  }

  // Find matches using configured method
  std::vector<std::pair<int, int>> matches;
  if( d->m_matching_method == "iou" )
  {
    matches = d->find_matches_iou( detections1, detections2 );
  }
  else // calibration
  {
    matches = d->find_matches_calibration( detections1, detections2 );
  }

  LOG_DEBUG( logger(), "Frame " << timestamp.get_frame()
             << ": Found " << matches.size() << " matches out of "
             << detections1.size() << " left, " << detections2.size() << " right detections" );

  // Track which detections have matches
  std::vector<bool> has_match1( detections1.size(), false );
  std::vector<bool> has_match2( detections2.size(), false );

  // Create tracks for matched pairs
  std::vector<kv::track_sptr> output_trks1, output_trks2;

  for( const auto& match : matches )
  {
    int i1 = match.first;
    int i2 = match.second;

    has_match1[i1] = true;
    has_match2[i2] = true;

    // Create tracks with same ID for matched pairs
    auto state1 = std::make_shared<kv::object_track_state>( timestamp, detections1[i1] );
    auto state2 = std::make_shared<kv::object_track_state>( timestamp, detections2[i2] );

    auto track1 = kv::track::create();
    track1->set_id( d->m_next_track_id );
    track1->append( state1 );

    auto track2 = kv::track::create();
    track2->set_id( d->m_next_track_id );
    track2->append( state2 );

    output_trks1.push_back( track1 );
    output_trks2.push_back( track2 );

    d->m_next_track_id++;
  }

  // Add unmatched detections as separate tracks if configured
  if( d->m_output_unmatched )
  {
    for( size_t i = 0; i < detections1.size(); ++i )
    {
      if( !has_match1[i] )
      {
        auto state = std::make_shared<kv::object_track_state>( timestamp, detections1[i] );
        auto track = kv::track::create();
        track->set_id( d->m_next_track_id );
        track->append( state );
        output_trks1.push_back( track );
        d->m_next_track_id++;
      }
    }

    for( size_t i = 0; i < detections2.size(); ++i )
    {
      if( !has_match2[i] )
      {
        auto state = std::make_shared<kv::object_track_state>( timestamp, detections2[i] );
        auto track = kv::track::create();
        track->set_id( d->m_next_track_id );
        track->append( state );
        output_trks2.push_back( track );
        d->m_next_track_id++;
      }
    }
  }

  // Create output sets
  auto output_track_set1 = std::make_shared<kv::object_track_set>( output_trks1 );
  auto output_track_set2 = std::make_shared<kv::object_track_set>( output_trks2 );

  // Push outputs
  push_to_port_using_trait( object_track_set1, output_track_set1 );
  push_to_port_using_trait( object_track_set2, output_track_set2 );
}

} // end namespace core
} // end namespace viame
