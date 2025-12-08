/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
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
 * \brief Stereo measurement process implementation
 */

#include "ocv_measurement_process.h"
#include "ocv_keypoints_from_mask.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/bounding_box.h>
#include <vital/types/camera_perspective.h>
#include <vital/io/camera_rig_io.h>

#include <sprokit/processes/kwiver_type_traits.h>

// Suppress warnings from external headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#include <arrows/mvg/triangulate.h>
#pragma GCC diagnostic pop

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <algorithm>
#include <limits>
#include <cmath>

namespace kv = kwiver::vital;

namespace viame
{

// Config traits
create_config_trait( calibration_file, std::string, "",
  "Stereo calibration file (JSON format compatible with kwiver camera_rig_stereo)" );
create_config_trait( measurement_file, std::string, "",
  "Output file to write detection measurements (CSV format)" );
create_config_trait( max_error_small, double, "6.0",
  "Maximum reprojection error threshold for small fish (fishlen <= small_len)" );
create_config_trait( max_error_large, double, "14.0",
  "Maximum reprojection error threshold for large fish (fishlen > small_len)" );
create_config_trait( small_len, double, "150.0",
  "Length threshold (in mm) to switch between small and large error thresholds" );

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
// Match data structure
struct MatchData
{
  int i;                    // Index in detections1
  int j;                    // Index in detections2
  double fishlen;           // Measured fish length in mm
  double range;             // Average Z distance
  double error;             // Reprojection error
  double dz;                // Z difference between keypoints
  std::vector<cv::Point2d> box_pts1;
  std::vector<cv::Point2d> box_pts2;
  Eigen::Vector3f world_pt1;  // 3D world point for head
  Eigen::Vector3f world_pt2;  // 3D world point for tail
};

// =============================================================================
// Private implementation class
class ocv_measurement_process::priv
{
public:
  explicit priv( ocv_measurement_process* parent );
  ~priv();

  // Triangulate a single point and compute reprojection error
  double triangulate_and_error(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const cv::Point2d& pt1,
    const cv::Point2d& pt2,
    Eigen::Vector3f& world_pt );

  // Find optimal matching between detection sets
  std::vector<MatchData> find_matches(
    const std::vector<kv::detected_object_sptr>& detections1,
    const std::vector<kv::detected_object_sptr>& detections2 );

  // Hungarian algorithm for minimum weight assignment
  std::vector<std::pair<int, int>> minimum_weight_assignment(
    const cv::Mat& cost_matrix );

  // Configuration values
  std::string m_calibration_file;
  std::string m_measurement_file;
  double m_max_error_small;
  double m_max_error_large;
  double m_small_len;

  // State
  kv::camera_rig_stereo_sptr m_calibration;
  std::ofstream m_output_file;
  kv::frame_id_t m_frame_id;
  kv::track_id_t m_track_id;

  ocv_measurement_process* parent;
};

// -----------------------------------------------------------------------------
ocv_measurement_process::priv
::priv( ocv_measurement_process* ptr )
  : m_calibration_file( "" )
  , m_measurement_file( "" )
  , m_max_error_small( 6.0 )
  , m_max_error_large( 14.0 )
  , m_small_len( 150.0 )
  , m_frame_id( 0 )
  , m_track_id( 0 )
  , parent( ptr )
{
}

// -----------------------------------------------------------------------------
ocv_measurement_process::priv
::~priv()
{
  if( m_output_file.is_open() )
  {
    m_output_file.close();
  }
}

// -----------------------------------------------------------------------------
double
ocv_measurement_process::priv
::triangulate_and_error(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const cv::Point2d& pt1,
  const cv::Point2d& pt2,
  Eigen::Vector3f& world_pt )
{
  // Convert to Eigen format
  Eigen::Matrix<float, 2, 1> left_pt( static_cast<float>(pt1.x), static_cast<float>(pt1.y) );
  Eigen::Matrix<float, 2, 1> right_pt( static_cast<float>(pt2.x), static_cast<float>(pt2.y) );

  // Triangulate using kwiver's fast two-view method
  world_pt = kwiver::arrows::mvg::triangulate_fast_two_view(
    left_cam, right_cam, left_pt, right_pt );

  // Compute reprojection error
  // Project 3D point back to both cameras
  kv::vector_3d world_pt_d( world_pt.x(), world_pt.y(), world_pt.z() );

  kv::vector_2d proj1 = left_cam.project( world_pt_d );
  kv::vector_2d proj2 = right_cam.project( world_pt_d );

  double err1 = std::pow( proj1.x() - pt1.x, 2 ) + std::pow( proj1.y() - pt1.y, 2 );
  double err2 = std::pow( proj2.x() - pt2.x, 2 ) + std::pow( proj2.y() - pt2.y, 2 );

  return ( err1 + err2 ) / 2.0;
}

// -----------------------------------------------------------------------------
std::vector<std::pair<int, int>>
ocv_measurement_process::priv
::minimum_weight_assignment( const cv::Mat& cost_matrix )
{
  int n1 = cost_matrix.rows;
  int n2 = cost_matrix.cols;
  int n = std::max( n1, n2 );

  // Embed in padded square matrix
  cv::Mat padded = cv::Mat::zeros( n, n, CV_64F );
  double large_val = 0;

  for( int i = 0; i < n1; ++i )
  {
    for( int j = 0; j < n2; ++j )
    {
      double val = cost_matrix.at<double>(i, j);
      if( std::isfinite( val ) && val > 0 )
      {
        large_val += val;
      }
    }
  }
  large_val = ( n + large_val ) * 2;

  for( int i = 0; i < n; ++i )
  {
    for( int j = 0; j < n; ++j )
    {
      if( i < n1 && j < n2 )
      {
        double val = cost_matrix.at<double>(i, j);
        padded.at<double>(i, j) = std::isfinite( val ) ? val : large_val;
      }
      else
      {
        padded.at<double>(i, j) = large_val;
      }
    }
  }

  // Simple greedy assignment (approximation to Hungarian algorithm)
  std::vector<std::pair<int, int>> assignment;
  std::vector<bool> row_used( n, false );
  std::vector<bool> col_used( n, false );

  for( int iter = 0; iter < n; ++iter )
  {
    double min_val = large_val;
    int min_i = -1, min_j = -1;

    for( int i = 0; i < n; ++i )
    {
      if( row_used[i] ) continue;
      for( int j = 0; j < n; ++j )
      {
        if( col_used[j] ) continue;
        if( padded.at<double>(i, j) < min_val )
        {
          min_val = padded.at<double>(i, j);
          min_i = i;
          min_j = j;
        }
      }
    }

    if( min_i >= 0 && min_j >= 0 && min_val < large_val )
    {
      if( min_i < n1 && min_j < n2 )
      {
        assignment.push_back( std::make_pair( min_i, min_j ) );
      }
      row_used[min_i] = true;
      col_used[min_j] = true;
    }
    else
    {
      break;
    }
  }

  return assignment;
}

// -----------------------------------------------------------------------------
std::vector<MatchData>
ocv_measurement_process::priv
::find_matches(
  const std::vector<kv::detected_object_sptr>& detections1,
  const std::vector<kv::detected_object_sptr>& detections2 )
{
  size_t n1 = detections1.size();
  size_t n2 = detections2.size();

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector<MatchData>();
  }

  // Get camera references
  kv::simple_camera_perspective& left_cam(
    dynamic_cast<kv::simple_camera_perspective&>( *(m_calibration->left()) ) );
  kv::simple_camera_perspective& right_cam(
    dynamic_cast<kv::simple_camera_perspective&>( *(m_calibration->right()) ) );

  // Pre-compute box points for all detections
  std::vector<std::vector<cv::Point2d>> box_pts1( n1 ), box_pts2( n2 );
  for( size_t i = 0; i < n1; ++i )
  {
    box_pts1[i] = compute_box_points( detections1[i] );
  }
  for( size_t j = 0; j < n2; ++j )
  {
    box_pts2[j] = compute_box_points( detections2[j] );
  }

  cv::Mat cost_errors = cv::Mat::zeros( static_cast<int>(n1), static_cast<int>(n2), CV_64F );
  std::map<std::pair<int, int>, MatchData> cand_data;

  // Initialize with infinity
  for( size_t i = 0; i < n1; ++i )
  {
    for( size_t j = 0; j < n2; ++j )
    {
      cost_errors.at<double>( static_cast<int>(i), static_cast<int>(j) ) =
        std::numeric_limits<double>::infinity();
    }
  }

  // Evaluate all pairs
  for( size_t i = 0; i < n1; ++i )
  {
    for( size_t j = 0; j < n2; ++j )
    {
      // Get keypoints from box points
      auto kp1 = center_keypoints( box_pts1[i] );
      auto kp2 = center_keypoints( box_pts2[j] );

      // Triangulate both keypoints
      Eigen::Vector3f world_pt_head, world_pt_tail;

      double err_head = triangulate_and_error(
        left_cam, right_cam, kp1.first, kp2.first, world_pt_head );
      double err_tail = triangulate_and_error(
        left_cam, right_cam, kp1.second, kp2.second, world_pt_tail );

      double error = ( err_head + err_tail ) / 2.0;

      // Compute fish length
      double fishlen = ( world_pt_head - world_pt_tail ).norm();

      // Compute range (average Z)
      double range = ( world_pt_head.z() + world_pt_tail.z() ) / 2.0;

      // Compute dz
      double dz = std::abs( world_pt_head.z() - world_pt_tail.z() );

      // Store candidate data
      MatchData data;
      data.i = static_cast<int>(i);
      data.j = static_cast<int>(j);
      data.fishlen = fishlen;
      data.range = range;
      data.error = error;
      data.dz = dz;
      data.box_pts1 = box_pts1[i];
      data.box_pts2 = box_pts2[j];
      data.world_pt1 = world_pt_head;
      data.world_pt2 = world_pt_tail;

      cand_data[std::make_pair( static_cast<int>(i), static_cast<int>(j) )] = data;

      // Check chirality (both Z coordinates must be positive - in front of cameras)
      bool both_in_front = ( world_pt_head.z() > 0 ) && ( world_pt_tail.z() > 0 );
      if( !both_in_front )
      {
        continue;
      }

      // Check reprojection error threshold
      double error_thresh = ( fishlen <= m_small_len ) ? m_max_error_small : m_max_error_large;
      if( error >= error_thresh )
      {
        continue;
      }

      cost_errors.at<double>( static_cast<int>(i), static_cast<int>(j) ) = error;
    }
  }

  // Find optimal assignment
  auto assignment = minimum_weight_assignment( cost_errors );

  // Collect match data
  std::vector<MatchData> matches;
  for( const auto& pair : assignment )
  {
    auto it = cand_data.find( pair );
    if( it != cand_data.end() )
    {
      matches.push_back( it->second );
    }
  }

  return matches;
}

// =============================================================================
ocv_measurement_process
::ocv_measurement_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new ocv_measurement_process::priv( this ) )
{
  make_ports();
  make_config();
}

ocv_measurement_process
::~ocv_measurement_process()
{
}

// -----------------------------------------------------------------------------
void
ocv_measurement_process
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
  declare_output_port_using_trait( detected_object_set1, optional );
  declare_output_port_using_trait( detected_object_set2, optional );
  declare_output_port_using_trait( object_track_set1, required );
  declare_output_port_using_trait( object_track_set2, required );
}

// -----------------------------------------------------------------------------
void
ocv_measurement_process
::make_config()
{
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( measurement_file );
  declare_config_using_trait( max_error_small );
  declare_config_using_trait( max_error_large );
  declare_config_using_trait( small_len );
}

// -----------------------------------------------------------------------------
void
ocv_measurement_process
::_configure()
{
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_measurement_file = config_value_using_trait( measurement_file );
  d->m_max_error_small = config_value_using_trait( max_error_small );
  d->m_max_error_large = config_value_using_trait( max_error_large );
  d->m_small_len = config_value_using_trait( small_len );

  // Load calibration using kwiver's stereo rig reader
  if( d->m_calibration_file.empty() )
  {
    throw std::runtime_error( "Must specify a valid camera calibration file" );
  }

  d->m_calibration = kv::read_stereo_rig( d->m_calibration_file );

  if( !d->m_calibration )
  {
    throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
  }

  // Open measurement output file
  if( !d->m_measurement_file.empty() )
  {
    d->m_output_file.open( d->m_measurement_file );
    if( !d->m_output_file.is_open() )
    {
      LOG_WARN( logger(), "Failed to open measurement output file: " << d->m_measurement_file );
    }
    else
    {
      // Write header
      d->m_output_file << "current_frame,fishlen,range,error,dz,box_pts1,box_pts2\n";
    }
  }

  LOG_INFO( logger(), "Stereo calibration loaded successfully from: " << d->m_calibration_file );
}

// -----------------------------------------------------------------------------
void
ocv_measurement_process
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

  // Find matches
  auto matches = d->find_matches( detections1, detections2 );

  LOG_DEBUG( logger(), "Found " << matches.size() << " matches" );

  // Write measurements to file
  if( d->m_output_file.is_open() )
  {
    for( const auto& match : matches )
    {
      d->m_output_file << d->m_frame_id << ","
                       << match.fishlen << ","
                       << match.range << ","
                       << match.error << ","
                       << match.dz << ",";

      // Box points as string
      d->m_output_file << "[";
      for( size_t k = 0; k < match.box_pts1.size(); ++k )
      {
        if( k > 0 ) d->m_output_file << ";";
        d->m_output_file << "[" << match.box_pts1[k].x << ";" << match.box_pts1[k].y << "]";
      }
      d->m_output_file << "],[";
      for( size_t k = 0; k < match.box_pts2.size(); ++k )
      {
        if( k > 0 ) d->m_output_file << ";";
        d->m_output_file << "[" << match.box_pts2[k].x << ";" << match.box_pts2[k].y << "]";
      }
      d->m_output_file << "]\n";
    }

    if( !matches.empty() )
    {
      d->m_output_file.flush();
    }
  }

  // Track which detections have matches
  std::vector<bool> has_match1( detections1.size(), false );
  std::vector<bool> has_match2( detections2.size(), false );

  // Assign lengths to matched detections and create tracks
  std::vector<kv::track_sptr> output_trks1, output_trks2;

  for( const auto& match : matches )
  {
    int i1 = match.i;
    int i2 = match.j;

    has_match1[i1] = true;
    has_match2[i2] = true;

    // Set length on detections
    detections1[i1]->set_length( match.fishlen );
    detections2[i2]->set_length( match.fishlen );

    // Add keypoints
    auto kp1 = center_keypoints( match.box_pts1 );
    auto kp2 = center_keypoints( match.box_pts2 );

    detections1[i1]->add_keypoint( "head", kv::point_2d( kp1.first.x, kp1.first.y ) );
    detections1[i1]->add_keypoint( "tail", kv::point_2d( kp1.second.x, kp1.second.y ) );
    detections2[i2]->add_keypoint( "head", kv::point_2d( kp2.first.x, kp2.first.y ) );
    detections2[i2]->add_keypoint( "tail", kv::point_2d( kp2.second.x, kp2.second.y ) );

    // Create tracks with same ID for matched pairs
    auto state1 = std::make_shared<kv::object_track_state>( timestamp, detections1[i1] );
    auto state2 = std::make_shared<kv::object_track_state>( timestamp, detections2[i2] );

    auto track1 = kv::track::create();
    track1->set_id( d->m_track_id );
    track1->append( state1 );

    auto track2 = kv::track::create();
    track2->set_id( d->m_track_id );
    track2->append( state2 );

    output_trks1.push_back( track1 );
    output_trks2.push_back( track2 );

    d->m_track_id++;
  }

  // Add unmatched detections as separate tracks
  for( size_t i = 0; i < detections1.size(); ++i )
  {
    if( !has_match1[i] )
    {
      auto state = std::make_shared<kv::object_track_state>( timestamp, detections1[i] );
      auto track = kv::track::create();
      track->set_id( d->m_track_id );
      track->append( state );
      output_trks1.push_back( track );
      d->m_track_id++;
    }
  }

  for( size_t i = 0; i < detections2.size(); ++i )
  {
    if( !has_match2[i] )
    {
      auto state = std::make_shared<kv::object_track_state>( timestamp, detections2[i] );
      auto track = kv::track::create();
      track->set_id( d->m_track_id );
      track->append( state );
      output_trks2.push_back( track );
      d->m_track_id++;
    }
  }

  // Create output sets
  auto output_det_set1 = std::make_shared<kv::detected_object_set>( detections1 );
  auto output_det_set2 = std::make_shared<kv::detected_object_set>( detections2 );
  auto output_track_set1 = std::make_shared<kv::object_track_set>( output_trks1 );
  auto output_track_set2 = std::make_shared<kv::object_track_set>( output_trks2 );

  // Push outputs
  push_to_port_using_trait( detected_object_set1, output_det_set1 );
  push_to_port_using_trait( detected_object_set2, output_det_set2 );
  push_to_port_using_trait( object_track_set1, output_track_set1 );
  push_to_port_using_trait( object_track_set2, output_track_set2 );

  d->m_frame_id++;
}

} // end namespace viame
