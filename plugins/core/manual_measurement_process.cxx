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
 * \brief Run manual measurement on input tracks
 */

#include "manual_measurement_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/image_container.h>
#include <vital/types/vector.h>
#include <vital/util/string.h>
#include <vital/io/camera_rig_io.h>

#include <arrows/mvg/triangulate.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <sprokit/processes/kwiver_type_traits.h>

#ifdef VIAME_ENABLE_OPENCV
  #include <arrows/ocv/image_container.h>

  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
  #include <opencv2/calib3d/calib3d.hpp>
  #include <opencv2/core/eigen.hpp>
#endif

#include <string>
#include <map>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( calibration_file, std::string, "",
  "Input filename for the calibration file to use" );

create_config_trait( matching_method, std::string, "depth_projection",
  "Method to use for finding corresponding points in right camera for left-only tracks. "
  "Options: 'depth_projection' (uses default_depth to project points), "
  "'feature_matching' (rectifies images and searches along epipolar lines using template matching)" );

create_config_trait( default_depth, double, "5.0",
  "Default depth (in meters) to use when projecting left camera points to right camera "
  "for tracks that only exist in the left camera, when using the depth_projection option" );

create_config_trait( template_size, int, "31",
  "Template window size (in pixels) for feature matching. Must be odd number. "
  "Only used when matching_method is 'feature_matching'" );

create_config_trait( search_range, int, "128",
  "Search range (in pixels) along epipolar line for feature matching. "
  "Only used when matching_method is 'feature_matching'" );

create_port_trait( object_track_set1, object_track_set,
  "The stereo filtered object tracks1.")
create_port_trait( object_track_set2, object_track_set,
  "The stereo filtered object tracks2.")

// =============================================================================
// Private implementation class
class manual_measurement_process::priv
{
public:
  explicit priv( manual_measurement_process* parent );
  ~priv();

  // Helper function to project a left camera point to the right camera
  // using the default depth
  kv::vector_2d project_left_to_right(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point );

#ifdef VIAME_ENABLE_OPENCV
  // Helper function to find corresponding point in right image using template matching
  // Returns true if match found, false otherwise
  bool find_corresponding_point_template_matching(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect );

  // Helper function to compute rectification maps
  void compute_rectification_maps(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const cv::Size& image_size );
#endif

  // Helper function to unrectify a point from rectified space back to original
  kv::vector_2d unrectify_point(
    const kv::vector_2d& rectified_point,
    bool is_right_camera,
    const kv::simple_camera_perspective& camera );

  // Configuration settings
  std::string m_calibration_file;
  double m_default_depth;
  std::string m_matching_method;
  int m_template_size;
  int m_search_range;

  // Other variables
  kv::camera_rig_stereo_sptr m_calibration;
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  manual_measurement_process* parent;

#ifdef VIAME_ENABLE_OPENCV
  // Rectification maps (computed on first use)
  bool m_rectification_computed;
  cv::Mat m_rectification_map_left_x;
  cv::Mat m_rectification_map_left_y;
  cv::Mat m_rectification_map_right_x;
  cv::Mat m_rectification_map_right_y;

  // Rectification matrices for unrectifying points
  cv::Mat m_K1, m_K2, m_R1, m_R2, m_P1, m_P2;
#endif
};


// -----------------------------------------------------------------------------
manual_measurement_process::priv
::priv( manual_measurement_process* ptr )
  : m_calibration_file( "" )
  , m_default_depth( 5.0 )
  , m_matching_method( "depth_projection" )
  , m_template_size( 31 )
  , m_search_range( 128 )
  , m_calibration()
  , m_frame_counter( 0 )
  , parent( ptr )
  , m_rectification_computed( false )
{
}


manual_measurement_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
#ifdef VIAME_ENABLE_OPENCV
void
manual_measurement_process::priv
::compute_rectification_maps(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const cv::Size& image_size )
{
  if( m_rectification_computed )
  {
    return;
  }

  // Get camera intrinsics
  auto left_intrinsics = left_cam.get_intrinsics();
  auto right_intrinsics = right_cam.get_intrinsics();

  // Convert to OpenCV matrices
  cv::Mat K1, K2, D1, D2, R, T;

  // Camera matrices
  Eigen::Matrix3d K1_eigen = left_intrinsics->as_matrix();
  Eigen::Matrix3d K2_eigen = right_intrinsics->as_matrix();
  cv::eigen2cv( K1_eigen, K1 );
  cv::eigen2cv( K2_eigen, K2 );

  // Distortion coefficients
  std::vector<double> left_dist = left_intrinsics->dist_coeffs();
  std::vector<double> right_dist = right_intrinsics->dist_coeffs();

  // Convert distortion coefficients to OpenCV format
  // Ensure we have at least 5 coefficients (k1, k2, p1, p2, k3)
  D1 = cv::Mat::zeros( 5, 1, CV_64F );
  D2 = cv::Mat::zeros( 5, 1, CV_64F );

  for( size_t i = 0; i < std::min( left_dist.size(), size_t(5) ); ++i )
  {
    D1.at<double>( i, 0 ) = left_dist[i];
  }

  for( size_t i = 0; i < std::min( right_dist.size(), size_t(5) ); ++i )
  {
    D2.at<double>( i, 0 ) = right_dist[i];
  }

  // Compute rotation and translation between cameras
  Eigen::Matrix3d R_left = left_cam.rotation().matrix();
  Eigen::Matrix3d R_right = right_cam.rotation().matrix();
  Eigen::Matrix3d R_relative = R_right * R_left.transpose();

  Eigen::Vector3d t_relative = right_cam.center() - left_cam.center();
  t_relative = R_right * t_relative;

  cv::eigen2cv( R_relative, R );
  cv::eigen2cv( t_relative, T );

  // Compute rectification transforms
  cv::Mat Q;
  cv::stereoRectify( K1, D1, K2, D2, image_size, R, T,
                     m_R1, m_R2, m_P1, m_P2, Q,
                     cv::CALIB_ZERO_DISPARITY, 0 );

  // Store camera matrices
  m_K1 = K1.clone();
  m_K2 = K2.clone();

  // Compute rectification maps
  cv::initUndistortRectifyMap( K1, D1, m_R1, m_P1, image_size, CV_32FC1,
    m_rectification_map_left_x, m_rectification_map_left_y );
  cv::initUndistortRectifyMap( K2, D2, m_R2, m_P2, image_size, CV_32FC1,
    m_rectification_map_right_x, m_rectification_map_right_y );

  m_rectification_computed = true;
}

// -----------------------------------------------------------------------------
kv::vector_2d
manual_measurement_process::priv
::unrectify_point(
  const kv::vector_2d& rectified_point,
  bool is_right_camera,
  const kv::simple_camera_perspective& camera )
{
  // Select appropriate matrices
  const cv::Mat& R = is_right_camera ? m_R2 : m_R1;
  const cv::Mat& P = is_right_camera ? m_P2 : m_P1;
  const cv::Mat& K = is_right_camera ? m_K2 : m_K1;

  // Convert point to homogeneous coordinates
  cv::Mat point_rect = ( cv::Mat_<double>( 3, 1 ) <<
    rectified_point.x(), rectified_point.y(), 1.0 );

  // Invert the projection matrix to get normalized coordinates
  // P = K * [R | t], so P^-1 gives us back to camera coordinates
  // For rectified stereo, t is typically [0,0,0] for left camera
  // We're only interested in the rotation part: K_rect = P[:, :3]
  cv::Mat K_rect = P( cv::Rect( 0, 0, 3, 3 ) );

  // Convert to normalized rectified coordinates
  cv::Mat norm_rect = K_rect.inv() * point_rect;

  // Apply inverse rotation to get back to original camera frame
  cv::Mat norm_orig = R.t() * norm_rect;

  // Get normalized coordinates (before applying camera matrix and distortion)
  double x_norm = norm_orig.at<double>( 0, 0 ) / norm_orig.at<double>( 2, 0 );
  double y_norm = norm_orig.at<double>( 1, 0 ) / norm_orig.at<double>( 2, 0 );

  // Get distortion coefficients from camera
  auto intrinsics = camera.get_intrinsics();
  std::vector<double> dist_coeffs = intrinsics->dist_coeffs();

  // Apply distortion model if distortion coefficients exist
  if( !dist_coeffs.empty() )
  {
    // Extract distortion coefficients (OpenCV distortion model)
    double k1 = dist_coeffs.size() > 0 ? dist_coeffs[0] : 0.0;
    double k2 = dist_coeffs.size() > 1 ? dist_coeffs[1] : 0.0;
    double p1 = dist_coeffs.size() > 2 ? dist_coeffs[2] : 0.0;
    double p2 = dist_coeffs.size() > 3 ? dist_coeffs[3] : 0.0;
    double k3 = dist_coeffs.size() > 4 ? dist_coeffs[4] : 0.0;

    // Compute r^2
    double r2 = x_norm * x_norm + y_norm * y_norm;
    double r4 = r2 * r2;
    double r6 = r2 * r4;

    // Radial distortion
    double radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

    // Tangential distortion
    double x_tangential = 2.0 * p1 * x_norm * y_norm + p2 * ( r2 + 2.0 * x_norm * x_norm );
    double y_tangential = p1 * ( r2 + 2.0 * y_norm * y_norm ) + 2.0 * p2 * x_norm * y_norm;

    // Apply distortion
    x_norm = x_norm * radial_distortion + x_tangential;
    y_norm = y_norm * radial_distortion + y_tangential;
  }

  // Project back using original camera matrix
  Eigen::Matrix3d K_eigen = intrinsics->as_matrix();
  double fx = K_eigen( 0, 0 );
  double fy = K_eigen( 1, 1 );
  double cx = K_eigen( 0, 2 );
  double cy = K_eigen( 1, 2 );

  double x_distorted = fx * x_norm + cx;
  double y_distorted = fy * y_norm + cy;

  return kv::vector_2d( x_distorted, y_distorted );
}

// -----------------------------------------------------------------------------
bool
manual_measurement_process::priv
::find_corresponding_point_template_matching(
  const cv::Mat& left_image_rect,
  const cv::Mat& right_image_rect,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect )
{
  int half_template = m_template_size / 2;
  int x_left = static_cast<int>( left_point_rect.x() );
  int y_left = static_cast<int>( left_point_rect.y() );

  // Check if template fits in left image
  if( x_left < half_template || x_left >= left_image_rect.cols - half_template ||
      y_left < half_template || y_left >= left_image_rect.rows - half_template )
  {
    return false;
  }

  // Extract template from left image
  cv::Rect template_rect( x_left - half_template, y_left - half_template,
                          m_template_size, m_template_size );
  cv::Mat template_img = left_image_rect( template_rect );

  // Define search region in right image (along the same scanline for rectified images)
  // Search to the left of the left image point (disparity is typically negative for standard stereo)
  int search_min_x = std::max( 0, x_left - m_search_range );
  int search_max_x = std::min( right_image_rect.cols - m_template_size, x_left );

  if( search_max_x <= search_min_x )
  {
    return false;
  }

  int search_y = std::max( half_template, std::min( y_left, right_image_rect.rows - half_template - 1 ) );

  cv::Rect search_rect( search_min_x, search_y - half_template,
                        search_max_x - search_min_x + m_template_size,
                        m_template_size );

  // Check search rect validity
  if( search_rect.x < 0 || search_rect.y < 0 ||
      search_rect.x + search_rect.width > right_image_rect.cols ||
      search_rect.y + search_rect.height > right_image_rect.rows )
  {
    return false;
  }

  cv::Mat search_region = right_image_rect( search_rect );

  // Perform template matching
  cv::Mat result;
  cv::matchTemplate( search_region, template_img, result, cv::TM_CCOEFF_NORMED );

  // Find best match
  double min_val, max_val;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc( result, &min_val, &max_val, &min_loc, &max_loc );

  // Use a threshold for match quality
  if( max_val < 0.7 )
  {
    return false;
  }

  // Compute the matched point in the right image
  right_point_rect = kv::vector_2d(
    search_rect.x + max_loc.x + half_template,
    search_rect.y + max_loc.y + half_template );

  return true;
}

#endif

// -----------------------------------------------------------------------------
kv::vector_2d
manual_measurement_process::priv
::project_left_to_right(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point )
{
  // Unproject the left camera point to normalized image coordinates
  const auto left_intrinsics = left_cam.get_intrinsics();
  const kv::vector_2d normalized_pt = left_intrinsics->unmap( left_point );

  // Convert to homogeneous coordinates and add depth
  kv::vector_3d ray_direction( normalized_pt.x(), normalized_pt.y(), 1.0 );
  ray_direction.normalize();

  // Compute 3D point at default depth in left camera coordinates
  kv::vector_3d point_3d_left_cam = ray_direction * m_default_depth;

  // Transform to world coordinates
  const auto& left_rotation = left_cam.rotation();
  const auto& left_center = left_cam.center();
  kv::vector_3d point_3d_world = left_rotation.inverse() * point_3d_left_cam + left_center;

  // Transform to right camera coordinates
  const auto& right_rotation = right_cam.rotation();
  const auto& right_center = right_cam.center();
  kv::vector_3d point_3d_right_cam = right_rotation * ( point_3d_world - right_center );

  // Project to right camera image
  const auto right_intrinsics = right_cam.get_intrinsics();
  kv::vector_2d normalized_right( point_3d_right_cam.x() / point_3d_right_cam.z(),
                                   point_3d_right_cam.y() / point_3d_right_cam.z() );
  return right_intrinsics->map( normalized_right );
}

// =============================================================================
manual_measurement_process
::manual_measurement_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new manual_measurement_process::priv( this ) )
{
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}


manual_measurement_process
::~manual_measurement_process()
{
}


// -----------------------------------------------------------------------------
void
manual_measurement_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( left_image, optional );
  declare_input_port_using_trait( right_image, optional );

  // -- outputs --
  declare_output_port_using_trait( object_track_set1, required );
  declare_output_port_using_trait( object_track_set2, optional );
  declare_output_port_using_trait( timestamp, optional );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::make_config()
{
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( default_depth );
  declare_config_using_trait( matching_method );
  declare_config_using_trait( template_size );
  declare_config_using_trait( search_range );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_configure()
{
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_default_depth = config_value_using_trait( default_depth );
  d->m_matching_method = config_value_using_trait( matching_method );
  d->m_template_size = config_value_using_trait( template_size );
  d->m_search_range = config_value_using_trait( search_range );
  d->m_calibration = kv::read_stereo_rig( d->m_calibration_file );

  // Ensure template size is odd
  if( d->m_template_size % 2 == 0 )
  {
    d->m_template_size++;
    LOG_WARN( logger(), "Template size must be odd, adjusted to " + std::to_string( d->m_template_size ) );
  }
}

// ----------------------------------------------------------------------------
void
manual_measurement_process
::_init()
{
  this->set_data_checking_level( check_valid );
}

// ----------------------------------------------------------------------------
void
manual_measurement_process
::input_port_undefined( port_t const& port_name )
{
  LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if( !kv::starts_with( port_name, "_" ) )
  {
    // Check for unique port name
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
        port_name,                                 // port name
        object_track_set_port_trait::type_name,    // port type
        required,                                  // port flags
        "object track set input" );

      d->p_port_list.insert( port_name );
    }
  }
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_step()
{
  std::vector< kv::object_track_set_sptr > inputs;
  kv::object_track_set_sptr output;
  kv::timestamp ts;

  for( auto const& port_name : d->p_port_list )
  {
    if( port_name != "timestamp" )
    {
      inputs.push_back(
        grab_from_port_as< kv::object_track_set_sptr >( port_name ) );
    }
    else
    {
      ts = grab_from_port_using_trait( timestamp );
    }
  }

  kv::frame_id_t cur_frame_id = ( ts.has_valid_frame() ?
                                  ts.get_frame() :
                                  d->m_frame_counter );

  d->m_frame_counter++;

  if( inputs.size() != 2 )
  {
    const std::string err = "Currently only 2 camera inputs are supported";
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Identify all input detections across all track sets on the current frame
  typedef std::vector< std::map< kv::track_id_t, kv::detected_object_sptr > > map_t;

  map_t dets( inputs.size() );

  for( unsigned i = 0; i < inputs.size(); ++i )
  {
    for( auto& trk : inputs[i]->tracks() )
    {
      for( auto& state : *trk )
      {
        auto obj_state =
          std::static_pointer_cast< kwiver::vital::object_track_state >( state );

        if( state->frame() == cur_frame_id )
        {
          dets[i][trk->id()] = obj_state->detection();
        }
      }
    }
  }

  // Identify which detections are matched on the current frame
  std::vector< kv::track_id_t > common_ids;
  std::vector< kv::track_id_t > left_only_ids;

  for( auto itr : dets[0] )
  {
    bool found_match = false;

    for( unsigned i = 1; i < inputs.size(); ++i )
    {
      if( dets[i].find( itr.first ) != dets[i].end() )
      {
        found_match = true;
        common_ids.push_back( itr.first );
        break;
      }
    }

    if( found_match )
    {
      LOG_INFO( logger(), "Found match for track ID " + std::to_string( itr.first ) );
    }
    else
    {
      LOG_INFO( logger(), "No match for track ID " + std::to_string( itr.first ) +
                          ", will compute right camera points using " + d->m_matching_method );
      left_only_ids.push_back( itr.first );
    }
  }

  // Get camera references (needed for both matched and left-only detections)
  kv::simple_camera_perspective& left_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->left())));
  kv::simple_camera_perspective& right_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->right())));

  // Run measurement on matched detections
  if( !common_ids.empty() )
  {
    for( const kv::track_id_t& id : common_ids )
    {
      const auto& det1 = dets[0][id];
      const auto& det2 = dets[1][id];

      if( !det1 || !det2 )
      {
        continue;
      }

      const auto& kp1 = det1->keypoints();
      const auto& kp2 = det2->keypoints();

      if( kp1.find( "head" ) == kp1.end() ||
          kp2.find( "head" ) == kp2.end() ||
          kp1.find( "tail" ) == kp1.end() ||
          kp2.find( "tail" ) == kp2.end() )
      {
        continue;
      }

      // Triangulate head keypoint across both cameras
      Eigen::Matrix<float, 2, 1>
        left_head( kp1.at("head")[0], kp1.at("head")[1] ),
        right_head( kp2.at("head")[0], kp2.at("head")[1] );
      auto pp1 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_head, right_head );

      // Triangulate tail keypoint across both cameras
      Eigen::Matrix<float, 2, 1>
        left_tail( kp1.at("tail")[0], kp1.at("tail")[1] ),
        right_tail( kp2.at("tail")[0], kp2.at("tail")[1] );
      auto pp2 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_tail, right_tail );

      const double length = ( pp2 - pp1 ).norm();

      LOG_INFO( logger(), "Computed Length: " + std::to_string( length ) );

      det1->set_length( length );
      det2->set_length( length );
    }
  }

  // Run measurement on left-only detections
  if( !left_only_ids.empty() )
  {
    // Get images if using feature matching
    cv::Mat left_image_rect, right_image_rect;
    bool use_feature_matching = ( d->m_matching_method == "feature_matching" );

    if( use_feature_matching )
    {
#ifdef VIAME_ENABLE_OPENCV
      // Read images from ports
      kv::image_container_sptr left_image_container;
      kv::image_container_sptr right_image_container;

      if( has_input_port_edge_using_trait( left_image ) &&
          has_input_port_edge_using_trait( right_image ) )
      {
        left_image_container = grab_from_port_using_trait( left_image );
        right_image_container = grab_from_port_using_trait( right_image );

        // Convert to OpenCV format and grayscale
        cv::Mat left_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
          left_image_container->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
        cv::Mat right_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
          right_image_container->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

        // Convert to grayscale if needed
        if( left_cv.channels() > 1 )
        {
          cv::cvtColor( left_cv, left_cv, cv::COLOR_BGR2GRAY );
          cv::cvtColor( right_cv, right_cv, cv::COLOR_BGR2GRAY );
        }

        // Compute rectification maps if needed
        d->compute_rectification_maps( left_cam, right_cam, left_cv.size() );

        // Rectify images
        cv::remap( left_cv, left_image_rect, d->m_rectification_map_left_x,
                   d->m_rectification_map_left_y, cv::INTER_LINEAR );
        cv::remap( right_cv, right_image_rect, d->m_rectification_map_right_x,
                   d->m_rectification_map_right_y, cv::INTER_LINEAR );
#else
        LOG_ERROR( logger(), "Code not compiled with rectification support" );
        use_feature_matching = false;
#endif
      }
      else
      {
        LOG_WARN( logger(), "Feature matching requested but images not provided, "
                            "falling back to depth projection" );
        use_feature_matching = false;
      }
    }

    for( const kv::track_id_t& id : left_only_ids )
    {
      const auto& det1 = dets[0][id];

      if( !det1 )
      {
        continue;
      }

      const auto& kp1 = det1->keypoints();

      if( kp1.find( "head" ) == kp1.end() ||
          kp1.find( "tail" ) == kp1.end() )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) + " " +
                            "missing required keypoints (head/tail)" );
        continue;
      }

      kv::vector_2d left_head_point( kp1.at("head")[0], kp1.at("head")[1] );
      kv::vector_2d left_tail_point( kp1.at("tail")[0], kp1.at("tail")[1] );
      kv::vector_2d right_head_point, right_tail_point;

      if( use_feature_matching )
      {
        // Rectify left keypoints using remap maps
        // Sample the rectification map at the point location
        int x_head = static_cast<int>( left_head_point.x() + 0.5 );
        int y_head = static_cast<int>( left_head_point.y() + 0.5 );
        int x_tail = static_cast<int>( left_tail_point.x() + 0.5 );
        int y_tail = static_cast<int>( left_tail_point.y() + 0.5 );

        // Check bounds
        if( x_head < 0 || x_head >= d->m_rectification_map_left_x.cols ||
            y_head < 0 || y_head >= d->m_rectification_map_left_x.rows ||
            x_tail < 0 || x_tail >= d->m_rectification_map_left_x.cols ||
            y_tail < 0 || y_tail >= d->m_rectification_map_left_x.rows )
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " keypoints out of bounds, skipping" );
          continue;
        }

        float rect_x_head = d->m_rectification_map_left_x.at<float>( y_head, x_head );
        float rect_y_head = d->m_rectification_map_left_y.at<float>( y_head, x_head );
        float rect_x_tail = d->m_rectification_map_left_x.at<float>( y_tail, x_tail );
        float rect_y_tail = d->m_rectification_map_left_y.at<float>( y_tail, x_tail );

        kv::vector_2d left_head_rect( rect_x_head, rect_y_head );
        kv::vector_2d left_tail_rect( rect_x_tail, rect_y_tail );

        // Find corresponding points in right image using template matching
        kv::vector_2d right_head_rect, right_tail_rect;
        bool head_found = d->find_corresponding_point_template_matching(
          left_image_rect, right_image_rect, left_head_rect, right_head_rect );
        bool tail_found = d->find_corresponding_point_template_matching(
          left_image_rect, right_image_rect, left_tail_rect, right_tail_rect );

        if( !head_found || !tail_found )
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " feature matching failed, skipping" );
          continue;
        }

        // Unrectify right points back to original image coordinates
        right_head_point = d->unrectify_point( right_head_rect, true, right_cam );
        right_tail_point = d->unrectify_point( right_tail_rect, true, right_cam );

        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " matched using template matching" );
      }
      else
      {
        // Project left camera keypoints to right camera using default depth
        right_head_point = d->project_left_to_right( left_cam, right_cam, left_head_point );
        right_tail_point = d->project_left_to_right( left_cam, right_cam, left_tail_point );
      }

      // Triangulate head keypoint
      Eigen::Matrix<float, 2, 1>
        left_head( kp1.at("head")[0], kp1.at("head")[1] ),
        right_head( right_head_point.x(), right_head_point.y() );
      auto pp1 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_head, right_head );

      // Triangulate tail keypoint
      Eigen::Matrix<float, 2, 1>
        left_tail( kp1.at("tail")[0], kp1.at("tail")[1] ),
        right_tail( right_tail_point.x(), right_tail_point.y() );
      auto pp2 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_tail, right_tail );

      const double length = ( pp2 - pp1 ).norm();

      LOG_INFO( logger(), "Computed Length (left-only, " + d->m_matching_method + "): " +
                          std::to_string( length ) );

      det1->set_length( length );
    }
  }

  // Push outputs
  push_to_port_using_trait( object_track_set1, inputs[0] );
  push_to_port_using_trait( object_track_set2, inputs[1] );
  push_to_port_using_trait( timestamp, ts );
}

} // end namespace core

} // end namespace viame
