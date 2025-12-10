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
#include "measurement_utilities.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/image_container.h>
#include <vital/types/vector.h>
#include <vital/types/point.h>
#include <vital/types/bounding_box.h>
#include <vital/types/track.h>
#include <vital/util/string.h>
#include <vital/io/camera_rig_io.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_fundamental_matrix.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <string>
#include <vector>
#include <map>
#include <set>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( calibration_file, std::string, "",
  "Input filename for the calibration file to use" );

create_config_trait( matching_methods, std::string, "input_pairs_only,template_matching",
  "Comma-separated list of methods to try (in order) for finding corresponding points "
  "in right camera for left-only tracks. Methods will be tried in the order specified "
  "until one succeeds. Valid options: "
  "'input_pairs_only' (use existing keypoints from right camera if available), "
  "'depth_projection' (uses default_depth to project points), "
  "'template_matching' (rectifies images and searches along epipolar lines), "
  "'sgbm_disparity' (uses Semi-Global Block Matching to compute disparity map), "
  "'feature_descriptor' (uses vital feature detection/descriptor/matching), "
  "'ransac_feature' (feature matching with RANSAC-based fundamental matrix filtering). "
  "Example: 'input_pairs_only,template_matching,depth_projection'" );

create_config_trait( template_size, int, "31",
  "Template window size (in pixels) for template matching. Must be odd number." );

create_config_trait( search_range, int, "128",
  "Search range (in pixels) along epipolar line for template matching." );

create_config_trait( default_depth, double, "5.0",
  "Default depth (in meters) to use when projecting left camera points to right camera "
  "for tracks that only exist in the left camera, when using the depth_projection option" );

create_config_trait( use_distortion, bool, "true",
  "Whether to use distortion coefficients from the calibration during rectification. "
  "If true, distortion coefficients from the calibration file are used. "
  "If false, zero distortion is assumed." );

create_config_trait( sgbm_min_disparity, int, "0",
  "Minimum possible disparity value for SGBM. Normally 0, but can be negative." );

create_config_trait( sgbm_num_disparities, int, "128",
  "Maximum disparity minus minimum disparity for SGBM. Must be divisible by 16." );

create_config_trait( sgbm_block_size, int, "5",
  "Block size for SGBM. Must be odd number >= 1. Typically 3-11." );

create_config_trait( feature_search_radius, double, "50.0",
  "Maximum distance (in pixels) to search for feature matches around the expected location. "
  "Used for feature_descriptor and ransac_feature methods." );

create_config_trait( ransac_inlier_scale, double, "3.0",
  "Inlier threshold for RANSAC fundamental matrix estimation. "
  "Points with reprojection error below this threshold are considered inliers." );

create_config_trait( min_ransac_inliers, int, "10",
  "Minimum number of inliers required for a valid RANSAC result." );

create_config_trait( box_scale_factor, double, "1.10",
  "Scale factor to expand the bounding box around keypoints when creating "
  "new detections for the right image. A value of 1.10 means 10% expansion." );

create_config_trait( record_stereo_method, bool, "true",
  "If true, record the stereo measurement method used as an attribute on each "
  "output detection object. The attribute will be ':stereo_method=METHOD' "
  "where METHOD is one of: input_kps_used, template_matching, sgbm_disparity, "
  "feature_descriptor, ransac_feature, or depth_projection." );

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

  // Configuration settings
  std::string m_calibration_file;
  std::string m_matching_methods_str;
  std::vector< std::string > m_matching_methods;
  bool m_record_stereo_method;

  // Other variables
  kv::camera_rig_stereo_sptr m_calibration;
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  manual_measurement_process* parent;

  // Optional vital algorithms for feature-based matching
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;
  kv::algo::estimate_fundamental_matrix_sptr m_fundamental_matrix_estimator;

  // Measurement utilities
  measurement_utilities m_utilities;
};


// -----------------------------------------------------------------------------
manual_measurement_process::priv
::priv( manual_measurement_process* ptr )
  : m_calibration_file( "" )
  , m_matching_methods_str( "input_pairs_only,template_matching" )
  , m_record_stereo_method( true )
  , m_calibration()
  , m_frame_counter( 0 )
  , parent( ptr )
{
}


manual_measurement_process::priv
::~priv()
{
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
  declare_config_using_trait( matching_methods );
  declare_config_using_trait( template_size );
  declare_config_using_trait( search_range );
  declare_config_using_trait( use_distortion );

  // SGBM configuration
  declare_config_using_trait( sgbm_min_disparity );
  declare_config_using_trait( sgbm_num_disparities );
  declare_config_using_trait( sgbm_block_size );

  // Feature matching configuration
  declare_config_using_trait( feature_search_radius );
  declare_config_using_trait( ransac_inlier_scale );
  declare_config_using_trait( min_ransac_inliers );

  // Right detection creation configuration
  declare_config_using_trait( box_scale_factor );

  // Recording options
  declare_config_using_trait( record_stereo_method );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_configure()
{
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_matching_methods_str = config_value_using_trait( matching_methods );
  d->m_record_stereo_method = config_value_using_trait( record_stereo_method );

  if( d->m_calibration_file.empty() )
  {
    LOG_ERROR( logger(), "Calibration file not specified" );
    throw std::runtime_error( "Calibration file not specified" );
  }

  d->m_calibration = kv::read_stereo_rig( d->m_calibration_file );

  if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
  {
    LOG_ERROR( logger(), "Failed to load calibration file: " + d->m_calibration_file );
    throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
  }

  // Parse matching methods
  d->m_matching_methods = measurement_utilities::parse_matching_methods( d->m_matching_methods_str );

  if( d->m_matching_methods.empty() )
  {
    LOG_ERROR( logger(), "No valid matching methods specified" );
    throw std::runtime_error( "No valid matching methods specified" );
  }

  // Validate method names
  auto valid_methods = measurement_utilities::get_valid_methods();
  for( const auto& method : d->m_matching_methods )
  {
    if( std::find( valid_methods.begin(), valid_methods.end(), method ) == valid_methods.end() )
    {
      LOG_ERROR( logger(), "Invalid matching method: " + method );
      throw std::runtime_error( "Invalid matching method: " + method );
    }
  }

  LOG_INFO( logger(), "Matching methods (in order): " + d->m_matching_methods_str );

  // Configure utilities
  d->m_utilities.set_default_depth( config_value_using_trait( default_depth ) );
  d->m_utilities.set_template_params(
    config_value_using_trait( template_size ),
    config_value_using_trait( search_range ) );
  d->m_utilities.set_use_distortion( config_value_using_trait( use_distortion ) );
  d->m_utilities.set_sgbm_params(
    config_value_using_trait( sgbm_min_disparity ),
    config_value_using_trait( sgbm_num_disparities ),
    config_value_using_trait( sgbm_block_size ) );
  d->m_utilities.set_feature_params(
    config_value_using_trait( feature_search_radius ),
    config_value_using_trait( ransac_inlier_scale ),
    config_value_using_trait( min_ransac_inliers ) );
  d->m_utilities.set_box_scale_factor( config_value_using_trait( box_scale_factor ) );

  // Configure optional vital algorithms for feature-based methods
  kv::config_block_sptr algo_config = get_config();

  kv::algo::detect_features::set_nested_algo_configuration(
    "feature_detector", algo_config, d->m_feature_detector );
  kv::algo::extract_descriptors::set_nested_algo_configuration(
    "descriptor_extractor", algo_config, d->m_descriptor_extractor );
  kv::algo::match_features::set_nested_algo_configuration(
    "feature_matcher", algo_config, d->m_feature_matcher );
  kv::algo::estimate_fundamental_matrix::set_nested_algo_configuration(
    "fundamental_matrix_estimator", algo_config, d->m_fundamental_matrix_estimator );

  d->m_utilities.set_feature_algorithms(
    d->m_feature_detector,
    d->m_descriptor_extractor,
    d->m_feature_matcher,
    d->m_fundamental_matrix_estimator );

  // Check if feature-based methods are requested but algorithms are not configured
  for( const auto& method : d->m_matching_methods )
  {
    if( method == "feature_descriptor" || method == "ransac_feature" )
    {
      if( !d->m_feature_detector )
      {
        LOG_WARN( logger(), "Feature detector not configured; " + method + " method may not work" );
      }
      if( !d->m_descriptor_extractor )
      {
        LOG_WARN( logger(), "Descriptor extractor not configured; " + method + " method may not work" );
      }
      if( !d->m_feature_matcher )
      {
        LOG_WARN( logger(), "Feature matcher not configured; " + method + " method may not work" );
      }
      if( method == "ransac_feature" && !d->m_fundamental_matrix_estimator )
      {
        LOG_WARN( logger(), "Fundamental matrix estimator not configured; ransac_feature method may not work" );
      }
    }
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

  if( !kv::starts_with( port_name, "_" ) )
  {
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      if( port_name.find( "image" ) != std::string::npos )
      {
        declare_input_port(
          port_name,
          image_port_trait::type_name,
          required,
          "image container input" );
      }
      else
      {
        declare_input_port(
          port_name,
          object_track_set_port_trait::type_name,
          required,
          "object track set input" );
      }

      d->p_port_list.insert( port_name );
    }
  }
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_step()
{
  std::vector< kv::object_track_set_sptr > input_tracks;
  std::vector< kv::image_container_sptr > input_images;
  kv::timestamp ts;

  // Read port names
  for( auto const& port_name : d->p_port_list )
  {
    if( port_name == "timestamp" )
    {
      ts = grab_from_port_using_trait( timestamp );
    }
    else if( port_name.find( "image" ) != std::string::npos )
    {
      input_images.push_back(
        grab_from_port_as< kv::image_container_sptr >( port_name ) );
    }
    else
    {
      input_tracks.push_back(
        grab_from_port_as< kv::object_track_set_sptr >( port_name ) );
    }
  }

  kv::frame_id_t cur_frame_id = ( ts.has_valid_frame() ?
                                  ts.get_frame() :
                                  d->m_frame_counter );

  d->m_frame_counter++;

  if( input_tracks.size() != 2 )
  {
    const std::string err = "Currently only 2 camera inputs are supported";
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Check if images are required but not provided
  bool images_required = false;
  for( const auto& method : d->m_matching_methods )
  {
    if( measurement_utilities::method_requires_images( method ) )
    {
      images_required = true;
      break;
    }
  }

  if( images_required && input_images.size() < 2 )
  {
    const std::string err = "Input images are required for the specified matching methods "
                            "but were not provided. Please connect image inputs to the process "
                            "or use only 'input_pairs_only' or 'depth_projection' methods.";
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Identify all input detections
  typedef std::vector< std::map< kv::track_id_t, kv::detected_object_sptr > > map_t;
  map_t dets( input_tracks.size() );

  for( unsigned i = 0; i < input_tracks.size(); ++i )
  {
    if( !input_tracks[i] )
    {
      continue;
    }

    for( auto& trk : input_tracks[i]->tracks() )
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

  // Identify which detections are matched
  std::vector< kv::track_id_t > common_ids;
  std::vector< kv::track_id_t > left_only_ids;

  for( auto itr : dets[0] )
  {
    bool found_match = false;

    for( unsigned i = 1; i < input_tracks.size(); ++i )
    {
      if( dets[i].find( itr.first ) != dets[i].end() )
      {
        found_match = true;
        common_ids.push_back( itr.first );
        break;
      }
    }

    if( !found_match )
    {
      left_only_ids.push_back( itr.first );
    }
  }

  // Get camera references
  kv::simple_camera_perspective& left_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->left())));
  kv::simple_camera_perspective& right_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->right())));

  // Separate matched detections
  std::vector< kv::track_id_t > fully_matched_ids;
  std::vector< kv::track_id_t > missing_right_kps_ids;

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

    bool left_has_kp = ( kp1.find( "head" ) != kp1.end() &&
                         kp1.find( "tail" ) != kp1.end() );
    bool right_has_kp = ( kp2.find( "head" ) != kp2.end() &&
                          kp2.find( "tail" ) != kp2.end() );

    if( left_has_kp && right_has_kp )
    {
      fully_matched_ids.push_back( id );
    }
    else if( left_has_kp && !right_has_kp )
    {
      missing_right_kps_ids.push_back( id );
    }
  }

  // Run measurement on fully matched detections
  for( const kv::track_id_t& id : fully_matched_ids )
  {
    const auto& det1 = dets[0][id];
    const auto& det2 = dets[1][id];

    const auto& kp1 = det1->keypoints();
    const auto& kp2 = det2->keypoints();

    kv::vector_2d left_head( kp1.at("head")[0], kp1.at("head")[1] );
    kv::vector_2d right_head( kp2.at("head")[0], kp2.at("head")[1] );
    kv::vector_2d left_tail( kp1.at("tail")[0], kp1.at("tail")[1] );
    kv::vector_2d right_tail( kp2.at("tail")[0], kp2.at("tail")[1] );

    const double length = d->m_utilities.compute_stereo_length(
      left_cam, right_cam, left_head, right_head, left_tail, right_tail );

    LOG_INFO( logger(), "Computed Length (input_kps_used): " + std::to_string( length ) );

    det1->set_length( length );
    det2->set_length( length );

    if( d->m_record_stereo_method )
    {
      det1->add_note( ":stereo_method=input_kps_used" );
      det2->add_note( ":stereo_method=input_kps_used" );
    }
  }

  // Combine left-only IDs and matched IDs missing right keypoints
  std::vector< kv::track_id_t > ids_needing_matching;
  ids_needing_matching.insert( ids_needing_matching.end(),
                               left_only_ids.begin(), left_only_ids.end() );
  ids_needing_matching.insert( ids_needing_matching.end(),
                               missing_right_kps_ids.begin(),
                               missing_right_kps_ids.end() );

  // Check if input_pairs_only is the only method - skip secondary matching
  bool only_input_pairs = ( d->m_matching_methods.size() == 1 &&
                            d->m_matching_methods[0] == "input_pairs_only" );

  if( !ids_needing_matching.empty() && !only_input_pairs )
  {
    // Update frame ID for caching
    d->m_utilities.set_frame_id( cur_frame_id );

    // Get images for methods that need them
    kv::image_container_sptr left_image = input_images.size() >= 1 ? input_images[0] : nullptr;
    kv::image_container_sptr right_image = input_images.size() >= 2 ? input_images[1] : nullptr;

    for( const kv::track_id_t& id : ids_needing_matching )
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
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " missing required keypoints (head/tail)" );
        continue;
      }

      bool is_left_only = ( dets[1].find( id ) == dets[1].end() );

      kv::vector_2d left_head( kp1.at("head")[0], kp1.at("head")[1] );
      kv::vector_2d left_tail( kp1.at("tail")[0], kp1.at("tail")[1] );

      // Check for existing right keypoints
      const kv::vector_2d* right_head_input = nullptr;
      const kv::vector_2d* right_tail_input = nullptr;
      kv::vector_2d right_head_tmp, right_tail_tmp;

      if( !is_left_only )
      {
        const auto& det2 = dets[1].at( id );
        if( det2 )
        {
          const auto& kp2 = det2->keypoints();
          if( kp2.find( "head" ) != kp2.end() && kp2.find( "tail" ) != kp2.end() )
          {
            right_head_tmp = kv::vector_2d( kp2.at("head")[0], kp2.at("head")[1] );
            right_tail_tmp = kv::vector_2d( kp2.at("tail")[0], kp2.at("tail")[1] );
            right_head_input = &right_head_tmp;
            right_tail_input = &right_tail_tmp;
          }
        }
      }

      // Find stereo correspondences using utility function
      auto result = d->m_utilities.find_stereo_correspondence(
        d->m_matching_methods, left_cam, right_cam,
        left_head, left_tail, right_head_input, right_tail_input,
        left_image, right_image );

      if( !result.success )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " no matching method succeeded, skipping" );
        continue;
      }

      // Compute length
      const double length = d->m_utilities.compute_stereo_length(
        left_cam, right_cam,
        result.left_head, result.right_head,
        result.left_tail, result.right_tail );

      LOG_INFO( logger(), "Computed Length (" + result.method_used + "): " +
                          std::to_string( length ) );

      det1->set_length( length );

      if( d->m_record_stereo_method )
      {
        det1->add_note( ":stereo_method=" + result.method_used );
      }

      if( is_left_only )
      {
        kv::bounding_box_d right_bbox =
          d->m_utilities.compute_bbox_from_keypoints( result.right_head, result.right_tail );

        auto det2 = std::make_shared< kv::detected_object >( right_bbox );
        det2->add_keypoint( "head", kv::point_2d( result.right_head.x(), result.right_head.y() ) );
        det2->add_keypoint( "tail", kv::point_2d( result.right_tail.x(), result.right_tail.y() ) );
        det2->set_length( length );

        if( d->m_record_stereo_method )
        {
          det2->add_note( ":stereo_method=" + result.method_used );
        }

        if( !input_tracks[1] )
        {
          input_tracks[1] = std::make_shared< kv::object_track_set >();
        }

        kv::track_sptr right_track = input_tracks[1]->get_track( id );

        if( !right_track )
        {
          right_track = kv::track::create();
          right_track->set_id( id );
          input_tracks[1]->insert( right_track );
        }

        kv::time_usec_t time_usec = ts.has_valid_time() ? ts.get_time_usec() : 0;
        auto new_state = std::make_shared< kv::object_track_state >(
          cur_frame_id, time_usec, det2 );
        right_track->append( new_state );
        input_tracks[1]->notify_new_state( new_state );

        LOG_INFO( logger(), "Created right detection for track ID " + std::to_string( id ) );
      }
      else
      {
        auto& det2 = dets[1][id];
        det2->add_keypoint( "head", kv::point_2d( result.right_head.x(), result.right_head.y() ) );
        det2->add_keypoint( "tail", kv::point_2d( result.right_tail.x(), result.right_tail.y() ) );
        det2->set_length( length );

        if( d->m_record_stereo_method )
        {
          det2->add_note( ":stereo_method=" + result.method_used );
        }
      }
    }
  }

  // Ensure output track sets exist
  if( !input_tracks[0] )
  {
    input_tracks[0] = std::make_shared< kv::object_track_set >();
  }
  if( !input_tracks[1] )
  {
    input_tracks[1] = std::make_shared< kv::object_track_set >();
  }

  // Push outputs
  push_to_port_using_trait( object_track_set1, input_tracks[0] );
  push_to_port_using_trait( object_track_set2, input_tracks[1] );
  push_to_port_using_trait( timestamp, ts );
}

} // end namespace core

} // end namespace viame
