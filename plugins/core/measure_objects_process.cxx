/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo measurement process implementation
 */

#include <string>
#include <vector>
#include <map>
#include <set>

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

#include <sprokit/processes/kwiver_type_traits.h>

#include "measure_objects_process.h"
#include "measurement_utilities.h"
#include "pair_stereo_detections.h"
#include "camera_rig_io.h"

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

// Only calibration_file is process-specific; rest comes from map_keypoints_to_camera_settings
create_config_trait( calibration_file, std::string, "",
  "Input filename for the calibration file to use" );

create_port_trait( object_track_set1, object_track_set,
  "The stereo filtered object tracks1.")
create_port_trait( object_track_set2, object_track_set,
  "The stereo filtered object tracks2.")
create_port_trait( disparity_image, image,
  "Disparity map (input for external_disparity method, output from compute_disparity method).")
create_port_trait( rectified_left_image, image,
  "Output rectified left image.")
create_port_trait( rectified_right_image, image,
  "Output rectified right image.")

// =============================================================================
// Private implementation class
class measure_objects_process::priv
{
public:
  explicit priv( measure_objects_process* parent );
  ~priv();

  // Process-specific config
  std::string m_calibration_file;

  // Measurement settings (contains all algo parameters and algorithm pointers)
  map_keypoints_to_camera_settings m_settings;

  // Parsed matching methods (derived from settings)
  std::vector< std::string > m_matching_methods;

  // Other variables
  kv::camera_rig_stereo_sptr m_calibration;
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  measure_objects_process* parent;

  // Measurement utilities
  map_keypoints_to_camera m_utilities;

  // Persistent right tracks created for left-only detections across frames
  std::map< kv::track_id_t, kv::track_sptr > m_created_right_tracks;
};


// -----------------------------------------------------------------------------
measure_objects_process::priv
::priv( measure_objects_process* ptr )
  : m_calibration_file( "" )
  , m_calibration()
  , m_frame_counter( 0 )
  , parent( ptr )
{
}


measure_objects_process::priv
::~priv()
{
}

// =============================================================================
measure_objects_process
::measure_objects_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new measure_objects_process::priv( this ) )
{
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}


measure_objects_process
::~measure_objects_process()
{
}


// -----------------------------------------------------------------------------
void
measure_objects_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( disparity_image, optional );

  // -- outputs --
  declare_output_port_using_trait( object_track_set1, required );
  declare_output_port_using_trait( object_track_set2, optional );
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( disparity_image, optional );
  declare_output_port_using_trait( rectified_left_image, optional );
  declare_output_port_using_trait( rectified_right_image, optional );
}

// -----------------------------------------------------------------------------
void
measure_objects_process
::make_config()
{
  // Process-specific config
  declare_config_using_trait( calibration_file );

  // Merge in map_keypoints_to_camera_settings configuration
  kv::config_block_sptr settings_config = d->m_settings.get_configuration();
  for( auto const& key : settings_config->available_values() )
  {
    declare_configuration_key(
      key,
      settings_config->get_value< std::string >( key ),
      settings_config->get_description( key ) );
  }
}

// -----------------------------------------------------------------------------
void
measure_objects_process
::_configure()
{
  // Get process-specific config
  d->m_calibration_file = config_value_using_trait( calibration_file );

  if( d->m_calibration_file.empty() )
  {
    LOG_ERROR( logger(), "Calibration file not specified" );
    throw std::runtime_error( "Calibration file not specified" );
  }

  d->m_calibration = viame::read_stereo_rig( d->m_calibration_file );

  if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
  {
    LOG_ERROR( logger(), "Failed to load calibration file: " + d->m_calibration_file );
    throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
  }

  // Configure settings from config block
  d->m_settings.set_configuration( get_config() );

  // Validate matching methods
  std::string validation_error = d->m_settings.validate_matching_methods();
  if( !validation_error.empty() )
  {
    LOG_ERROR( logger(), validation_error );
    throw std::runtime_error( validation_error );
  }

  // Get parsed methods
  d->m_matching_methods = d->m_settings.get_matching_methods();

  LOG_INFO( logger(), "Matching methods (in order): " + d->m_settings.matching_methods );

  // Configure utilities from settings
  d->m_utilities.configure( d->m_settings );

  // Check for warnings about feature algorithms
  auto warnings = d->m_settings.check_feature_algorithm_warnings();
  for( const auto& warning : warnings )
  {
    LOG_WARN( logger(), warning );
  }
}

// ----------------------------------------------------------------------------
void
measure_objects_process
::_init()
{
  this->set_data_checking_level( check_valid );
}

// ----------------------------------------------------------------------------
void
measure_objects_process
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
measure_objects_process
::_step()
{
  std::vector< kv::object_track_set_sptr > input_tracks;
  std::vector< kv::image_container_sptr > input_images;
  kv::image_container_sptr external_disparity;
  kv::timestamp ts;

  // Grab optional disparity image if connected
  if( has_input_port_edge_using_trait( disparity_image ) )
  {
    external_disparity = grab_from_port_using_trait( disparity_image );
  }

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

  if( input_tracks.size() == 1 )
  {
    // Single track input: create empty 2nd track set so all tracks are
    // treated as left-only and go through the stereo matching path.
    input_tracks.push_back( std::make_shared< kv::object_track_set >() );
  }
  else if( input_tracks.size() != 2 )
  {
    const std::string err = "Expected 1 or 2 track inputs, got "
                            + std::to_string( input_tracks.size() );
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Check if images are required but not provided
  bool images_required = false;
  for( const auto& method : d->m_matching_methods )
  {
    if( method_requires_images( method ) )
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

  // Collect right-only IDs (in right but not in left)
  std::vector< kv::track_id_t > right_only_ids;
  for( auto itr : dets[1] )
  {
    if( dets[0].find( itr.first ) == dets[0].end() )
    {
      right_only_ids.push_back( itr.first );
    }
  }

  // Get camera references
  kv::simple_camera_perspective& left_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->left())));
  kv::simple_camera_perspective& right_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->right())));

  // Detection pairing: match left-only and right-only detections
  if( !d->m_settings.detection_pairing_method.empty() &&
      !left_only_ids.empty() && !right_only_ids.empty() )
  {
    // Build detection vectors from left-only and right-only IDs
    std::vector< kv::detected_object_sptr > left_only_dets;
    std::vector< kv::detected_object_sptr > right_only_dets;

    for( const auto& id : left_only_ids )
    {
      left_only_dets.push_back( dets[0][id] );
    }
    for( const auto& id : right_only_ids )
    {
      right_only_dets.push_back( dets[1][id] );
    }

    std::vector< std::pair< int, int > > paired;

    if( d->m_settings.detection_pairing_method == "epipolar_iou" )
    {
      epipolar_iou_matching_options opts;
      opts.iou_threshold = d->m_settings.detection_pairing_threshold;
      opts.default_depth = d->m_settings.default_depth;

      paired = find_stereo_matches_epipolar_iou(
        left_only_dets, right_only_dets, left_cam, right_cam, opts, logger() );
    }
    else if( d->m_settings.detection_pairing_method == "keypoint_projection" )
    {
      keypoint_projection_matching_options opts;
      opts.max_keypoint_distance = d->m_settings.detection_pairing_threshold;
      opts.default_depth = d->m_settings.default_depth;

      paired = find_stereo_matches_keypoint_projection(
        left_only_dets, right_only_dets, left_cam, right_cam, opts, logger() );
    }
    else
    {
      LOG_WARN( logger(), "Unknown detection_pairing_method: " +
                d->m_settings.detection_pairing_method );
    }

    // Merge paired detections: insert right det under left track ID
    std::set< kv::track_id_t > paired_left_ids;
    for( const auto& p : paired )
    {
      kv::track_id_t left_id = left_only_ids[p.first];
      kv::track_id_t right_id = right_only_ids[p.second];

      // Insert right detection under the left track ID
      dets[1][left_id] = dets[1][right_id];

      // Erase old right-only entry if IDs differ
      if( left_id != right_id )
      {
        dets[1].erase( right_id );
      }

      common_ids.push_back( left_id );
      paired_left_ids.insert( left_id );

      LOG_INFO( logger(), "Paired left track " + std::to_string( left_id ) +
                " with right track " + std::to_string( right_id ) +
                " via " + d->m_settings.detection_pairing_method );
    }

    // Rebuild left_only_ids without paired entries
    std::vector< kv::track_id_t > remaining_left;
    for( const auto& id : left_only_ids )
    {
      if( paired_left_ids.find( id ) == paired_left_ids.end() )
      {
        remaining_left.push_back( id );
      }
    }
    left_only_ids = remaining_left;
  }

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

    const auto measurement = viame::core::compute_stereo_measurement(
      left_cam, right_cam, left_head, right_head, left_tail, right_tail );

    LOG_INFO( logger(), "Computed Length (input_kps_used): " + std::to_string( measurement.length ) );
    LOG_INFO( logger(), "  Midpoint (x,y,z): (" + std::to_string( measurement.x ) + ", "
              + std::to_string( measurement.y ) + ", " + std::to_string( measurement.z ) + ")" );
    LOG_INFO( logger(), "  Range: " + std::to_string( measurement.range ) +
              ", RMS: " + std::to_string( measurement.rms ) );

    if( d->m_settings.record_stereo_method )
    {
      det1->add_note( ":stereo_method=input_kps_used" );
      det2->add_note( ":stereo_method=input_kps_used" );
    }

    add_measurement_attributes( det1, measurement );
    add_measurement_attributes( det2, measurement );
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

    // Pre-compute DINO crop regions if using DINO descriptor
    if( d->m_utilities.epipolar_descriptor_type() == "dino" &&
        left_image && right_image )
    {
      std::vector< kv::vector_2d > all_heads, all_tails;
      for( const kv::track_id_t& id : ids_needing_matching )
      {
        const auto& det = dets[0][id];
        if( !det ) continue;
        const auto& kp = det->keypoints();
        if( kp.find( "head" ) != kp.end() && kp.find( "tail" ) != kp.end() )
        {
          all_heads.push_back( kv::vector_2d( kp.at("head")[0], kp.at("head")[1] ) );
          all_tails.push_back( kv::vector_2d( kp.at("tail")[0], kp.at("tail")[1] ) );
        }
      }
      if( !all_heads.empty() )
      {
        d->m_utilities.precompute_dino_crops(
          left_cam, right_cam, all_heads, all_tails, left_image, right_image );
      }
    }

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
        left_image, right_image, external_disparity );

      if( !result.success )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " no matching method succeeded, skipping" );
        continue;
      }

      // Per-keypoint depth consistency check: if both keypoints matched,
      // triangulate each separately and reject any whose depth is
      // inconsistent with the other. Valid stereo matches on the same
      // object should have similar depths (ratio close to 1.0).
      double max_depth_ratio = d->m_settings.depth_consistency_max_ratio;

      if( max_depth_ratio > 0 && result.head_found && result.tail_found )
      {
        auto head_3d = viame::core::triangulate_point(
          left_cam, right_cam, result.left_head, result.right_head );
        auto tail_3d = viame::core::triangulate_point(
          left_cam, right_cam, result.left_tail, result.right_tail );

        double depth_head = head_3d.z();
        double depth_tail = tail_3d.z();

        if( depth_head > 0 && depth_tail > 0 )
        {
          double depth_ratio = std::max( depth_head, depth_tail ) /
                               std::min( depth_head, depth_tail );

          if( depth_ratio > max_depth_ratio )
          {
            // Keep the keypoint with smaller depth (closer, higher disparity)
            // as that is more likely to be the correct match
            if( depth_head > depth_tail )
            {
              result.head_found = false;
              LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                " depth check: rejecting head (depth " +
                std::to_string( depth_head ) + " vs tail " +
                std::to_string( depth_tail ) + ", ratio " +
                std::to_string( depth_ratio ) + ")" );
            }
            else
            {
              result.tail_found = false;
              LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                " depth check: rejecting tail (depth " +
                std::to_string( depth_tail ) + " vs head " +
                std::to_string( depth_head ) + ", ratio " +
                std::to_string( depth_ratio ) + ")" );
            }
            result.success = result.head_found || result.tail_found;

            if( !result.success )
            {
              LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                " depth check: both keypoints rejected, skipping" );
              continue;
            }
          }
        }
      }

      bool both_found = result.head_found && result.tail_found;

      if( both_found )
      {
        // Compute full measurement (requires both keypoints)
        const auto measurement = viame::core::compute_stereo_measurement(
          left_cam, right_cam,
          result.left_head, result.right_head,
          result.left_tail, result.right_tail );

        LOG_INFO( logger(), "Computed Length (" + result.method_used + "): " +
                            std::to_string( measurement.length ) );
        LOG_INFO( logger(), "  Midpoint (x,y,z): (" + std::to_string( measurement.x ) + ", "
                  + std::to_string( measurement.y ) + ", " + std::to_string( measurement.z ) + ")" );
        LOG_INFO( logger(), "  Range: " + std::to_string( measurement.range ) +
                  ", RMS: " + std::to_string( measurement.rms ) );

        if( d->m_settings.record_stereo_method )
        {
          det1->add_note( ":stereo_method=" + result.method_used );
        }

        add_measurement_attributes( det1, measurement );
      }
      else
      {
        // Partial match: record method but no length measurement
        std::string matched_kp = result.head_found ? "head" : "tail";
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " partial match (" + result.method_used + "): only " +
                            matched_kp + " found, no length measurement" );

        if( d->m_settings.record_stereo_method )
        {
          det1->add_note( ":stereo_method=" + result.method_used + "_partial" );
        }
      }

      if( is_left_only )
      {
        // Create right detection with available keypoints
        kv::vector_2d bbox_pt1, bbox_pt2;

        if( both_found )
        {
          bbox_pt1 = result.right_head;
          bbox_pt2 = result.right_tail;
        }
        else if( result.head_found )
        {
          // Single point: create small bbox around matched keypoint
          bbox_pt1 = kv::vector_2d( result.right_head.x() - 10, result.right_head.y() - 10 );
          bbox_pt2 = kv::vector_2d( result.right_head.x() + 10, result.right_head.y() + 10 );
        }
        else
        {
          bbox_pt1 = kv::vector_2d( result.right_tail.x() - 10, result.right_tail.y() - 10 );
          bbox_pt2 = kv::vector_2d( result.right_tail.x() + 10, result.right_tail.y() + 10 );
        }

        kv::bounding_box_d right_bbox = both_found
          ? d->m_utilities.compute_bbox_from_keypoints( bbox_pt1, bbox_pt2 )
          : kv::bounding_box_d( bbox_pt1.x(), bbox_pt1.y(), bbox_pt2.x(), bbox_pt2.y() );

        auto det2 = std::make_shared< kv::detected_object >( right_bbox );

        if( result.head_found )
          det2->add_keypoint( "head", kv::point_2d( result.right_head.x(), result.right_head.y() ) );
        if( result.tail_found )
          det2->add_keypoint( "tail", kv::point_2d( result.right_tail.x(), result.right_tail.y() ) );

        if( det1->type() )
        {
          det2->set_type( det1->type() );
        }
        det2->set_confidence( det1->confidence() );

        if( d->m_settings.record_stereo_method )
        {
          det2->add_note( ":stereo_method=" + result.method_used +
                          ( both_found ? "" : "_partial" ) );
        }

        if( both_found )
        {
          const auto measurement = viame::core::compute_stereo_measurement(
            left_cam, right_cam,
            result.left_head, result.right_head,
            result.left_tail, result.right_tail );
          add_measurement_attributes( det2, measurement );
        }

        // Look up or create a persistent right track for this left-only ID
        kv::track_sptr right_track;
        auto it = d->m_created_right_tracks.find( id );

        if( it != d->m_created_right_tracks.end() )
        {
          right_track = it->second;
        }
        else
        {
          right_track = kv::track::create();
          right_track->set_id( id );
          d->m_created_right_tracks[id] = right_track;
        }

        kv::time_usec_t time_usec = ts.has_valid_time() ? ts.get_time_usec() : 0;
        auto new_state = std::make_shared< kv::object_track_state >(
          cur_frame_id, time_usec, det2 );
        right_track->append( std::move( new_state ) );

        LOG_INFO( logger(), "Created right detection for track ID " + std::to_string( id ) +
                  ( both_found ? "" : " (partial)" ) );
      }
      else
      {
        auto& det2 = dets[1][id];

        if( result.head_found )
          det2->add_keypoint( "head", kv::point_2d( result.right_head.x(), result.right_head.y() ) );
        if( result.tail_found )
          det2->add_keypoint( "tail", kv::point_2d( result.right_tail.x(), result.right_tail.y() ) );

        if( d->m_settings.record_stereo_method )
        {
          det2->add_note( ":stereo_method=" + result.method_used +
                          ( both_found ? "" : "_partial" ) );
        }

        if( both_found )
        {
          const auto measurement = viame::core::compute_stereo_measurement(
            left_cam, right_cam,
            result.left_head, result.right_head,
            result.left_tail, result.right_tail );
          add_measurement_attributes( det2, measurement );
        }
      }
    }

    // Clear DINO crop state after the detection loop
    d->m_utilities.clear_dino_crop_info();
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

  // Merge persistent right tracks (created for left-only detections) into output
  for( const auto& pair : d->m_created_right_tracks )
  {
    if( !input_tracks[1]->get_track( pair.first ) )
    {
      input_tracks[1]->insert( pair.second );
    }
  }

  // Push outputs
  push_to_port_using_trait( object_track_set1, input_tracks[0] );
  push_to_port_using_trait( object_track_set2, input_tracks[1] );
  push_to_port_using_trait( timestamp, ts );

  // Push disparity image if computed
  kv::image_container_sptr computed_disparity = d->m_utilities.get_cached_disparity();
  push_to_port_using_trait( disparity_image, computed_disparity );

  // Push rectified images if available
  kv::image_container_sptr rectified_left = d->m_utilities.get_cached_rectified_left();
  kv::image_container_sptr rectified_right = d->m_utilities.get_cached_rectified_right();
  push_to_port_using_trait( rectified_left_image, rectified_left );
  push_to_port_using_trait( rectified_right_image, rectified_right );
}

} // end namespace core

} // end namespace viame
