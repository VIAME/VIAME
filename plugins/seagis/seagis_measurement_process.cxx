/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief SEAGIS stereo measurement process implementation
 */

#include "seagis_measurement_process.h"

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

#include "../core/measurement_utilities.h"
#include "../core/pair_stereo_detections.h"

#ifdef VIAME_ENABLE_OPENCV
  #include <arrows/ocv/image_container.h>
  #include <opencv2/imgproc/imgproc.hpp>
#endif

#include <LX_StereoInterface.h>

#include <string>
#include <vector>
#include <map>
#include <set>
#include <cmath>

namespace kv = kwiver::vital;

namespace viame
{

namespace seagis
{

// Config traits
create_config_trait( left_camera_file, std::string, "",
  "Input filename for the left camera calibration file (.CAM)" );

create_config_trait( right_camera_file, std::string, "",
  "Input filename for the right camera calibration file (.CAM)" );

create_config_trait( licence_key1, std::string, "",
  "SEAGIS licence key 1 (optional if using USB licence)" );

create_config_trait( licence_key2, std::string, "",
  "SEAGIS licence key 2 (optional if using USB licence)" );

create_config_trait( image_measurement_sd, double, "1.0",
  "Image measurement standard deviation in pixels" );

create_config_trait( camera_pair_id, unsigned int, "0",
  "Camera pair ID for SEAGIS library (typically 0)" );

create_config_trait( enable_epipolar_matching, bool, "false",
  "If true, attempt to find corresponding keypoints in the other camera using "
  "SEAGIS epipolar line projection and template matching when only one camera "
  "has head/tail keypoints." );

create_config_trait( epipolar_min_depth, double, "0.0",
  "Minimum depth (in calibration units) for epipolar template matching. "
  "Defines the near end of the depth range sampled along the epipolar line. "
  "Default is 0 (off). Ignored when epipolar_min_disparity and "
  "epipolar_max_disparity are both > 0. Either disparity or depth parameters "
  "must be set for epipolar matching to work." );

create_config_trait( epipolar_max_depth, double, "0.0",
  "Maximum depth (in calibration units) for epipolar template matching. "
  "Defines the far end of the depth range sampled along the epipolar line. "
  "Default is 0 (off). Ignored when epipolar_min_disparity and "
  "epipolar_max_disparity are both > 0. Either disparity or depth parameters "
  "must be set for epipolar matching to work." );

create_config_trait( epipolar_min_disparity, double, "0.0",
  "Minimum expected disparity in pixels for epipolar template matching "
  "(corresponds to the farthest objects). When both epipolar_min_disparity and "
  "epipolar_max_disparity are > 0, the range is computed automatically from the "
  "SEAGIS camera model. This is the recommended way to configure epipolar search "
  "range since disparity is unit-independent and can be estimated directly from "
  "the images." );

create_config_trait( epipolar_max_disparity, double, "0.0",
  "Maximum expected disparity in pixels for epipolar template matching "
  "(corresponds to the nearest objects). See epipolar_min_disparity for details." );

create_config_trait( template_size, int, "13",
  "Template window size (in pixels) for epipolar template matching. Must be odd." );

create_config_trait( template_matching_threshold, double, "0.5",
  "Minimum NCC threshold for epipolar template matching (0.0 to 1.0)." );

create_config_trait( use_census_transform, bool, "false",
  "If true, use census transform preprocessing for epipolar template matching." );

create_config_trait( assume_inputs_paired, bool, "true",
  "If true (default), assume the two input track sets are pre-paired: detections "
  "with the same track ID across cameras are treated as corresponding stereo pairs. "
  "If false, ignore track ID correspondence entirely — remap right-camera IDs to "
  "avoid collisions, then use detection_pairing_method to match left/right "
  "detections. Use false when feeding tracks from two independent trackers." );

create_config_trait( detection_pairing_method, std::string, "",
  "Method for pairing left/right detections that do not share the same track ID. "
  "Set to empty string (default) to disable. Valid options: "
  "'iou' (match by raw bounding box IOU), "
  "'keypoint_distance' (match by raw head/tail keypoint pixel distance). "
  "When assume_inputs_paired is false, this must be set." );

create_config_trait( detection_pairing_threshold, double, "0.1",
  "Threshold for detection pairing. For 'iou' method, this is the minimum IOU "
  "threshold (default 0.1). For 'keypoint_distance' method, this is the maximum "
  "average keypoint pixel distance (default 50.0)." );

create_port_trait( object_track_set1, object_track_set,
  "The stereo filtered object tracks1.")
create_port_trait( object_track_set2, object_track_set,
  "The stereo filtered object tracks2.")

// =============================================================================
// Private implementation class
class seagis_measurement_process::priv
{
public:
  explicit priv( seagis_measurement_process* parent );
  ~priv();

  // Configuration
  std::string m_left_camera_file;
  std::string m_right_camera_file;
  std::string m_licence_key1;
  std::string m_licence_key2;
  double m_image_measurement_sd;
  unsigned int m_camera_pair_id;

  // SEAGIS stereo interface
  std::unique_ptr< CStereoInt > m_stereo;

  // Epipolar matching configuration
  bool m_enable_epipolar_matching;
  double m_epipolar_min_depth;
  double m_epipolar_max_depth;
  double m_epipolar_min_disparity;
  double m_epipolar_max_disparity;

  // Detection pairing configuration
  bool m_assume_inputs_paired;
  std::string m_detection_pairing_method;
  double m_detection_pairing_threshold;

  // Measurement utilities for template matching
  core::map_keypoints_to_camera m_utilities;

  // Other variables
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  seagis_measurement_process* parent;

  // Use shared measurement result type from core
  using measurement_result = core::stereo_measurement_result;

  // Helper functions
  measurement_result compute_measurement(
    const kv::vector_2d& left_head,
    const kv::vector_2d& right_head,
    const kv::vector_2d& left_tail,
    const kv::vector_2d& right_tail );

  // Get epipolar points from SEAGIS for a 2D point in one camera
  std::vector< kv::vector_2d > get_epipolar_points(
    CAMERA_ID cam_id,
    const kv::vector_2d& pt );

#ifdef VIAME_ENABLE_OPENCV
  // Find corresponding point in the other camera using epipolar template matching
  bool find_corresponding_point_epipolar(
    CAMERA_ID source_cam_id,
    const cv::Mat& source_image,
    const cv::Mat& target_image,
    const kv::vector_2d& source_point,
    kv::vector_2d& target_point );
#endif
};


// -----------------------------------------------------------------------------
seagis_measurement_process::priv
::priv( seagis_measurement_process* ptr )
  : m_left_camera_file( "" )
  , m_right_camera_file( "" )
  , m_licence_key1( "" )
  , m_licence_key2( "" )
  , m_image_measurement_sd( 1.0 )
  , m_camera_pair_id( 0 )
  , m_stereo()
  , m_enable_epipolar_matching( false )
  , m_epipolar_min_depth( 0.0 )
  , m_epipolar_max_depth( 0.0 )
  , m_epipolar_min_disparity( 0.0 )
  , m_epipolar_max_disparity( 0.0 )
  , m_assume_inputs_paired( true )
  , m_detection_pairing_method( "" )
  , m_detection_pairing_threshold( 0.1 )
  , m_frame_counter( 0 )
  , parent( ptr )
{
}


seagis_measurement_process::priv
::~priv()
{
}


// -----------------------------------------------------------------------------
seagis_measurement_process::priv::measurement_result
seagis_measurement_process::priv
::compute_measurement(
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail )
{
  measurement_result result;

  // Intersect head points
  C2DPt ptLeftHead( left_head.x(), left_head.y() );
  C2DPt ptRightHead( right_head.x(), right_head.y() );
  C3DPt pt3DHead, pt3DHeadSD;
  double dRMSHead;

  RESULT res = m_stereo->Intersect( m_camera_pair_id, ptLeftHead, ptRightHead,
                                    pt3DHead, dRMSHead, pt3DHeadSD );
  if( res != OK )
  {
    return result;
  }

  // Intersect tail points
  C2DPt ptLeftTail( left_tail.x(), left_tail.y() );
  C2DPt ptRightTail( right_tail.x(), right_tail.y() );
  C3DPt pt3DTail, pt3DTailSD;
  double dRMSTail;

  res = m_stereo->Intersect( m_camera_pair_id, ptLeftTail, ptRightTail,
                             pt3DTail, dRMSTail, pt3DTailSD );
  if( res != OK )
  {
    return result;
  }

  // Compute length (distance between head and tail)
  double dSD;
  result.length = CStereoInt::Distance( pt3DHead, pt3DHeadSD, pt3DTail, pt3DTailSD, dSD );

  // Compute midpoint of head-tail line (real-world location)
  result.x = ( pt3DHead.X() + pt3DTail.X() ) / 2.0;
  result.y = ( pt3DHead.Y() + pt3DTail.Y() ) / 2.0;
  result.z = ( pt3DHead.Z() + pt3DTail.Z() ) / 2.0;

  // Compute range (distance from midpoint to camera origin at 0,0,0)
  result.range = std::sqrt( result.x * result.x +
                            result.y * result.y +
                            result.z * result.z );

  // Compute combined RMS (average of head and tail RMS values)
  result.rms = ( dRMSHead + dRMSTail ) / 2.0;

  result.valid = true;
  return result;
}


// -----------------------------------------------------------------------------
std::vector< kv::vector_2d >
seagis_measurement_process::priv
::get_epipolar_points(
  CAMERA_ID cam_id,
  const kv::vector_2d& pt )
{
  std::vector< kv::vector_2d > points;

  C2DPt image_pt( pt.x(), pt.y() );
  int num_points = 0;

  RESULT res = m_stereo->EpipolarLine( m_camera_pair_id, cam_id,
    image_pt, m_epipolar_min_depth, m_epipolar_max_depth, num_points );

  if( res != OK || num_points <= 0 )
  {
    return points;
  }

  points.reserve( num_points );

  for( int i = 0; i < num_points; ++i )
  {
    C2DPt ep_pt;
    res = m_stereo->GetEpipolarPoint( m_camera_pair_id, i, ep_pt );

    if( res == OK )
    {
      points.push_back( kv::vector_2d( ep_pt.X(), ep_pt.Y() ) );
    }
  }

  return points;
}


#ifdef VIAME_ENABLE_OPENCV
// -----------------------------------------------------------------------------
bool
seagis_measurement_process::priv
::find_corresponding_point_epipolar(
  CAMERA_ID source_cam_id,
  const cv::Mat& source_image,
  const cv::Mat& target_image,
  const kv::vector_2d& source_point,
  kv::vector_2d& target_point )
{
  auto epipolar_points = get_epipolar_points( source_cam_id, source_point );

  if( epipolar_points.empty() )
  {
    return false;
  }

  return m_utilities.find_corresponding_point_epipolar_template_matching(
    source_image, target_image, source_point, epipolar_points, target_point );
}
#endif


// =============================================================================
seagis_measurement_process
::seagis_measurement_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new seagis_measurement_process::priv( this ) )
{
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}


seagis_measurement_process
::~seagis_measurement_process()
{
}


// -----------------------------------------------------------------------------
void
seagis_measurement_process
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
seagis_measurement_process
::make_config()
{
  declare_config_using_trait( left_camera_file );
  declare_config_using_trait( right_camera_file );
  declare_config_using_trait( licence_key1 );
  declare_config_using_trait( licence_key2 );
  declare_config_using_trait( image_measurement_sd );
  declare_config_using_trait( camera_pair_id );
  declare_config_using_trait( enable_epipolar_matching );
  declare_config_using_trait( epipolar_min_depth );
  declare_config_using_trait( epipolar_max_depth );
  declare_config_using_trait( epipolar_min_disparity );
  declare_config_using_trait( epipolar_max_disparity );
  declare_config_using_trait( template_size );
  declare_config_using_trait( template_matching_threshold );
  declare_config_using_trait( use_census_transform );
  declare_config_using_trait( assume_inputs_paired );
  declare_config_using_trait( detection_pairing_method );
  declare_config_using_trait( detection_pairing_threshold );
}

// -----------------------------------------------------------------------------
void
seagis_measurement_process
::_configure()
{
  // Get configuration values
  d->m_left_camera_file = config_value_using_trait( left_camera_file );
  d->m_right_camera_file = config_value_using_trait( right_camera_file );
  d->m_licence_key1 = config_value_using_trait( licence_key1 );
  d->m_licence_key2 = config_value_using_trait( licence_key2 );
  d->m_image_measurement_sd = config_value_using_trait( image_measurement_sd );
  d->m_camera_pair_id = config_value_using_trait( camera_pair_id );
  d->m_enable_epipolar_matching = config_value_using_trait( enable_epipolar_matching );
  d->m_epipolar_min_depth = config_value_using_trait( epipolar_min_depth );
  d->m_epipolar_max_depth = config_value_using_trait( epipolar_max_depth );
  d->m_epipolar_min_disparity = config_value_using_trait( epipolar_min_disparity );
  d->m_epipolar_max_disparity = config_value_using_trait( epipolar_max_disparity );
  d->m_assume_inputs_paired = config_value_using_trait( assume_inputs_paired );
  d->m_detection_pairing_method = config_value_using_trait( detection_pairing_method );
  d->m_detection_pairing_threshold = config_value_using_trait( detection_pairing_threshold );

  if( !d->m_assume_inputs_paired && d->m_detection_pairing_method.empty() )
  {
    LOG_WARN( logger(), "assume_inputs_paired is false but detection_pairing_method "
      "is empty. No detections will be paired across cameras." );
  }

  // Configure template matching utilities
  {
    int tmpl_size = config_value_using_trait( template_size );
    double tmpl_threshold = config_value_using_trait( template_matching_threshold );
    bool census = config_value_using_trait( use_census_transform );

    d->m_utilities.set_template_params(
      tmpl_size, 0 /* search_range unused */, tmpl_threshold,
      0.0 /* disparity unused */, false /* sgbm hint */,
      false /* multires */, 4 /* multires step */,
      census, 0 /* epipolar band */ );
  }

  if( d->m_left_camera_file.empty() )
  {
    LOG_ERROR( logger(), "Left camera file not specified" );
    throw std::runtime_error( "Left camera file not specified" );
  }

  if( d->m_right_camera_file.empty() )
  {
    LOG_ERROR( logger(), "Right camera file not specified" );
    throw std::runtime_error( "Right camera file not specified" );
  }

  // Create SEAGIS stereo interface
  d->m_stereo = std::make_unique< CStereoInt >();

  // Set licence keys if provided
  if( !d->m_licence_key1.empty() && !d->m_licence_key2.empty() )
  {
    if( !d->m_stereo->SetLicenceKeys( d->m_licence_key1, d->m_licence_key2 ) )
    {
      LOG_WARN( logger(), "SEAGIS licence keys invalid, checking for USB licence" );
    }
  }

  // Check licence
  if( !d->m_stereo->LicencePresent() )
  {
    LOG_ERROR( logger(), "SEAGIS licence not present" );
    throw std::runtime_error( "SEAGIS licence not present" );
  }

  // Load camera files
  RESULT res = d->m_stereo->LoadCameraFile( d->m_camera_pair_id, LEFT,
                                            d->m_left_camera_file );
  if( res != OK )
  {
    std::string err = "Failed to load left camera file: " +
                      CStereoInt::ErrorString( res );
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  res = d->m_stereo->LoadCameraFile( d->m_camera_pair_id, RIGHT,
                                     d->m_right_camera_file );
  if( res != OK )
  {
    std::string err = "Failed to load right camera file: " +
                      CStereoInt::ErrorString( res );
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Set image measurement standard deviation
  d->m_stereo->SetImageMeasurementSD( d->m_image_measurement_sd );

  // Log version info
  int major, minor;
  CStereoInt::Version( major, minor );
  LOG_INFO( logger(), "SEAGIS StereoLibLX version: " << major << "." << minor );

  // Log units
  std::string units;
  if( d->m_stereo->GetUnits( d->m_camera_pair_id, units ) == OK )
  {
    LOG_INFO( logger(), "SEAGIS measurement units: " << units );
  }

  // Convert disparity parameters to range if specified
  if( d->m_epipolar_min_disparity > 0.0 && d->m_epipolar_max_disparity > 0.0 )
  {
    // Estimate focal_length * baseline product from the SEAGIS camera model
    // by projecting the image center via EpipolarLine at a known test range
    // and measuring the horizontal disparity
    int nRows = 0, nCols = 0;
    d->m_stereo->GetCameraFormat( d->m_camera_pair_id, LEFT, nRows, nCols );

    C2DPt center( nCols / 2.0, nRows / 2.0 );
    double test_range = 5000.0;
    int npts = 0;

    RESULT ep_res = d->m_stereo->EpipolarLine( d->m_camera_pair_id, LEFT,
      center, test_range * 0.99, test_range * 1.01, npts );

    if( ep_res == OK && npts > 0 )
    {
      C2DPt ep_pt;
      d->m_stereo->GetEpipolarPoint( d->m_camera_pair_id, npts / 2, ep_pt );

      double disparity = std::abs( center.X() - ep_pt.X() );

      if( disparity > 0.1 )
      {
        double fb = disparity * test_range;

        d->m_epipolar_min_depth = fb / d->m_epipolar_max_disparity;
        d->m_epipolar_max_depth = fb / d->m_epipolar_min_disparity;

        LOG_INFO( logger(), "Estimated focal*baseline product: " << fb
          << " (disparity " << disparity << " px at range " << test_range << ")" );
        LOG_INFO( logger(), "Converted disparity range [" << d->m_epipolar_min_disparity
          << ", " << d->m_epipolar_max_disparity << "] px to range ["
          << d->m_epipolar_min_depth << ", " << d->m_epipolar_max_depth << "]" );
      }
      else
      {
        LOG_WARN( logger(), "Could not estimate focal*baseline product: "
          "disparity too small (" << disparity << " px at range " << test_range
          << "). Falling back to epipolar_min_depth/epipolar_max_depth." );
      }
    }
    else
    {
      LOG_WARN( logger(), "Could not estimate focal*baseline product: "
        "EpipolarLine failed. Falling back to epipolar_min_depth/epipolar_max_depth." );
    }
  }
}

// ----------------------------------------------------------------------------
void
seagis_measurement_process
::_init()
{
  this->set_data_checking_level( check_valid );
}

// ----------------------------------------------------------------------------
void
seagis_measurement_process
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
seagis_measurement_process
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

  // Identify which detections are matched (same track ID in both cameras)
  std::vector< kv::track_id_t > common_ids;
  std::vector< kv::track_id_t > left_only_ids;
  std::vector< kv::track_id_t > right_only_ids;

  if( !d->m_assume_inputs_paired )
  {
    // Inputs come from independent trackers with potentially overlapping IDs.
    // Remap right-side IDs to avoid collisions, then treat everything as
    // unmatched so that detection_pairing_method can find the real matches.
    kv::track_id_t remap_offset = 1000000;
    for( const auto& kv : dets[0] )
    {
      remap_offset = std::max( remap_offset, kv.first + 1000000 );
    }

    std::map< kv::track_id_t, kv::detected_object_sptr > remapped_right;
    for( const auto& kv : dets[1] )
    {
      remapped_right[ kv.first + remap_offset ] = kv.second;
    }
    dets[1] = remapped_right;

    // All left detections are "left-only", all right detections are "right-only"
    for( const auto& kv : dets[0] )
    {
      left_only_ids.push_back( kv.first );
    }
    for( const auto& kv : dets[1] )
    {
      right_only_ids.push_back( kv.first );
    }
  }
  else
  {
    // Standard mode: match by track ID correspondence
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

    // Collect right-only IDs
    for( auto itr : dets[1] )
    {
      if( dets[0].find( itr.first ) == dets[0].end() )
      {
        right_only_ids.push_back( itr.first );
      }
    }
  }

  // Detection pairing: match left-only and right-only detections
  // Note: SeaGIS uses CStereoInt (not kwiver cameras), so projection-based methods
  // are not directly available. Instead we use raw bbox IOU or raw keypoint distance.
  if( !d->m_detection_pairing_method.empty() &&
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

    if( d->m_detection_pairing_method == "iou" )
    {
      // Use raw bbox IOU (no projection available in SeaGIS context)
      core::iou_matching_options opts;
      opts.iou_threshold = d->m_detection_pairing_threshold;

      paired = core::find_stereo_matches_iou( left_only_dets, right_only_dets, opts );
    }
    else if( d->m_detection_pairing_method == "keypoint_distance" )
    {
      // Use raw keypoint pixel distance (no projection)
      // Build cost matrix manually using the same pattern
      int n1 = static_cast< int >( left_only_dets.size() );
      int n2 = static_cast< int >( right_only_dets.size() );

      std::vector< std::vector< double > > cost_matrix(
        n1, std::vector< double >( n2, 1e10 ) );

      for( int i = 0; i < n1; ++i )
      {
        const auto& det1 = left_only_dets[i];
        const auto& kp1 = det1->keypoints();

        if( kp1.find( "head" ) == kp1.end() || kp1.find( "tail" ) == kp1.end() )
          continue;

        kv::vector_2d lh( kp1.at("head")[0], kp1.at("head")[1] );
        kv::vector_2d lt( kp1.at("tail")[0], kp1.at("tail")[1] );

        for( int j = 0; j < n2; ++j )
        {
          const auto& det2 = right_only_dets[j];
          const auto& kp2 = det2->keypoints();

          if( kp2.find( "head" ) == kp2.end() || kp2.find( "tail" ) == kp2.end() )
            continue;

          kv::vector_2d rh( kp2.at("head")[0], kp2.at("head")[1] );
          kv::vector_2d rt( kp2.at("tail")[0], kp2.at("tail")[1] );

          double avg_dist = ( ( lh - rh ).norm() + ( lt - rt ).norm() ) / 2.0;
          if( avg_dist <= d->m_detection_pairing_threshold )
          {
            cost_matrix[i][j] = avg_dist;
          }
        }
      }

      paired = core::greedy_assignment( cost_matrix, n1, n2 );
    }
    else
    {
      LOG_WARN( logger(), "Unknown detection_pairing_method: " +
                d->m_detection_pairing_method );
    }

    // Merge paired detections
    for( const auto& p : paired )
    {
      kv::track_id_t left_id = left_only_ids[p.first];
      kv::track_id_t right_id = right_only_ids[p.second];

      // Insert right detection under the left track ID
      dets[1][left_id] = dets[1][right_id];

      if( left_id != right_id )
      {
        dets[1].erase( right_id );
      }

      common_ids.push_back( left_id );

      LOG_INFO( logger(), "Paired left track " << left_id <<
                " with right track " << right_id <<
                " via " << d->m_detection_pairing_method );
    }
  }

  // Process matched detections
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

    // Check for head and tail keypoints
    bool left_has_kp = ( kp1.find( "head" ) != kp1.end() &&
                         kp1.find( "tail" ) != kp1.end() );
    bool right_has_kp = ( kp2.find( "head" ) != kp2.end() &&
                          kp2.find( "tail" ) != kp2.end() );

    if( !left_has_kp || !right_has_kp )
    {
      LOG_INFO( logger(), "Track ID " << id <<
                          " missing required keypoints (head/tail)" );
      continue;
    }

    kv::vector_2d left_head( kp1.at("head")[0], kp1.at("head")[1] );
    kv::vector_2d right_head( kp2.at("head")[0], kp2.at("head")[1] );
    kv::vector_2d left_tail( kp1.at("tail")[0], kp1.at("tail")[1] );
    kv::vector_2d right_tail( kp2.at("tail")[0], kp2.at("tail")[1] );

    const auto measurement = d->compute_measurement(
      left_head, right_head, left_tail, right_tail );

    if( !measurement.valid )
    {
      LOG_WARN( logger(), "Failed to compute measurement for track ID " << id );
      continue;
    }

    LOG_INFO( logger(), "Computed Length (SEAGIS): " << measurement.length );
    LOG_INFO( logger(), "  Midpoint (x,y,z): (" << measurement.x << ", "
              << measurement.y << ", " << measurement.z << ")" );
    LOG_INFO( logger(), "  Range: " << measurement.range << ", RMS: " << measurement.rms );

    det1->add_note( ":stereo_method=seagis" );
    det2->add_note( ":stereo_method=seagis" );

    core::add_measurement_attributes( det1, measurement );
    core::add_measurement_attributes( det2, measurement );
  }

#ifdef VIAME_ENABLE_OPENCV
  // Epipolar matching for detections where only one camera has keypoints
  if( d->m_enable_epipolar_matching && input_images.size() >= 2 )
  {
    // Convert input images to grayscale cv::Mat
    cv::Mat left_gray, right_gray;

    {
      cv::Mat left_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
        input_images[0]->get_image(),
        kwiver::arrows::ocv::image_container::BGR_COLOR );
      cv::Mat right_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
        input_images[1]->get_image(),
        kwiver::arrows::ocv::image_container::BGR_COLOR );

      if( left_cv.channels() == 3 )
        cv::cvtColor( left_cv, left_gray, cv::COLOR_BGR2GRAY );
      else if( left_cv.channels() == 4 )
        cv::cvtColor( left_cv, left_gray, cv::COLOR_BGRA2GRAY );
      else
        left_gray = left_cv;

      if( right_cv.channels() == 3 )
        cv::cvtColor( right_cv, right_gray, cv::COLOR_BGR2GRAY );
      else if( right_cv.channels() == 4 )
        cv::cvtColor( right_cv, right_gray, cv::COLOR_BGRA2GRAY );
      else
        right_gray = right_cv;
    }

    for( const kv::track_id_t& id : common_ids )
    {
      const auto& det1 = dets[0][id];
      const auto& det2 = dets[1][id];

      if( !det1 || !det2 )
      {
        continue;
      }

      // Skip if already measured (both cameras had keypoints)
      auto has_length = []( const kv::detected_object_sptr& d ) {
        for( const auto& n : d->notes() )
          if( n.find( ":length=" ) != std::string::npos ) return true;
        return false;
      };
      if( has_length( det1 ) )
      {
        continue;
      }

      const auto& kp1 = det1->keypoints();
      const auto& kp2 = det2->keypoints();

      bool left_has_kp = ( kp1.find( "head" ) != kp1.end() &&
                           kp1.find( "tail" ) != kp1.end() );
      bool right_has_kp = ( kp2.find( "head" ) != kp2.end() &&
                            kp2.find( "tail" ) != kp2.end() );

      // Only process if exactly one camera has keypoints
      if( left_has_kp == right_has_kp )
      {
        continue;
      }

      kv::vector_2d left_head, left_tail, right_head, right_tail;
      bool head_found = false, tail_found = false;

      if( left_has_kp && !right_has_kp )
      {
        // Left has keypoints, find corresponding points in right
        left_head = kv::vector_2d( kp1.at("head")[0], kp1.at("head")[1] );
        left_tail = kv::vector_2d( kp1.at("tail")[0], kp1.at("tail")[1] );

        head_found = d->find_corresponding_point_epipolar(
          LEFT, left_gray, right_gray, left_head, right_head );
        tail_found = d->find_corresponding_point_epipolar(
          LEFT, left_gray, right_gray, left_tail, right_tail );
      }
      else if( right_has_kp && !left_has_kp )
      {
        // Right has keypoints, find corresponding points in left
        right_head = kv::vector_2d( kp2.at("head")[0], kp2.at("head")[1] );
        right_tail = kv::vector_2d( kp2.at("tail")[0], kp2.at("tail")[1] );

        head_found = d->find_corresponding_point_epipolar(
          RIGHT, right_gray, left_gray, right_head, left_head );
        tail_found = d->find_corresponding_point_epipolar(
          RIGHT, right_gray, left_gray, right_tail, left_tail );
      }

      if( head_found && tail_found )
      {
        const auto measurement = d->compute_measurement(
          left_head, right_head, left_tail, right_tail );

        if( measurement.valid )
        {
          LOG_INFO( logger(), "Computed Length (SEAGIS epipolar): " << measurement.length );
          LOG_INFO( logger(), "  Midpoint (x,y,z): (" << measurement.x << ", "
                    << measurement.y << ", " << measurement.z << ")" );
          LOG_INFO( logger(), "  Range: " << measurement.range
                    << ", RMS: " << measurement.rms );

          det1->add_note( ":stereo_method=seagis_epipolar" );
          det2->add_note( ":stereo_method=seagis_epipolar" );

          core::add_measurement_attributes( det1, measurement );
          core::add_measurement_attributes( det2, measurement );
        }
      }
    }
  }
#endif

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

} // end namespace seagis

} // end namespace viame
