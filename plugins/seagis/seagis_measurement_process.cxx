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

#include <LX_StereoInterface.h>

#include <string>
#include <vector>
#include <map>
#include <set>

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

create_config_trait( image_measurement_sd, double, 1.0,
  "Image measurement standard deviation in pixels" );

create_config_trait( camera_pair_id, unsigned int, 0,
  "Camera pair ID for SEAGIS library (typically 0)" );

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

  // Other variables
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  seagis_measurement_process* parent;

  // Helper functions
  double compute_length(
    const kv::vector_2d& left_head,
    const kv::vector_2d& right_head,
    const kv::vector_2d& left_tail,
    const kv::vector_2d& right_tail );
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
  , m_frame_counter( 0 )
  , parent( ptr )
{
}


seagis_measurement_process::priv
::~priv()
{
}


// -----------------------------------------------------------------------------
double
seagis_measurement_process::priv
::compute_length(
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail )
{
  // Intersect head points
  C2DPt ptLeftHead( left_head.x(), left_head.y() );
  C2DPt ptRightHead( right_head.x(), right_head.y() );
  C3DPt pt3DHead, pt3DHeadSD;
  double dRMSHead;

  RESULT res = m_stereo->Intersect( m_camera_pair_id, ptLeftHead, ptRightHead,
                                    pt3DHead, dRMSHead, pt3DHeadSD );
  if( res != OK )
  {
    return -1.0;
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
    return -1.0;
  }

  // Compute distance
  double dSD;
  double length = CStereoInt::Distance( pt3DHead, pt3DHeadSD, pt3DTail, pt3DTailSD, dSD );

  return length;
}


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

  for( auto itr : dets[0] )
  {
    for( unsigned i = 1; i < input_tracks.size(); ++i )
    {
      if( dets[i].find( itr.first ) != dets[i].end() )
      {
        common_ids.push_back( itr.first );
        break;
      }
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

    const double length = d->compute_length(
      left_head, right_head, left_tail, right_tail );

    if( length < 0 )
    {
      LOG_WARN( logger(), "Failed to compute length for track ID " << id );
      continue;
    }

    LOG_INFO( logger(), "Computed Length (SEAGIS): " << length );

    det1->set_length( length );
    det2->set_length( length );

    det1->add_note( ":stereo_method=seagis" );
    det2->add_note( ":stereo_method=seagis" );
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

} // end namespace seagis

} // end namespace viame
