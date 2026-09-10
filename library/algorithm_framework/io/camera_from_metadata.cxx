// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Function to generate \ref kwiver::vital::camera_rpc from metadata

#include "camera_from_metadata.h"

#include <viame/core_types/math_constants.h>
#include <viame/core_types/geodesy.h>
#include <viame/core_types/metadata_traits.h>
#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

/// Extract scale or offset metadata to a vector
vector_d
tags_to_vector(
  metadata_sptr const& md,
  std::vector< vital_metadata_tag > tags )
{
  auto vec_length = tags.size();

  vector_d rslt( vec_length );

  for( size_t i = 0; i < vec_length; ++i )
  {
    if( auto& mdi = md->find( tags[ i ] ) )
    {
      rslt[ i ] = mdi.as_double();
    }
    else
    {
      VITAL_THROW(
        metadata_exception, "Missing RPC metadata: " +
        tag_traits_by_tag( tags[ i ] ).name() );
    }
  }

  return rslt;
}

/// Extract coefficient metadata to a matrix
rpc_matrix
tags_to_matrix(
  metadata_sptr const& md,
  std::vector< vital_metadata_tag > tags )
{
  if( tags.size() != 4 )
  {
    VITAL_THROW(
      metadata_exception,
      "Should have 4 metadata tags for RPC coefficients" );
  }

  rpc_matrix rslt;

  for( int i = 0; i < 4; ++i )
  {
    if( auto& mdi = md->find( tags[ i ] ) )
    {
      // `row()` was an assignable block in Eigen and is a value here.
      rslt.set_row( i, vector_< 20, double >::from_dynamic(
                         string_to_vector( mdi.as_string() ) ) );
    }
    else
    {
      VITAL_THROW(
        metadata_exception, "Missing RPC metadata: " +
        tag_traits_by_tag( tags[ i ] ).name() );
    }
  }

  return rslt;
}

/// Convert space separated strings to Eigen vector
vector_d
VITAL_EXPORT
string_to_vector( std::string const& s )
{
  std::vector< std::string > tokens;
  std::string token;
  std::istringstream tokenStream( s );
  while( std::getline( tokenStream, token, ' ' ) )
  {
    tokens.push_back( token );
  }

  vector_d result( tokens.size() );
  for( size_t i = 0; i < tokens.size(); ++i )
  {
    result[ i ] = std::stod( tokens[ i ] );
  }

  return result;
}

/// Produce RPC camera from metadata
camera_sptr
VITAL_EXPORT
camera_from_metadata( metadata_sptr const& md )
{
  vector_3d world_scale, world_offset;
  vector_2d image_scale, image_offset;
  rpc_matrix rpc_coeffs;

  std::vector< vital_metadata_tag > world_scale_tags = {
    VITAL_META_RPC_LONG_SCALE,
    VITAL_META_RPC_LAT_SCALE,
    VITAL_META_RPC_HEIGHT_SCALE };
  world_scale = vector_3d::from_dynamic( tags_to_vector( md, world_scale_tags ) );

  std::vector< vital_metadata_tag > world_offset_tags = {
    VITAL_META_RPC_LONG_OFFSET,
    VITAL_META_RPC_LAT_OFFSET,
    VITAL_META_RPC_HEIGHT_OFFSET };
  world_offset = vector_3d::from_dynamic( tags_to_vector( md, world_offset_tags ) );

  std::vector< vital_metadata_tag > image_scale_tags = {
    VITAL_META_RPC_ROW_SCALE,
    VITAL_META_RPC_COL_SCALE };
  image_scale = vector_2d::from_dynamic( tags_to_vector( md, image_scale_tags ) );

  std::vector< vital_metadata_tag > image_offset_tags = {
    VITAL_META_RPC_ROW_OFFSET,
    VITAL_META_RPC_COL_OFFSET };
  image_offset = vector_2d::from_dynamic( tags_to_vector( md, image_offset_tags ) );

  std::vector< vital_metadata_tag > rpc_coeffs_tags = {
    VITAL_META_RPC_ROW_NUM_COEFF,
    VITAL_META_RPC_ROW_DEN_COEFF,
    VITAL_META_RPC_COL_NUM_COEFF,
    VITAL_META_RPC_COL_DEN_COEFF };
  rpc_coeffs = tags_to_matrix( md, rpc_coeffs_tags );

  return std::make_shared< simple_camera_rpc >(
    world_scale, world_offset,
    image_scale, image_offset,
    rpc_coeffs );
}

/// Use metadata to construct intrinsics
VITAL_EXPORT
camera_intrinsics_sptr
intrinsics_from_metadata(
  metadata const& md,
  size_t image_width,
  size_t image_height )
{
  double im_w = static_cast< double >( image_width );
  double im_h = static_cast< double >( image_height );
  double focal_len = 0;

  auto& md_slant_range =
    md.find( VITAL_META_SLANT_RANGE );
  auto& md_target_width =
    md.find( VITAL_META_TARGET_WIDTH );
  if( md_slant_range && md_target_width )
  {
    focal_len =
      im_w * ( md_slant_range.as_double() / md_target_width.as_double() );
  }
  else
  {
    auto& md_hfov =
      md.find( VITAL_META_SENSOR_HORIZONTAL_FOV );
    if( md_hfov )
    {
      focal_len =
        ( im_w / 2.0 ) / tan( 0.5 * md_hfov.as_double() * deg_to_rad );
    }
    else
    {
      return nullptr;
    }
  }

  vector_2d pp( 0.5 * im_w, 0.5 * im_h );
  return std::make_shared< simple_camera_intrinsics >(
    focal_len, pp, 1.0, 0.0,
    vector_d(), image_width, image_height );
}

/// Use a sequence of metadata objects to initialize a sequence of cameras
std::map< frame_id_t, camera_sptr >
initialize_cameras_with_metadata(
  std::map< frame_id_t, metadata_sptr > const& md_map,
  simple_camera_perspective const& base_camera,
  local_tangent_space& local_space,
  bool init_intrinsics,
  rotation_d const& rot_offset )
{
  std::map< frame_id_t, camera_sptr > cam_map;
  vector_3d mean( 0, 0, 0 );
  simple_camera_perspective active_cam( base_camera );

  bool update_local_origin = false;
  if( !local_space.valid() && !md_map.empty() )
  {
    // if a local coordinate system has not been established,
    // use the coordinates of the first camera
    for( auto m : md_map )
    {
      if( !m.second )
      {
        continue;
      }
      if( auto& mdi = m.second->find( VITAL_META_SENSOR_LOCATION ) )
      {
        auto gloc = mdi.get< geo_point >();

        // set the origin to the ground
        auto const crs = vital::SRID::lat_lon_WGS84;
        auto loc = gloc.location( crs );
        loc[ 2 ] = 0.0;
        gloc.set_location( loc, crs );

        local_space = local_tangent_space( gloc );
        update_local_origin = true;
        break;
      }
    }
  }

  if( !local_space.valid() )
  {
    return cam_map;
  }

  for( auto const& p : md_map )
  {
    auto md = p.second;
    if( !md )
    {
      continue;
    }
    if( init_intrinsics )
    {
      auto K = base_camera.get_intrinsics();
      K = intrinsics_from_metadata( *md, K->image_width(), K->image_height() );
      if( K )
      {
        active_cam.set_intrinsics( K );
      }
    }
    if( update_camera_from_metadata(
      *md, local_space, active_cam,
      rot_offset ) )
    {
      mean += active_cam.center();
      cam_map[ p.first ] =
        std::make_shared< simple_camera_perspective >( active_cam );
    }
  }

  if( update_local_origin )
  {
    mean /= static_cast< double >( cam_map.size() );
    // only use the mean easting and northing
    mean[ 2 ] = 0.0;

    // shift the origin to the mean of the cameras
    local_tangent_space new_local_space( local_space.to_global( mean ) );

    // shift all cameras to the new coordinate system.
    typedef std::map< frame_id_t, camera_sptr >::value_type cam_map_val_t;
    for( cam_map_val_t const& p : cam_map )
    {
      simple_camera_perspective* cam =
        dynamic_cast< simple_camera_perspective* >( p.second.get() );
      auto const gloc = local_space.to_global( cam->get_center() );
      auto const loc = new_local_space.to_local( gloc );
      cam->set_center( loc );
      cam->set_rotation(
        new_local_space.to_local(
          local_space.to_global(
            cam->get_rotation(), gloc ), gloc ) );
    }
    local_space = std::move( new_local_space );
  }

  return cam_map;
}

/// Use the pose data provided by metadata to update camera pose
bool
update_camera_from_metadata(
  metadata const& md,
  local_tangent_space const& local_space,
  simple_camera_perspective& cam,
  [[maybe_unused]] rotation_d const& rot_offset )
{
  if( auto& mdi = md.find( VITAL_META_SENSOR_LOCATION ) )
  {
    auto const gloc = mdi.get< geo_point >();
    auto const loc = local_space.to_local( gloc );
    cam.set_center( loc );

    if( auto const mdi2 = md.find( VITAL_META_SENSOR_ORIENTATION ) )
    {
      auto const crs = SRID::lat_lon_WGS84;
      auto const rotation = ned_to_enu( mdi2.get< rotation_d >() );
      cam.set_rotation(
        sensor_to_camera(
          local_space.to_local( rotation, { gloc.location( crs ), crs } ) ) );
    }

    return true;
  }

  return false;
}

/// Update a sequence of metadata from a sequence of cameras and
/// local_tangent_space
void
update_metadata_from_cameras(
  std::map< frame_id_t, camera_sptr > const& cam_map,
  local_tangent_space const& local_space,
  std::map< frame_id_t, metadata_sptr >& md_map )
{
  if( local_space.origin().is_empty() )
  {
    // TODO throw an exception here?
    logger_handle_t
      logger( get_logger( "update_metadata_from_cameras" ) );
    LOG_WARN(logger, "local geo coordinates do not have an origin" );
    return;
  }

  typedef std::map< frame_id_t, camera_sptr >::value_type cam_map_val_t;
  for( cam_map_val_t const& p : cam_map )
  {
    auto active_md = md_map[ p.first ];
    if( !active_md )
    {
      md_map[ p.first ] = active_md = std::make_shared< metadata >();
    }

    auto cam = dynamic_cast< simple_camera_perspective* >( p.second.get() );
    if( active_md && cam )
    {
      update_metadata_from_camera( *cam, local_space, *active_md );
    }
  }
}

/// Use the camera pose to update the metadata structure
void
update_metadata_from_camera(
  simple_camera_perspective const& cam,
  local_tangent_space const& local_space,
  metadata& md )
{
  auto const location = local_space.to_global( cam.center() );
  md.add< VITAL_META_SENSOR_LOCATION >( location );

  auto const crs = SRID::lat_lon_WGS84;
  auto orientation =
    enu_to_ned(
      local_space.to_global(
        camera_to_sensor( cam.rotation() ),
        { location.location( crs ), crs } ) );
  md.add< VITAL_META_SENSOR_ORIENTATION >( orientation );
}

} // namespace vital

}   // end of namespace
