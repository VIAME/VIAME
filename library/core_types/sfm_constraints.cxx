// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for kwiver::vital::sfm_constraints class storing
///       constraints to be used in SfM.

#include <viame/core_types/sfm_constraints.h>

#include <viame/core_types/math_constants.h>
#include <viame/core_types/geodesy.h>
#include <viame/core_types/rotation.h>

namespace kwiver {

namespace vital {

/// Private implementation class
class sfm_constraints::priv
{
public:
  /// Constructor
  priv();

  /// Destructor
  ~priv();

  struct im_data
  {
    int width = -1;
    int height = -1;

    im_data() = default;

    im_data( int w_, int h_ )
      : width( w_ ),
        height( h_ )
    {}

    im_data( im_data const& other ) = default;
    im_data& operator=( im_data const& other ) = default;
  };

  metadata_map_sptr m_md;
  local_tangent_space m_local_space;
  std::map< frame_id_t, im_data > m_image_data;
};

sfm_constraints::priv
::priv()
{}

sfm_constraints::priv
::~priv()
{}

sfm_constraints
::sfm_constraints( const sfm_constraints& other )
  : m_priv( new priv )
{
  m_priv->m_local_space = other.m_priv->m_local_space;
  m_priv->m_md = other.m_priv->m_md;
  m_priv->m_image_data = other.m_priv->m_image_data;
}

sfm_constraints
::sfm_constraints()
  : m_priv( new priv )
{}

sfm_constraints
::sfm_constraints(
  metadata_map_sptr md,
  local_tangent_space const& local_space )
  : m_priv( new priv )
{
  m_priv->m_md = md;
  m_priv->m_local_space = local_space;
}

sfm_constraints
::~sfm_constraints()
{}

metadata_map_sptr
sfm_constraints
::get_metadata()
{
  return m_priv->m_md;
}

void
sfm_constraints
::set_metadata( metadata_map_sptr md )
{
  m_priv->m_md = md;
}

local_tangent_space
sfm_constraints
::get_local_space()
{
  return m_priv->m_local_space;
}

void
sfm_constraints
::set_local_space( local_tangent_space const& local_space )
{
  m_priv->m_local_space = local_space;
}

bool
sfm_constraints
::get_focal_length_prior( frame_id_t fid, float& focal_length ) const
{
  if( !m_priv->m_md )
  {
    return false;
  }

  auto& md = *m_priv->m_md;

  std::set< frame_id_t > frame_ids_to_try;

  int image_width = -1;
  if( !get_image_width( fid, image_width ) )
  {
    return false;
  }

  if( fid >= 0 )
  {
    frame_ids_to_try.insert( fid );
  }
  else
  {
    frame_ids_to_try = md.frames();
  }

  std::vector< double > focal_lengths;
  for( auto test_fid : frame_ids_to_try )
  {
    if( md.has< VITAL_META_SENSOR_HORIZONTAL_FOV >( test_fid ) )
    {
      double hfov = md.get< VITAL_META_SENSOR_HORIZONTAL_FOV >( test_fid );
      focal_lengths.push_back(
        static_cast< float >(
          ( image_width * 0.5 ) / tan( 0.5 * hfov * deg_to_rad ) ) );
      continue;
    }

    if( md.has< VITAL_META_TARGET_WIDTH >( test_fid ) &&
        md.has< VITAL_META_SLANT_RANGE >( test_fid ) )
    {
      focal_length = static_cast< float >( image_width *
                                           md.get< VITAL_META_SLANT_RANGE >(
                                             test_fid ) /
                                           md.get< VITAL_META_TARGET_WIDTH >(
                                             test_fid ) );
      focal_lengths.push_back( focal_length );
      continue;
    }
  }
  if( focal_lengths.empty() )
  {
    return false;
  }
  // compute the median focal length
  std::nth_element(
    focal_lengths.begin(),
    focal_lengths.begin() + focal_lengths.size() / 2,
    focal_lengths.end() );
  focal_length = focal_lengths[ focal_lengths.size() / 2 ];

  return true;
}

bool
sfm_constraints
::get_camera_orientation_prior_local(
  frame_id_t fid,
  rotation_d& R_loc ) const
{
  if( !m_priv->m_local_space.valid() )
  {
    return false;
  }

  if( !m_priv->m_md )
  {
    return false;
  }

  auto& md = *m_priv->m_md;

  auto const& location_item =
    md.get_item( VITAL_META_SENSOR_LOCATION, fid );

  if( !location_item )
  {
    return false;
  }

  auto const location = location_item.get< geo_point >();

  auto const orientation_item =
    md.get_item( VITAL_META_SENSOR_ORIENTATION, fid );
  if( !orientation_item )
  {
    return false;
  }

  auto const crs = SRID::lat_lon_WGS84;
  R_loc =
    sensor_to_camera(
      m_priv->m_local_space.to_local(
        ned_to_enu( orientation_item.get< rotation_d >() ),
        geo_point{ location.location( crs ), crs } ) );

  return true;
}

bool
sfm_constraints
::get_camera_position_prior_local(
  frame_id_t fid,
  vector_3d& pos_loc ) const
{
  if( !m_priv->m_local_space.valid() )
  {
    return false;
  }

  if( !m_priv->m_md )
  {
    return false;
  }

  kwiver::vital::geo_point gloc;
  if( auto const& item =
        m_priv->m_md->get_item( VITAL_META_SENSOR_LOCATION, fid ) )
  {
    pos_loc = m_priv->m_local_space.to_local( item.get< vital::geo_point >() );
    return true;
  }
  else
  {
    return false;
  }
}

sfm_constraints::position_map
sfm_constraints
::get_camera_position_priors() const
{
  position_map local_positions;

  if( !m_priv->m_md )
  {
    return local_positions;
  }

  for( auto mdv : m_priv->m_md->metadata() )
  {
    auto fid = mdv.first;

    vector_3d loc;
    if( !get_camera_position_prior_local( fid, loc ) )
    {
      continue;
    }
    if( local_positions.empty() )
    {
      local_positions[ fid ] = loc;
    }
    else
    {
      auto last_loc = local_positions.crbegin()->second;
      if( loc == last_loc )
      {
        continue;
      }
      local_positions[ fid ] = loc;
    }
  }
  return local_positions;
}

void
sfm_constraints
::store_image_size( frame_id_t fid, int image_width, int image_height )
{
  m_priv->m_image_data[ fid ] = priv::im_data( image_width, image_height );
}

bool
sfm_constraints
::get_image_height( frame_id_t fid, int& image_height ) const
{
  if( fid >= 0 )
  {
    auto data_it = m_priv->m_image_data.find( fid );
    if( data_it == m_priv->m_image_data.end() )
    {
      return false;
    }
    image_height = data_it->second.height;
    return true;
  }
  else
  {
    if( m_priv->m_image_data.empty() )
    {
      return false;
    }
    image_height = m_priv->m_image_data.begin()->second.height;
    return true;
  }
}

bool
sfm_constraints
::get_image_width( frame_id_t fid, int& image_width ) const
{
  if( fid >= 0 )
  {
    auto data_it = m_priv->m_image_data.find( fid );
    if( data_it == m_priv->m_image_data.end() )
    {
      return false;
    }
    image_width = data_it->second.width;
    return true;
  }
  else
  {
    if( m_priv->m_image_data.empty() )
    {
      return false;
    }
    image_width = m_priv->m_image_data.begin()->second.width;
    return true;
  }
}

} // namespace vital

} // namespace kwiver
