// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/local_tangent_space.h>

#include <viame/core_types/geodesy.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <viame/core_types/math_constants.h>

#include <cfloat>

namespace kwiver {

namespace vital {

namespace {

// ----------------------------------------------------------------------------
void
assert_valid( local_tangent_space const& space )
{
  if( !space.valid() )
  {
    throw std::runtime_error( "Invalid local tangent space" );
  }
}

// ----------------------------------------------------------------------------
matrix_3x3d
axes_at_point( vital::geo_point const& point )
{
  matrix_3x3d axes;

  if( point.is_empty() )
  {
    axes << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    return axes;
  }

  auto const ecef = point.location( SRID::ECEF_WGS84 );

  // Check for being on the polar axis and return standardized coordinates
  constexpr double epsilon = 1e-6;
  if( std::abs( ecef[ 0 ] ) < epsilon && std::abs( ecef[ 1 ] ) < epsilon )
  {
    auto const sign = ecef[ 2 ] < 0.0 ? -1.0 : 1.0;
    axes <<
      1.0, 0.0, 0.0,
      0.0, sign, 0.0,
      0.0, 0.0, sign;
    return axes;
  }

  auto const lon_lat = point.location( SRID::lat_lon_WGS84 );

  // Compute tangents using trigonometry
  auto const sin_lon = std::sin( lon_lat[ 0 ] * deg_to_rad );
  auto const cos_lon = std::cos( lon_lat[ 0 ] * deg_to_rad );
  auto const sin_lat = std::sin( lon_lat[ 1 ] * deg_to_rad );
  auto const cos_lat = std::cos( lon_lat[ 1 ] * deg_to_rad );

  axes <<
    -sin_lon, cos_lon, 0.0, // East
    -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat, // North
    cos_lon* cos_lat, sin_lon* cos_lat, sin_lat;   // Up
  return axes;
}

} // namespace <anonymous>

// ----------------------------------------------------------------------------
local_tangent_space
::local_tangent_space()
  : m_origin{},
    m_axes{ axes_at_point( m_origin )  }
{}

// ----------------------------------------------------------------------------
local_tangent_space
::local_tangent_space( geo_point const& origin )
  : m_origin{ origin },
    m_axes{ axes_at_point( origin ) }
{}

// ----------------------------------------------------------------------------
bool
local_tangent_space
::valid() const
{
  return !m_origin.is_empty();
}

// ----------------------------------------------------------------------------
geo_point const&
local_tangent_space
::origin() const
{
  return m_origin;
}

// ----------------------------------------------------------------------------
vector_3d
local_tangent_space
::to_local( geo_point const& global_point ) const
{
  assert_valid( *this );

  vector_3d point = global_point.location( SRID::ECEF_WGS84 );
  point -= m_origin.location( SRID::ECEF_WGS84 );
  return m_axes * point;
}

// ----------------------------------------------------------------------------
rotation_d
local_tangent_space
::to_local(
  rotation_d const& global_rotation, geo_point const& global_point ) const
{
  assert_valid( *this );

  auto rotation = global_rotation;
  switch( global_point.crs() )
  {
    case SRID::lat_lon_WGS84:
      rotation =
        rotation *
        vital::rotation_d{ axes_at_point( global_point ) } *
      vital::rotation_d{ matrix_3x3d{ m_axes.transpose() } };
      break;
    case SRID::ECEF_WGS84:
      rotation =
        rotation *
        vital::rotation_d{ matrix_3x3d{ m_axes.transpose() } };
      break;
    default:
      throw std::runtime_error( "Unsupported CRS" );
  }
  return rotation;
}

// ----------------------------------------------------------------------------
geo_point
local_tangent_space
::to_global( vector_3d const& local_point ) const
{
  assert_valid( *this );

  vector_3d point = m_axes.transpose() * local_point;
  point += m_origin.location( SRID::ECEF_WGS84 );
  return { point, SRID::ECEF_WGS84 };
}

// ----------------------------------------------------------------------------
rotation_d
local_tangent_space
::to_global(
  rotation_d const& local_rotation, geo_point const& global_point ) const
{
  assert_valid( *this );

  auto rotation = local_rotation;
  switch( global_point.crs() )
  {
    case SRID::lat_lon_WGS84:
      rotation =
        rotation *
        vital::rotation_d{ m_axes } *
      vital::rotation_d{ matrix_3x3d{
                           axes_at_point( global_point ).transpose() } };
      break;
    case SRID::ECEF_WGS84:
      rotation =
        rotation *
        vital::rotation_d{ m_axes };
      break;
    default:
      throw std::runtime_error( "Unsupported CRS" );
  }
  return rotation;
}

// ----------------------------------------------------------------------------
local_tangent_space
read_local_tangent_space_from_file( std::string const& filepath )
{
  std::ifstream ifs( filepath );
  if( !ifs )
  {
    throw std::runtime_error(
      "Failed to open local tangent space file for reading" );
  }

  double lat = 0, lon = 0, alt = 0;
  ifs >> lat >> lon >> alt;
  if( !ifs )
  {
    throw std::runtime_error( "Failed to parse local tangent space file" );
  }

  return local_tangent_space(
    geo_point(
      vector_3d( lon, lat, alt ),
      SRID::lat_lon_WGS84 ) );
}

// ----------------------------------------------------------------------------
void
write_local_tangent_space_to_file(
  local_tangent_space const& local_space, std::string const& filepath )
{
  assert_valid( local_space );

  auto lon_lat_alt = local_space.origin().location( SRID::lat_lon_WGS84 );
  std::ofstream ofs( filepath );
  if( !ofs )
  {
    throw std::runtime_error(
      "Failed to open local tangent space file for writing" );
  }

  ofs
    << std::setprecision( DBL_DIG + 1 )
    << lon_lat_alt[ 1 ] << " "
    << lon_lat_alt[ 0 ] << " "
    << lon_lat_alt[ 2 ] << std::endl;

  if( !ofs )
  {
    throw std::runtime_error( "Failed to write to local tangent space file" );
  }
}

} // namespace vital

} // namespace kwiver
