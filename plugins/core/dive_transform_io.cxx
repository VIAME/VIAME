/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "dive_transform_io.h"

#include <vital/types/homography.h>

// Pulled in for the vendored rapidjson headers and to route rapidjson
// assertions to exceptions rather than aborts
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/external/rapidjson/ostreamwrapper.h>
#include <vital/internal/cereal/external/rapidjson/prettywriter.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace viame
{

namespace
{

const std::string registration_file_type = "dive-camera-registration";

// One "pairs" entry: camera names plus whichever fitted directions the
// producer wrote (either may be absent or null)
struct registration_pair
{
  std::string left;
  std::string right;
  bool has_left_to_right = false;
  bool has_right_to_left = false;
  Eigen::Matrix< double, 3, 3 > left_to_right;
  Eigen::Matrix< double, 3, 3 > right_to_left;
};

// Read a row-major 3x3 matrix stored as [ [ a, b, c ], ... ]; false when the
// value is null or otherwise not a numeric 3x3
bool read_matrix3( rapidjson::Value const& value,
                   Eigen::Matrix< double, 3, 3 >& output )
{
  if( !value.IsArray() || value.Size() != 3 )
  {
    return false;
  }

  for( unsigned r = 0; r < 3; ++r )
  {
    rapidjson::Value const& row = value[ r ];

    if( !row.IsArray() || row.Size() != 3 )
    {
      return false;
    }

    for( unsigned c = 0; c < 3; ++c )
    {
      if( !row[ c ].IsNumber() )
      {
        return false;
      }

      output( r, c ) = row[ c ].GetDouble();
    }
  }

  return true;
}

Eigen::Matrix< double, 3, 3 >
invert_homography( Eigen::Matrix< double, 3, 3 > const& matrix,
                   std::string const& filename )
{
  if( std::abs( matrix.determinant() ) <= 1e-12 )
  {
    throw std::runtime_error( "Non-invertible homography in: " + filename );
  }

  return matrix.inverse();
}

std::vector< registration_pair >
parse_registration_file( std::string const& filename )
{
  std::ifstream input( filename );

  if( !input )
  {
    throw std::runtime_error( "Unable to open: " + filename );
  }

  std::stringstream buffer;
  buffer << input.rdbuf();

  rapidjson::Document document;

  if( document.Parse( buffer.str().c_str() ).HasParseError() )
  {
    throw std::runtime_error( "Unable to parse JSON in: " + filename );
  }

  if( !document.IsObject() )
  {
    throw std::runtime_error( "Not a DIVE camera registration: " + filename );
  }

  // Self-identified files must match; files without a type field are
  // accepted when they carry a pairs array (matching DIVE's own loader)
  if( document.HasMember( "type" ) && document[ "type" ].IsString() &&
      document[ "type" ].GetString() != registration_file_type )
  {
    throw std::runtime_error( "Not a DIVE camera registration: " + filename );
  }

  if( !document.HasMember( "pairs" ) || !document[ "pairs" ].IsArray() )
  {
    throw std::runtime_error( "No camera pairs in: " + filename );
  }

  std::vector< registration_pair > pairs;

  for( auto const& entry : document[ "pairs" ].GetArray() )
  {
    if( !entry.IsObject() ||
        !entry.HasMember( "left" ) || !entry[ "left" ].IsString() ||
        !entry.HasMember( "right" ) || !entry[ "right" ].IsString() )
    {
      continue;
    }

    registration_pair pair;
    pair.left = entry[ "left" ].GetString();
    pair.right = entry[ "right" ].GetString();

    if( entry.HasMember( "leftToRight" ) )
    {
      pair.has_left_to_right =
        read_matrix3( entry[ "leftToRight" ], pair.left_to_right );
    }
    if( entry.HasMember( "rightToLeft" ) )
    {
      pair.has_right_to_left =
        read_matrix3( entry[ "rightToLeft" ], pair.right_to_left );
    }

    if( pair.has_left_to_right || pair.has_right_to_left )
    {
      pairs.push_back( pair );
    }
  }

  if( pairs.empty() )
  {
    throw std::runtime_error( "No fitted camera pairs in: " + filename );
  }

  return pairs;
}

// Resolve the selected pair to a from->to matrix, inverting the stored
// direction when only the opposite one was fitted
Eigen::Matrix< double, 3, 3 >
forward_matrix( registration_pair const& pair, std::string const& filename )
{
  if( pair.has_left_to_right )
  {
    return pair.left_to_right;
  }

  return invert_homography( pair.right_to_left, filename );
}

Eigen::Matrix< double, 3, 3 >
reverse_matrix( registration_pair const& pair, std::string const& filename )
{
  if( pair.has_right_to_left )
  {
    return pair.right_to_left;
  }

  return invert_homography( pair.left_to_right, filename );
}

} // end anonymous namespace


dive_transform_io
::dive_transform_io()
{
}

dive_transform_io
::~dive_transform_io()
{
}


kwiver::vital::config_block_sptr
dive_transform_io
::get_configuration() const
{
  auto config = kwiver::vital::algo::transform_2d_io::get_configuration();

  config->set_value( "from_camera", m_from_camera,
    "Camera whose image coordinates the transform maps from. Leave empty "
    "with to_camera to use the single pair in the file, left to right." );
  config->set_value( "to_camera", m_to_camera,
    "Camera whose image coordinates the transform maps into. Leave empty "
    "with from_camera to use the single pair in the file, left to right." );

  return config;
}

void
dive_transform_io
::set_configuration( kwiver::vital::config_block_sptr config )
{
  m_from_camera = config->get_value< std::string >(
    "from_camera", m_from_camera );
  m_to_camera = config->get_value< std::string >(
    "to_camera", m_to_camera );
}

bool
dive_transform_io
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  // Camera selection is meaningless unless both endpoints are named
  return config->get_value< std::string >( "from_camera", "" ).empty() ==
         config->get_value< std::string >( "to_camera", "" ).empty();
}

kwiver::vital::transform_2d_sptr
dive_transform_io
::load_( std::string const& filename ) const
{
  auto const pairs = parse_registration_file( filename );

  Eigen::Matrix< double, 3, 3 > matrix;

  if( m_from_camera.empty() && m_to_camera.empty() )
  {
    if( pairs.size() != 1 )
    {
      throw std::runtime_error( "Multiple camera pairs in " + filename +
        "; set from_camera and to_camera to select one" );
    }

    matrix = forward_matrix( pairs[ 0 ], filename );
  }
  else
  {
    bool found = false;

    for( auto const& pair : pairs )
    {
      if( pair.left == m_from_camera && pair.right == m_to_camera )
      {
        matrix = forward_matrix( pair, filename );
        found = true;
        break;
      }
      if( pair.left == m_to_camera && pair.right == m_from_camera )
      {
        matrix = reverse_matrix( pair, filename );
        found = true;
        break;
      }
    }

    if( !found )
    {
      throw std::runtime_error( "No pair registering " + m_from_camera +
        " onto " + m_to_camera + " in: " + filename );
    }
  }

  return std::make_shared< kwiver::vital::homography_< double > >( matrix );
}

void
dive_transform_io
::save_( std::string const& filename,
         kwiver::vital::transform_2d_sptr data ) const
{
  auto homog =
    std::dynamic_pointer_cast< kwiver::vital::homography >( data );

  if( !homog )
  {
    throw std::runtime_error(
      "Only homography transforms can be saved in DIVE format" );
  }

  Eigen::Matrix< double, 3, 3 > const matrix = homog->matrix();

  std::string const left =
    m_from_camera.empty() ? "left" : m_from_camera;
  std::string const right =
    m_to_camera.empty() ? "right" : m_to_camera;

  rapidjson::Document document;
  document.SetObject();
  auto& alloc = document.GetAllocator();

  document.AddMember( "type",
    rapidjson::Value( registration_file_type.c_str(), alloc ), alloc );
  document.AddMember( "version", 1, alloc );

  auto make_matrix = [&alloc]( Eigen::Matrix< double, 3, 3 > const& m )
  {
    rapidjson::Value rows( rapidjson::kArrayType );
    for( unsigned r = 0; r < 3; ++r )
    {
      rapidjson::Value row( rapidjson::kArrayType );
      for( unsigned c = 0; c < 3; ++c )
      {
        row.PushBack( m( r, c ), alloc );
      }
      rows.PushBack( row, alloc );
    }
    return rows;
  };

  rapidjson::Value pair( rapidjson::kObjectType );
  pair.AddMember( "left", rapidjson::Value( left.c_str(), alloc ), alloc );
  pair.AddMember( "right", rapidjson::Value( right.c_str(), alloc ), alloc );
  pair.AddMember( "points", rapidjson::Value( rapidjson::kArrayType ), alloc );
  pair.AddMember( "leftToRight", make_matrix( matrix ), alloc );

  if( std::abs( matrix.determinant() ) > 1e-12 )
  {
    pair.AddMember( "rightToLeft",
      make_matrix( Eigen::Matrix< double, 3, 3 >( matrix.inverse() ) ), alloc );
  }
  else
  {
    pair.AddMember( "rightToLeft",
      rapidjson::Value( rapidjson::kNullType ), alloc );
  }

  rapidjson::Value pairs( rapidjson::kArrayType );
  pairs.PushBack( pair, alloc );
  document.AddMember( "pairs", pairs, alloc );

  std::ofstream output( filename );

  if( !output )
  {
    throw std::runtime_error( "Unable to write: " + filename );
  }

  rapidjson::OStreamWrapper wrapper( output );
  rapidjson::PrettyWriter< rapidjson::OStreamWrapper > writer( wrapper );
  document.Accept( writer );
}

} // end namespace viame
