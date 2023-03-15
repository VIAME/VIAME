/*ckwg +29
 * Copyright 2023 by Kitware, Inc.
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
 * \brief Read habcam metadata
 */

#include "read_habcam_metadata_process.h"

#include <vital/vital_types.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <string>
#include <fstream>
#include <sstream>


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( scan_length, unsigned, "1000",
  "Number of characters at end of file to scan for metadata." );

// =============================================================================
// Private implementation class
class read_habcam_metadata_process::priv
{
public:
  explicit priv( read_habcam_metadata_process* parent );
  ~priv();

  // Configuration settings
  unsigned m_scan_length;

  // Other variables
  read_habcam_metadata_process* parent;
};


// -----------------------------------------------------------------------------
read_habcam_metadata_process::priv
::priv( read_habcam_metadata_process* ptr )
  : m_scan_length( 1000 )
  , parent( ptr )
{
}


read_habcam_metadata_process::priv
::~priv()
{
}


// =============================================================================
read_habcam_metadata_process
::read_habcam_metadata_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new read_habcam_metadata_process::priv( this ) )
{
  make_ports();
  make_config();
}


read_habcam_metadata_process
::~read_habcam_metadata_process()
{
}


// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( file_name, required );

  // -- outputs --
  declare_output_port_using_trait( metadata, optional );
  declare_output_port_using_trait( gsd, optional );
}

// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::make_config()
{
  declare_config_using_trait( scan_length );
}

// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::_configure()
{
  d->m_scan_length = config_value_using_trait( scan_length );
}

// -----------------------------------------------------------------------------
void
read_habcam_metadata_process
::_step()
{
  std::string file_name = grab_from_port_using_trait( file_name );

  kwiver::vital::metadata_vector output_md_vec;
  double output_gsd = -1.0;

  std::ifstream fin( file_name.c_str() );

  if( !fin )
  {
    throw std::runtime_error( "Unable to load: " + file_name );
  }

  std::string ascii_snippet; // = fin.readline(); // TODO

  auto meta_start = ascii_snippet.find( "pixelformat=" );

  if( meta_start == std::string::npos )
  {
    push_to_port_using_trait( metadata, output_md_vec );
    push_to_port_using_trait( gsd, output_gsd );
    return;
  }

  std::shared_ptr< kwiver::vital::metadata > output_md =
    std::make_shared< kwiver::vital::metadata >(); 

  ascii_snippet = ascii_snippet.substr( meta_start );

  std::vector< std::string > tokens;

  std::stringstream ss( ascii_snippet );
  std::string line;
  const std::string delims = "\n\t\v ,";

  while( std::getline( ss, line ) )
  {
    std::size_t prev = 0, pos;
    while( ( pos = line.find_first_of( delims, prev ) ) != std::string::npos )
    {
      if( pos > prev )
      {
        std::string token = line.substr( prev, pos-prev );
        if( !token.empty() )
        {
          tokens.push_back( token );
        }
      }
      prev = pos + 1;
    }
    if( prev < line.length() )
    {
      std::string token = line.substr( prev, std::string::npos );
      if( !token.empty() )
      {
        tokens.push_back( token );
      }
    }
  }

#define CHECK_FIELD( STR, METAID )                       \
  {                                                      \
    std::string field = STR ;                            \
    std::size_t pos = field.size() + 1;                  \
    if( token.substr( 0, pos ) == field + "=" )          \
    {                                                    \
      try                                                \
      {                                                  \
        double value = std::stod( token.substr( pos ) ); \
                                                         \
        if( value > 0.0 )                                \
        {                                                \
          output_md->add< METAID >( value );             \
        }                                                \
      }                                                  \
      catch( ... )                                       \
      {                                                  \
      }                                                  \
    }                                                    \
  }

  for( std::string token : tokens )
  {
    CHECK_FIELD( "hdg", kwiver::vital::VITAL_META_SENSOR_YAW_ANGLE );
    CHECK_FIELD( "pitch", kwiver::vital::VITAL_META_SENSOR_PITCH_ANGLE );
    CHECK_FIELD( "roll", kwiver::vital::VITAL_META_SENSOR_ROLL_ANGLE );
    CHECK_FIELD( "alt0", kwiver::vital::VITAL_META_SENSOR_ALTITUDE );
    CHECK_FIELD( "alt1", kwiver::vital::VITAL_META_SENSOR_ALTITUDE );
  }

  output_md_vec.push_back( output_md );

  push_to_port_using_trait( metadata, output_md_vec );
  push_to_port_using_trait( gsd, output_gsd );
}

} // end namespace core

} // end namespace viame
