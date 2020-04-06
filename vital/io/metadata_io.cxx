/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief Implementation of file IO functions for metadata
 *
 */

#include "metadata_io.h"

#include <fstream>

#include <vital/exceptions.h>
#include <vital/util/tokenize.h>
#include <vital/types/geodesy.h>
#include <vital/types/metadata_traits.h>
#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace vital {


/// Extract an image file basename from metadata and (if needed) frame number
std::string
basename_from_metadata(metadata_sptr md,
                       frame_id_t frame)
{
  typedef kwiversys::SystemTools  ST;

  std::string basename = "frame";
  if( md )
  {
    if ( auto& mdi = md->find( kwiver::vital::VITAL_META_IMAGE_URI ) )
    {
      return ST::GetFilenameWithoutLastExtension( mdi.as_string() );
    }
  }

  if( md )
  {
    if ( auto& mdi = md->find( kwiver::vital::VITAL_META_VIDEO_URI ) )
    {
      basename = ST::GetFilenameWithoutLastExtension( mdi.as_string() );
    }
  }

  char frame_str[6];
  std::snprintf(frame_str, 6, "%05d", static_cast<int>(frame));
  basename += "-" + std::string(frame_str);

  return basename;
}

// ----------------------------------------------------------------------------
/// Read in a POS file, producing a metadata object
metadata_sptr
read_pos_file( path_t const& file_path )
{
  // Check that file exists
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    VITAL_THROW( file_not_found_exception,
                 file_path, "File does not exist." );
  }
  else if (  kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_not_found_exception,
                 file_path, "Path given doesn't point to a regular file!" );
  }

  // Reading in input file data
  std::ifstream in_stream( file_path.c_str(), std::fstream::in );
  if ( ! in_stream )
  {
    VITAL_THROW( file_not_read_exception, file_path,
                 "Could not open file at given path." );
  }

  // Read the file

  std::string line;
  getline( in_stream, line );

  // Tokenize the string
  std::vector< std::string > tokens;
  kwiver::vital::tokenize( line, tokens, ",", true );

  unsigned int base = 0;

  // some POS files do not have the source name
  if ( ( tokens.size() < 14 ) || ( tokens.size() > 15 ) )
  {
    std::ostringstream ss;
    ss  << "Incorrect number of fields found in file "
        << file_path
        << "  (discovered " << tokens.size() << " field(s), expected "
        << "14 or 15).";
    VITAL_THROW( vital::invalid_data, ss.str() );
  }

  // make a new metadata container.
  auto md = std::make_shared<metadata>();
  md->add( NEW_METADATA_ITEM( VITAL_META_METADATA_ORIGIN, std::string( "POS-file") ) );

  if ( tokens.size() == 15 )
  {
    base = 1;
    md->add( NEW_METADATA_ITEM( VITAL_META_IMAGE_SOURCE_SENSOR, tokens[0] ) );
  }
  else
  {
    // Set name to "KWIVER"
    md->add( NEW_METADATA_ITEM( VITAL_META_IMAGE_SOURCE_SENSOR, std::string( "KWIVER" ) ) );
  }

  md->add( NEW_METADATA_ITEM( VITAL_META_SENSOR_YAW_ANGLE, std::stod( tokens[base + 0] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_SENSOR_PITCH_ANGLE, std::stod( tokens[ base + 1] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_SENSOR_ROLL_ANGLE, std::stod( tokens[base + 2] ) ) );

  // altitude is in feet in a POS file and needs to be converted to meters
  constexpr double feet2meters = 0.3048;
  const double altitude = std::stod(tokens[base + 5]) * feet2meters;
  kwiver::vital::vector_3d raw_geo{ std::stod( tokens[ base + 4 ]),
                                    std::stod( tokens[ base + 3 ] ),
                                    altitude };
  kwiver::vital::geo_point geo_pt{ raw_geo, SRID::lat_lon_WGS84 };
  md->add( NEW_METADATA_ITEM( VITAL_META_SENSOR_LOCATION, geo_pt) );

  md->add( NEW_METADATA_ITEM( VITAL_META_GPS_SEC, std::stod( tokens[base + 6] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_GPS_WEEK, std::stoi( tokens[base + 7] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_NORTHING_VEL, std::stod( tokens[base + 8] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_EASTING_VEL, std::stod( tokens[base + 9] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_UP_VEL, std::stod( tokens[base + 10] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_IMU_STATUS, std::stoi( tokens[base + 11] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_LOCAL_ADJ, std::stoi( tokens[base + 12] ) ) );
  md->add( NEW_METADATA_ITEM( VITAL_META_DST_FLAGS, std::stoi( tokens[base + 13] ) ) );

  return md;
}

// ----------------------------------------------------------------------------
// Output the given \c metadata object to the specified POS file path
void
write_pos_file( metadata const& md,
                path_t const& file_path )
{

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_write_exception, file_path,
                 "Path given is a directory, can not write file." );
  }

  // Check that the directory of the given filepath exists, creating necessary
  // directories where needed.
  std::string parent_dir = kwiversys::SystemTools::GetFilenamePath(
    kwiversys::SystemTools::CollapseFullPath( file_path ));
  if ( ! kwiversys::SystemTools::FileIsDirectory( parent_dir ) )
  {
    if ( ! kwiversys::SystemTools::MakeDirectory( parent_dir ) )
    {
      VITAL_THROW( file_write_exception, parent_dir,
                   "Attempted directory creation, but no directory created! No idea what happened here..." );
    }
  }

  // lambda function to retrive metadata value or fallback to default
  auto print_default = [&md] (std::ostream& os,
                              vital_metadata_tag const& tag,
                              std::string const& dflt) -> std::ostream&
  {
    if ( auto& mdi = md.find(tag) )
    {
      mdi.print_value(os);
    }
    else
    {
      os << dflt;
    }
    return os;
  };

  // open output file and write the POS data
  std::ofstream ofile( file_path.c_str() );
  ofile.precision( 12 );
  print_default( ofile, VITAL_META_IMAGE_SOURCE_SENSOR, "KWIVER" ) << ", ";
  print_default( ofile, VITAL_META_SENSOR_YAW_ANGLE,    "0" ) << ", ";
  print_default( ofile, VITAL_META_SENSOR_PITCH_ANGLE,  "0" ) << ", ";
  print_default( ofile, VITAL_META_SENSOR_ROLL_ANGLE,   "0" ) << ", ";

  if ( auto& mdi = md.find( VITAL_META_SENSOR_LOCATION ) )
  {
    kwiver::vital::geo_point geo_pt;
    mdi.data( geo_pt );
    auto const& raw_loc = geo_pt.location( SRID::lat_lon_WGS84 );
    // altitude is in feet in a POS file and needs to be converted to feet
    constexpr double feet2meters = 0.3048;
    ofile << raw_loc[1] << ", "
          << raw_loc[0] << ", "
          << raw_loc[2] / feet2meters << ", ";
  }
  else
  {
    ofile << "0, 0, 0, ";
  }

  print_default( ofile, VITAL_META_GPS_SEC,       "0" ) << ", ";
  print_default( ofile, VITAL_META_GPS_WEEK,      "0" ) << ", ";
  print_default( ofile, VITAL_META_NORTHING_VEL,  "0" ) << ", ";
  print_default( ofile, VITAL_META_EASTING_VEL,   "0" ) << ", ";
  print_default( ofile, VITAL_META_UP_VEL,        "0" ) << ", ";
  print_default( ofile, VITAL_META_IMU_STATUS,    "-1" ) << ", ";
  print_default( ofile, VITAL_META_LOCAL_ADJ,     "0" ) << ", ";
  print_default( ofile, VITAL_META_DST_FLAGS,     "0" );

  ofile.close();
}


} } // end of namespace
