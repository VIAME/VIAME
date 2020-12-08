// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "file_format_apix.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <shapefil.h>
#include <geographic/geo_coords.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::map;
using std::string;

namespace // anon
{

// a helper class so that DBFClose is always called even if we return on an error
struct shp_handle_type
{
  DBFHandle hDBF;
  shp_handle_type( const string& filename )
  {
    hDBF = DBFOpen( filename.c_str(), "rb" );
  }
  ~shp_handle_type()
  {
    DBFClose( hDBF );
  }
};

} // anon

namespace kwiver {
namespace track_oracle {

apix_reader_opts&
apix_reader_opts
::operator=( const file_format_reader_opts_base& rhs_base )
{
  const apix_reader_opts* rhs = dynamic_cast<const apix_reader_opts*>(&rhs_base);

  if (rhs)
  {
    this->set_verbose( rhs->verbose );
  }
  else
  {
    LOG_WARN( main_logger, "Assigned a non-apix options structure to a apix options structure: Slicing the class");
  }

  return *this;
}

bool
file_format_apix
::inspect_file( const string& fn ) const
{
  shp_handle_type h( fn );
  return ( h.hDBF != NULL );
}

bool
file_format_apix
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  shp_handle_type h( fn );
  if ( h.hDBF == NULL )
  {
    LOG_ERROR( main_logger,  "Couldn't open '" << fn << "' to read an APIX track" );
    return false;
  }

  // load field -> field_index map
  map< string, int > field_map;
  for (int i=0; i<DBFGetFieldCount( h.hDBF ); ++i)
  {
    char title[12];
    int width, decimals;
    DBFFieldType ft = DBFGetFieldInfo(  h.hDBF, i, title, &width, &decimals );
    if (ft != FTInvalid )
    {
      field_map[ title ] = i;
    }
  }

  // verify that the required fields are present
  const char* reqFields[] = {"Latitude","Longitude","DataUTCTim","DataTimeMS","MGRS","FrameNum", 0};
  bool allOkay = true;
  for (unsigned i=0; reqFields[i] != 0; ++i)
  {
    if ( field_map.find( reqFields[i] ) == field_map.end() )
    {
      LOG_ERROR( main_logger,  "APIX track file '" << fn << "' does not contain a '" << reqFields[i] << "' field" );
      allOkay = false;
    }
  }
  if ( ! allOkay ) return false;

  // load the track
  track_apix_type apix_track;
  track_handle_type t = apix_track.create();
  for (int i=0; i<DBFGetRecordCount( h.hDBF ); ++i)
  {
    frame_handle_type f = apix_track.create_frame();
    apix_track[ f ].lat() = DBFReadDoubleAttribute( h.hDBF, i, field_map[ "Latitude" ] );
    apix_track[ f ].lon() = DBFReadDoubleAttribute( h.hDBF, i, field_map[ "Longitude" ] );
    int tsecs = DBFReadIntegerAttribute( h.hDBF, i, field_map[ "DataUTCTim" ] );
    int tmsecs = DBFReadIntegerAttribute( h.hDBF, i, field_map[ "DataTimeMS" ] );
    int frame_num = DBFReadIntegerAttribute( h.hDBF, i, field_map[ "FrameNum" ] );
    apix_track[ f ].utc_timestamp() = vital::timestamp( (tsecs*1.0e6) + (tmsecs*1.0e3), frame_num );

    if ( this->opts.verbose )
    {
      // double-check MGRS vs lat-lon
      string mgrs = DBFReadStringAttribute( h.hDBF, i, field_map[ "MGRS" ] );
      kwiver::geographic::geo_coords mgrs_coords( mgrs );
      kwiver::geographic::geo_coords latlon_coords( apix_track[f].lat(), apix_track[f].lon() );

      if ( ! ( mgrs_coords.is_valid() && latlon_coords.is_valid() ))
      {
        LOG_INFO( main_logger,  "APIX track file '" << fn << "' frame " << frame_num << ": geocoord error: "
                  << "mgrs '" << mgrs << "'; valid: " << mgrs_coords.is_valid() << "; lat/lon "
                  << apix_track[f].lat() << "," << apix_track[f].lon() << ": valid: " << latlon_coords.is_valid() );
        continue;
      }
      double dEasting = latlon_coords.easting() - mgrs_coords.easting();
      double dNorthing = latlon_coords.northing() - mgrs_coords.northing();
      LOG_INFO( main_logger, "Info: APIX track file '" << fn << "' frame " << frame_num << ": lat/long vs MGRS "
                << dEasting << " , " << dNorthing << " " << tsecs << "." << tmsecs );
    }
  }

  tracks.push_back( t );
  return true;
}

} // ...track_oracle
} // ...kwiver

