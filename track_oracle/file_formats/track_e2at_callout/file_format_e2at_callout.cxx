/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_e2at_callout.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <kwiversys/RegularExpression.hxx>

#include <track_oracle/utils/tokenizers.h>
#include <track_oracle/data_terms/data_terms.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::getline;
using std::ifstream;
using std::istream;
using std::istringstream;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;

namespace { // anon

//
// e.g. "01:34" -> 94 seconds
//
// empty string indicates no errors, for silent testing
// of possibly non-compliant file formats
//

pair< string, double >
parse_time( const string& s )
{
  kwiversys::RegularExpression re("^([0-9]+):([0-9]+)$");
  if ( re.find( s ))
  {
    istringstream iss( re.match(1) + " " + re.match(2) );
    double min, sec;
    if ( (iss >> min >> sec ))
    {
      return make_pair( "", (min * 60.0) + sec );
    }
    else
    {
      return make_pair( "e2at-callout: Logic error: couldn't parse timestamp from '"+iss.str()+"'", 0 );
    }
  }
  else
  {
    return make_pair( "e2at-callout: timestamp regexp fails on '"+s+"'", 0 );
  }
}

bool
parse_latlon( const string& s,
              double& lat,
              double& lon )
{
  kwiversys::RegularExpression re("([0-9\\-\\.]+)([NnSs])([0-9\\-\\.]+)([WwEe])");

  if ( ! re.find( s )) return false;
  istringstream latss( re.match(1) ), lonss( re.match(3) );
  if ( ! ( latss >> lat ))
  {
    LOG_ERROR( main_logger, "Logic error: expected to parse a latitude from '" << re.match(1) << "'" );
    return false;
  }
  if ( ! ( lonss >> lon ))
  {
    LOG_ERROR( main_logger, "Logic error: expected to parse a longitude from '" << re.match(3) << "'" );
    return false;
  }
  char ns = re.match(2)[0];
  if ((ns == 's') || (ns == 'S'))
  {
    lat = -1.0 * lat;
  }
  char ew = re.match(4)[0];
  if ((ew == 'w') || (ew == 'W'))
  {
    lon = -1.0 * lon;
  }
  return true;
}


} // anon

namespace kwiver {
namespace track_oracle {

void
trim( string& s )
{
  size_t trim = s.find_last_not_of( " \n\r\t" );
  if ( trim == string::npos )
  {
    s.clear();
  }
  else
  {
    s.erase( trim+1 );
  }
}

bool
file_format_e2at_callout
::inspect_file( const string& fn ) const
{
  //
  // Look for a CSV whose second token is parsable as a timestamp.
  //

  ifstream is( fn.c_str() );
  if (! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "' for inspection" );
    return false;
  }

  // Try the first two lines, because the first is PROBABLY the header.
  for (size_t i=0; i<2; ++i)
  {
    string line;
    if ( ! getline( is, line ))
    {
      if (i == 0)
      {
        LOG_ERROR( main_logger, "Couldn't read first line from '" << fn << "' during inspection" );
      }
      return false;
    }
    trim( line );
    vector< string > tokens;

    tokens = csv_tokenizer::parse( line );
    if (tokens.size() < 5) return false;
    pair< string, double > probe = parse_time( tokens[1] );

    // if no errors, we're good!
    if ( probe.first.empty() ) return true;
  }

  return false;
}

bool
file_format_e2at_callout
::read( const string& fn,
        track_handle_list_type& tracks ) const
{
  ifstream is( fn.c_str() );
  if ( ! is )
  {
    LOG_ERROR( main_logger, "Couldn't open '" << fn << "'" );
    return false;
  }

  return this->read( is, tracks );
}

bool
file_format_e2at_callout
::read( istream& is,
        track_handle_list_type& tracks ) const
{
  track_e2at_callout_type callout;
  track_field< dt::tracking::track_style > track_style;
  string line;
  vector< string > tokens;
  tracks.clear();

  size_t c = 0;
  while ( getline( is ,line ) )
  {
    trim( line );
    ++c;
    tokens = csv_tokenizer::parse( line );
    if (tokens.size() < 5)
    {
      LOG_WARN( main_logger, "E2AT callout: line " << c << ": skipping too-short line '" << line << "'" );
      continue;
    }

    pair< string, double > ts_probe = parse_time( tokens[1] );
    if ( ! ts_probe.first.empty() )
    {
      LOG_WARN( main_logger, "E2AT callout: line " << c << ": skipping due to timestamp conversion failure '" << ts_probe.first << "'; line is '" << line << "'" );
      continue;
    }

    track_handle_type h = callout.create();
    callout( h ).clip_filename() = tokens[0];
    callout( h ).start_time_secs() = ts_probe.second;
    // leave end_time_secs unset for now
    callout( h ).basic_annotation() = tokens[2];
    callout( h ).augmented_annotation() = tokens[3];
    double lat, lon;
    if ( parse_latlon( tokens[4], lat, lon ))
    {
      callout( h ).latitude() = lat;
      callout( h ).longitude() = lon;
    }

    track_style( h.row ) = "trackE2ATCallout";

    tracks.push_back( h );
  }

  return true;
}

} // ...track_oracle
} // ...kwiver
