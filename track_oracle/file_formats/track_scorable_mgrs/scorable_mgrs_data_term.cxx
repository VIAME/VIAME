// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "scorable_mgrs_data_term.h"
#include <track_oracle/kwiver_io_helpers.h>

#include <iostream>
#include <string>

#include <tinyxml.h>

#include <vital/logger/logger.h>

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::ostream;
using std::string;
using std::vector;
using std::map;
using std::ostringstream;
using std::istringstream;
using std::stringstream;

using kwiver::track_oracle::scorable_mgrs;

namespace kwiver {
namespace track_oracle {
namespace dt {
namespace tracking {

context mgrs_pos::c( mgrs_pos::get_context_name(), mgrs_pos::get_context_description() );

ostream& mgrs_pos::to_stream( ostream& os, const scorable_mgrs& m ) const
{
  os << m;
  return os;
}

bool mgrs_pos::from_str( const string& s, scorable_mgrs& m ) const
{
  istringstream iss( s );
  return static_cast<bool>( iss >> m );
}

void mgrs_pos::write_xml( ostream& os, const string& indent, const scorable_mgrs& m ) const
{
  os << indent << "<" << mgrs_pos::c.name << ">\n";
  if ( ! m.valid )
  {
    os << indent << "<!-- invalid mgrs -->\n";
  }
  else
  {
    for (int zone = scorable_mgrs::ZONE_BEGIN; zone < scorable_mgrs::N_ZONES; ++zone)
    {
      if ( ! m.entry_valid[ zone ] ) continue;
      os << indent << "  <mgrs_zone index=\"" << zone << "\" >\n";
      os << indent << "    <zone> " << m.zone[ zone ] << " </zone>\n";
      os << indent << "    <northing> ";
      kwiver_write_highprecision( os, m.northing[ zone ] );
      os << " </northing>\n";
      os << indent << "    <easting> ";
      kwiver_write_highprecision( os, m.easting[ zone ] );
      os << " </easting>\n";
      os << indent << "  </mgrs_zone>\n";
    }
  }
  os << indent << "</" << mgrs_pos::c.name << ">\n";
}

bool mgrs_pos::read_xml( const TiXmlElement* const_e, scorable_mgrs& m ) const
{
  TiXmlElement* e = const_cast< TiXmlElement* >( const_e );
  m.valid = false;
  for (int zone = scorable_mgrs::ZONE_BEGIN; zone < scorable_mgrs::N_ZONES; ++zone)
  {
    m.entry_valid[ zone ] = false;
  }

  TiXmlElement* zone = e->FirstChild( "mgrs_zone" )->ToElement();
  for( ; zone; zone = zone->NextSiblingElement() )
  {
    TiXmlElement* zone_e = zone->FirstChild( "zone" )->ToElement();
    TiXmlElement* northing_e = zone->FirstChild( "northing" )->ToElement();
    TiXmlElement* easting_e = zone->FirstChild( "easting" )->ToElement();
    const char* index_str = zone->Attribute( "index" );
    if ( ! ( northing_e && easting_e && index_str && zone_e->GetText() ))
    {
      LOG_ERROR( main_logger, "MGRS: zone missing index, zone, northing and/or easting at " << zone->Row() );
      return false;
    }
    istringstream iss( string(index_str) + " " + zone_e->GetText()+ " " + northing_e->GetText() + " " + easting_e->GetText() );
    int index, zone_val;
    double northing_val, easting_val;
    if ( ! ( iss >> index >> zone_val >> northing_val >> easting_val ))
    {
      LOG_ERROR( main_logger, "MGRS: couldn't parse zone / northing / easting from zone at " << zone->Row() );
      return false;
    }
    if ( ! ( (scorable_mgrs::ZONE_BEGIN <= index) && (index < scorable_mgrs::N_ZONES)))
    {
      LOG_ERROR( main_logger, "MGRS: Bad zone index " << index << " at " << zone->Row() );
      return false;
    }

    // whew
    m.entry_valid[ index ] = true;
    m.valid = true;
    m.zone[ index ] = zone_val;
    m.northing[ index ] = northing_val;
    m.easting[ index ] = easting_val;
  }

  return true;
}

vector<string> mgrs_pos::csv_headers() const
{
  vector<string> r;
  for (int zone = scorable_mgrs::ZONE_BEGIN; zone < scorable_mgrs::N_ZONES; ++zone)
  {
    ostringstream oss;
    oss << "mgrs_zone_" << zone;
    string prefix = oss.str();

    r.push_back( prefix+"_valid" );
    r.push_back( prefix+"_zone" );
    r.push_back( prefix+"_northing" );
    r.push_back( prefix+"_easting" );
  }
  return r;
}

bool mgrs_pos::from_csv( const map<string, string>& header_value_map, scorable_mgrs& m ) const
{
  m.valid = false;
  for (int index = scorable_mgrs::ZONE_BEGIN; index < scorable_mgrs::N_ZONES; ++index)
  {
    ostringstream oss;
    oss << "mgrs_zone_" << index;
    string prefix = oss.str();

    m.entry_valid[ index ] = false;
    map<string, string>::const_iterator p = header_value_map.find( prefix+"_valid" );
    if ( ( p == header_value_map.end() ||
           p->second.empty() ||
           p->second == "0" ))
    {
      continue;
    }

    stringstream ss;
    p = header_value_map.find( prefix+"_zone" );
    if ( p == header_value_map.end() )
    {
      LOG_ERROR( main_logger, "MGRS CSV: Zone index " << index << " marked valid but missing zone?" );
      return false;
    }
    ss << p->second << " ";

    p = header_value_map.find( prefix+"_northing" );
    if ( p == header_value_map.end() )
    {
      LOG_ERROR( main_logger, "MGRS CSV: Zone " << index << " marked valid but missing northing?" );
      return false;
    }
    ss << p->second << " ";

    p = header_value_map.find( prefix+"_easting" );
    if ( p == header_value_map.end() )
    {
      LOG_ERROR( main_logger, "MGRS CSV: Zone " << index << " marked valid but missing easting?" );
      return false;
    }
    ss << p->second;

    if ( ! ( ss >> m.zone[ index ] >>  m.northing[ index ] >> m.easting[ index ] ))
    {
      LOG_ERROR( main_logger, "MGRS CSV: Zone " << index << " couldn't parse northing / easting" );
      return false;
    }

    m.entry_valid[ index ] = true;
    m.valid = true;
  }
  return true;
}

ostream& mgrs_pos::to_csv( ostream& os, const scorable_mgrs& m ) const
{
  for (int zone = scorable_mgrs::ZONE_BEGIN; zone < scorable_mgrs::N_ZONES; ++zone)
  {
    if (( ! m.valid ) || ( ! m.entry_valid[zone] ))
    {
      os << "0" << "," << "" << "," << "" << "," << "";
      continue;
    }
    os << "1" << "," << m.zone[zone] << ",";
    kwiver_write_highprecision( os, m.northing[zone] );
    os << ",";
    kwiver_write_highprecision( os, m.easting[zone] );
    if ( zone != (scorable_mgrs::N_ZONES-1))
    {
      os << ",";
    }
  }
  return os;
}

} // ..tracking
} // ..dt
} // ..track_oracle
} // ..kwiver
