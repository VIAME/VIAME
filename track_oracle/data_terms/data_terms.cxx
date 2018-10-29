/*ckwg +5
 * Copyright 2014-2018 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "data_terms.h"

#include <sstream>

#include <tinyxml.h>

#include <track_oracle/core/kwiver_io_helpers.h>
#include <track_oracle/aries_interface/aries_interface.h>
#include <track_oracle/core/kwiver_io_base_instantiation.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::vector;
using std::map;
using std::ostream;
using std::ostringstream;
using std::istringstream;
using std::stringstream;
using std::streamsize;
using std::pair;
using std::string;

namespace // anon
{

template< typename T >
bool kwiver_csv_read( const vector<string>& headers,
                      const map<string, string>& header_value_map,
                      T& d )
{
  string s;
  for (size_t i=0; i<headers.size(); ++i)
  {
    map<string,string>::const_iterator p = header_value_map.find( headers[i] );
    if (p == header_value_map.end()) return false;
    s += p->second + " ";
  }
  return kwiver::track_oracle::kwiver_read( s, d );
}

void
vector_unsigned_to_stream( ostream& os, const vector<unsigned>& d )
{
  for (size_t i=0, n=d.size(); i<n; ++i)
  {
    os << d[i];
    if ( i != n-1)
    {
      os << ":";
    }
  }
}

void
vector_unsigned_from_str( const string& s, vector<unsigned>& d )
{
  string src(s);
  size_t found = src.find_first_of( ":" );
  while ( found != string::npos )
  {
    src[found] = ' ';
    found = src.find_first_of( ":", found+1);
  }
  istringstream iss( src );
  d.clear();
  unsigned tmp;
  while ((iss >> tmp))
  {
    d.push_back( tmp );
  }
}


} // anon

namespace kwiver {
namespace track_oracle {

namespace dt {

#define DEF_DT(NAME) \
  context NAME::c( NAME::get_context_name(), NAME::get_context_description() );

namespace detection {
  DEF_DT( detection_id );
}

namespace tracking {

  DEF_DT( external_id );
  DEF_DT( timestamp_usecs );
  DEF_DT( frame_number );
  DEF_DT( fg_mask_area );
  DEF_DT( track_location );
  DEF_DT( obj_x );
  DEF_DT( obj_y );
  DEF_DT( obj_location );
  DEF_DT( velocity_x );
  DEF_DT( velocity_y );
  DEF_DT( bounding_box );
  DEF_DT( world_x );
  DEF_DT( world_y );
  DEF_DT( world_z );
  DEF_DT( world_location );
  DEF_DT( world_gcs );
  DEF_DT( latitude );
  DEF_DT( longitude );
  DEF_DT( time_stamp );
  DEF_DT( track_uuid );
  DEF_DT( track_style );

//
// track_location I/O overrides
//

ostream& track_location::to_stream( ostream& os, const vgl_point_2d<double>& d ) const
{
  return kwiver_write( os, d, " " );
}

ostream& track_location::to_csv( ostream&  os, const vgl_point_2d<double>& d ) const
{
  return kwiver_write( os, d, "," );
}

bool track_location::from_str( const string& s, vgl_point_2d<double>& d ) const
{
  return kwiver_read( s, d );
}

vector<string> track_location::csv_headers() const
{
  return kwiver_csv_header_pair( track_location::c.name, "_x", "_y");
}

bool track_location::from_csv( const map<string, string>& header_value_map, vgl_point_2d<double>& d ) const
{
  return kwiver_csv_read( track_location::csv_headers(), header_value_map, d );
}


//
// obj_location I/O overrides
//

ostream& obj_location::to_stream( ostream& os, const vgl_point_2d<double>& d ) const
{
  return kwiver_write( os, d, " " );
}

ostream& obj_location::to_csv( ostream& os, const vgl_point_2d<double>& d ) const
{
  return kwiver_write( os, d, "," );
}

bool obj_location::from_str( const string& s, vgl_point_2d<double>& d ) const
{
  return kwiver_read( s, d );
}

vector<string> obj_location::csv_headers() const
{
  return kwiver_csv_header_pair( obj_location::c.name, "_x", "_y");
}

bool obj_location::from_csv( const map<string, string>& header_value_map, vgl_point_2d<double>& d ) const
{
  return kwiver_csv_read( obj_location::csv_headers(), header_value_map, d );
}

//
// bounding_box I/O overrides
//

ostream& bounding_box::to_stream( ostream& os, const vgl_box_2d<double>& d ) const
{
  return kwiver_write( os, d, " " );
}

ostream& bounding_box::to_csv( ostream& os, const vgl_box_2d<double>& d ) const
{
  return kwiver_write( os, d, "," );
}

bool bounding_box::from_str( const string& s, vgl_box_2d<double>& d ) const
{
  return kwiver_read( s, d );
}

vector<string> bounding_box::csv_headers() const
{
  return kwiver_box_2d_headers( bounding_box::c.name );
}

bool bounding_box::from_csv( const map<string, string>& header_value_map, vgl_box_2d<double>& d ) const
{
  return kwiver_csv_read( bounding_box::csv_headers(), header_value_map, d );
}

bool bounding_box::read_xml( const TiXmlElement* e, vgl_box_2d<double>& d ) const
{
  if (! e->GetText() ) return false;
  return kwiver_read( e->GetText(), d );
}

void bounding_box::write_xml( ostream& os,
                              const string& indent,
                              const vgl_box_2d<double>& d ) const
{
  os << indent << "<" << bounding_box::c.name
     << " w=\"" << d.width() << "\" h=\"" << d.height() << "\" > ";
  this->to_stream( os, d );
  os << " </" << name << ">\n";
}


//
// world x/y/z output overrides (increase precision)
//


ostream& world_x::to_stream( ostream& os, const double& d ) const
{
  return kwiver_write_highprecision( os, d );
}

ostream& world_y::to_stream( ostream& os, const double& d ) const
{
  return kwiver_write_highprecision( os, d );
}

ostream& world_z::to_stream( ostream& os, const double& d ) const
{
  return kwiver_write_highprecision( os, d );
}

//
// world_location I/O overrides
//

ostream& world_location::to_stream( ostream& os, const vgl_point_3d<double>& d ) const
{
  streamsize old_prec = os.precision( 10 );
  kwiver_write( os, d, " " );
  os.precision( old_prec );
  return os;
}

ostream& world_location::to_csv( ostream& os, const vgl_point_3d<double>& d ) const
{
  streamsize old_prec = os.precision( 10 );
  kwiver_write( os, d, "," );
  os.precision( old_prec );
  return os;
}

bool world_location::from_str( const string& s, vgl_point_3d<double>& d ) const
{
  return kwiver_read( s, d );
}

vector<string> world_location::csv_headers() const
{
  return kwiver_point_3d_headers( world_location::c.name );
}

bool world_location::from_csv( const map<string, string>& header_value_map, vgl_point_3d<double>& d ) const
{
  return kwiver_csv_read( world_location::csv_headers(), header_value_map, d );
}

//
// lat/lon output overrides
//

ostream& latitude::to_stream( ostream& os, const double& d ) const
{
  return kwiver_write_highprecision( os, d );
}

ostream& longitude::to_stream( ostream& os, const double& d) const
{
  return kwiver_write_highprecision( os, d );
}

//
// timestamp
//

ostream& time_stamp::to_stream( ostream& os, const vital::timestamp& ts ) const
{
  return kwiver_write( os, ts );
}

bool time_stamp::from_str( const string& s, vital::timestamp& ts ) const
{
  return kwiver_read( s, ts );
}

bool time_stamp::read_xml( const TiXmlElement* e, vital::timestamp& ts ) const
{
  TiXmlHandle h( const_cast<TiXmlElement*>(e) );
  TiXmlElement* frame_e = h.FirstChild( "frame" ).ToElement();
  TiXmlElement* time_e = h.FirstChild( "time" ).ToElement();
  if ( (! frame_e) && (! time_e)) return false;
  string frame_s =
    frame_e && frame_e->GetText()
    ? frame_e->GetText()
    : "none";
  string time_s =
    time_e && time_e->GetText()
    ? time_e->GetText()
    : "none";
  return kwiver_ts_string_read( frame_s, time_s, ts );
}

void time_stamp::write_xml( ostream& os, const string& indent, const vital::timestamp& ts ) const
{
  pair< string, string > ts_strings = kwiver_ts_to_strings( ts );
  os << indent << "<" << time_stamp::c.name << ">\n";
  os << indent << "  <frame> " << ts_strings.first << " </frame>\n";
  os << indent << "  <time> " << ts_strings.second << " </time>\n";
  os << indent << "</" << time_stamp::c.name << ">\n";
}

vector<string> time_stamp::csv_headers() const
{
  vector<string> r;
  r.push_back( "ts_frame" );
  r.push_back( "ts_time" );
  return r;
}

bool time_stamp::from_csv( const map<string, string>& header_value_map, vital::timestamp& ts ) const
{
  map<string, string>::const_iterator ts_f = header_value_map.find( "ts_frame" );
  map<string, string>::const_iterator ts_t = header_value_map.find( "ts_time" );
  bool okay = true;
  if (ts_f == header_value_map.end())
  {
    LOG_ERROR( main_logger, "timestamp::csv: no header 'ts_frame'" );
    okay = false;
  }
  if (ts_t == header_value_map.end())
  {
    LOG_ERROR( main_logger, "timestamp::csv: no header 'ts_time'" );
    okay = false;
  }
  return
    okay
    ? kwiver_ts_string_read( ts_f->second, ts_t->second, ts )
    : false;
}

ostream& time_stamp::to_csv( ostream& os, const vital::timestamp& ts ) const
{
  pair< string, string > ts_strings = kwiver_ts_to_strings( ts );
  os << ts_strings.first << "," << ts_strings.second;
  return os;
}

//
// uid
//

ostream& track_uuid::to_stream( ostream& os, const vital::uid& uid ) const
{
  return kwiver_write( os, uid );
}

bool track_uuid::from_str( const string& s, vital::uid& uid ) const
{
  return kwiver_read( s, uid );
}

} // ...tracking

namespace events {

  DEF_DT( event_id );
  DEF_DT( event_type );
  DEF_DT( event_probability );
  DEF_DT( source_track_ids );
  DEF_DT( actor_track_rows );
  DEF_DT( kpf_activity_domain );
  DEF_DT( kpf_activity_start );
  DEF_DT( kpf_activity_stop );

//
// event type
//

ostream& event_type::to_stream( ostream& os, const int& d ) const
{
  const map<size_t, string>& i2a = aries_interface::index_to_activity_map();
  map<size_t, string>::const_iterator p = i2a.find( d );
  if ( p == i2a.end() )
  {
    os << "invalid_event";
  }
  else
  {
    os << "virat:" << p->second;
  }
  return os;
}

bool event_type::from_str( const string& s, int& d ) const
{
  bool okay = false;
  try
  {
    if ( s == "invalid_event" )
    {
      // not ideal, but a true fix must await event unification
      d = aries_interface::activity_to_index( "NotScored" );
      okay = true;
    }
    else
    {
      const string domain_tag( "virat:" );
      const size_t tag_len = domain_tag.size();
      size_t domain_pos = s.find( domain_tag );
      if ( domain_pos == 0 )
      {
        d = aries_interface::activity_to_index( s.substr( tag_len ));
        okay = true;
      }
      else
      {
        LOG_WARN( main_logger, "event_string '" << s << "' has no domain tag" );
      }
    }
  }
  catch ( const aries_interface_exception& e )
  {
    LOG_ERROR( main_logger, e.what() );
  }

  return okay;
}

vector<string> event_type::csv_headers() const
{
  vector<string> r;
  r.push_back( "event_domain" );
  r.push_back( "event_type" );
  return r;
}

ostream& event_type::to_csv( ostream& os, const int& d ) const
{
  os << "virat,";
  const map<size_t, string>& i2a = aries_interface::index_to_activity_map();
  map<size_t, string>::const_iterator p = i2a.find( d );
  if ( p == i2a.end() )
  {
    os << "invalid_event";
  }
  else
  {
    os << p->second;
  }
  return os;
}

bool event_type::from_csv( const map<string, string>& header_value_map, int& d ) const
{
  map<string, string>::const_iterator e_d = header_value_map.find( "event_domain" );
  map<string, string>::const_iterator e_t = header_value_map.find( "event_type" );
  bool okay = true;
  if ( e_d == header_value_map.end() )
  {
    LOG_ERROR( main_logger, "event_type::from_csv: no header 'event_domiain'" );
    okay = false;
  }
  if ( e_t == header_value_map.end() )
  {
    LOG_ERROR( main_logger, "event_type::from_csv: no header 'event_type'" );
    okay = false;
  }
  return
    okay
    ? this->from_str( e_d->second+":"+e_t->second, d )
    : false;
}

void event_type::write_xml( ostream& os, const string& indent, const int& d ) const
{
  os << indent << "<" << event_type::c.name << " domain=\"virat\"  index=\"" << d << "\" > ";
  const map<size_t, string>& i2a = aries_interface::index_to_activity_map();
  map<size_t, string>::const_iterator p = i2a.find( d );
  if ( p == i2a.end() )
  {
    os << "invalid_event";
  }
  else
  {
    os << p->second;
  }
  os << " </" << event_type::c.name << ">\n";
}

bool event_type::read_xml( const TiXmlElement* const_e, int& d ) const
{
  TiXmlElement* e = const_cast< TiXmlElement* >( const_e );
  if ( ! e->GetText() ) return false;
  const char* domain_str = e->Attribute( "domain" );
  if ( ! domain_str )
  {
    LOG_ERROR( main_logger, "event_type: missing domain at " << e->Row() );
    return false;
  }
  return this->from_str( string(domain_str)+":"+e->GetText(), d );
}

//
// source track IDs
// Note that the stream rep is colon separated for CSV compatibility,
// but from_str() works with both colon and space separated...
//

ostream& source_track_ids::to_stream( ostream& os, const vector<unsigned>& d ) const
{
  vector_unsigned_to_stream( os, d );
  return os;
}

bool source_track_ids::from_str( const string& s, vector<unsigned>& d ) const
{
  vector_unsigned_from_str( s, d );
  return true;
}

//
// actor track rows (hmm, not really PORTABLE)
//


ostream& actor_track_rows::to_stream( ostream& os, const track_handle_list_type& d ) const
{
  vector<unsigned> tmp;
  for (auto i: d)
  {
    tmp.push_back( i.row );
  }
  vector_unsigned_to_stream( os, tmp );
  return os;
}

bool actor_track_rows::from_str( const string& s, track_handle_list_type& d ) const
{
  vector<unsigned> tmp;
  vector_unsigned_from_str( s, tmp );
  for (auto i: tmp)
  {
    d.push_back( track_handle_type( i ));
  }
  return true;
}


} // ...events

namespace virat {

  DEF_DT( descriptor_classifier );

//
// VIRAT descriptor classifier.
// I/O copied from source_track_ids.  Maybe should make this the default?
//

ostream& descriptor_classifier::to_stream( ostream& os, const vector<double>& d ) const
{
  for (size_t i=0, n=d.size(); i<n; ++i)
  {
    os << d[i];
    if ( i != n-1)
    {
      os << ":";
    }
  }
  return os;
}

bool descriptor_classifier::from_str( const string& s, vector<double>& d ) const
{
  string src(s);
  size_t found = src.find_first_of( ":" );
  while ( found != string::npos )
  {
    src[found] = ' ';
    found = src.find_first_of( ":", found+1);
  }
  istringstream iss( src );
  d.clear();
  unsigned tmp;
  while ((iss >> tmp))
  {
    d.push_back( tmp );
  }
  return true;
}

void descriptor_classifier::write_xml( ostream& os, const string& indent, const vector<double>& d ) const
{
  const map<size_t, string>& i2a = aries_interface::index_to_activity_map();
  size_t n_nonzero = 0, n = d.size();
  for (size_t i=0; i<n; ++i)
  {
    if (d[i] != 0)
    {
      ++n_nonzero;
    }
  }
  os << indent << "<" << descriptor_classifier::c.name << " size=\"" << n << "\" n_nonzero=\"" << n_nonzero << "\" >\n";
  for (size_t i=0; i<n; ++i)
  {
    if (d[i] != 0)
    {
      map<size_t, string>::const_iterator probe = i2a.find(i);
      if ( probe == i2a.end() )
      {
        os << "<!-- invalid activity index " << i << " (probability " << d[i] << ") -->\n";
      }
      else
      {
        os << indent << "  <probability activity=\"" << probe->second << "\" value=\"" << d[i] << "\" />\n";
      }
    }
  }
  os << indent << "</" << descriptor_classifier::c.name << ">\n";
}

bool descriptor_classifier::read_xml( const TiXmlElement* const_e, vector<double>& d ) const
{
  TiXmlElement* e = const_cast< TiXmlElement* >( const_e );
  const map<size_t, string>& i2a = aries_interface::index_to_activity_map();
  d.resize( i2a.size(), 0.0 );
  for ( TiXmlElement* probNode = e->FirstChildElement( "probability" );
        probNode != 0;
        probNode = probNode->NextSiblingElement( "probability" ) )
  {
    size_t activity_idx;
    try
    {
      activity_idx = kwiver::track_oracle::aries_interface::activity_to_index( probNode->Attribute( "activity" ));
    }
    catch (kwiver::track_oracle::aries_interface_exception& /*e*/)
    {
      LOG_ERROR( main_logger, "Couldn't recognize " << probNode->Attribute("activity")
                 << " as a valid activity?");
      return false;
    }
    double prob;
    if (probNode->QueryDoubleAttribute( "value", &prob )  != TIXML_SUCCESS )
    {
      LOG_ERROR( main_logger, "Couldn't find a probability at " << probNode->Row() << "?");
      prob = 0;
    }
    d[ activity_idx ] = prob;
  }
  return true;
}


} // ...virat

#undef DEF_DT

} // ...dt

} // ...track_oracle
} // ...kwiver
