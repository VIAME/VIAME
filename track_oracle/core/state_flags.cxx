/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "state_flags.h"

#include <string>
#include <sstream>
#include <mutex>

#include <tinyxml.h>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/element_descriptor.h>

#include <vital/util/tokenize.h>
#include <vital/util/string.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::map;
using std::string;
using std::pair;
using std::vector;
using std::make_pair;
using std::runtime_error;
using std::ostream;
using std::istream;
using std::istringstream;

namespace kwiver {
namespace track_oracle {

struct state_flags_backend
{
  typedef map< string, size_t > string_index_map_t;
  typedef map< string, size_t >::iterator sim_it;
  typedef map< string, size_t >::const_iterator sim_cit;

  string_index_map_t component_map;
  map< size_t, string_index_map_t > status_maps;

  state_flags_backend() :
    special_slot_zero_tag( "_invalid_component" )
  {
    component_map[ special_slot_zero_tag ] = 0;
  }

  static state_flags_backend& get_instance();

  pair< size_t, size_t > set_flag( const string& component, const string& status );
  map< string, string > get_map( const vector< size_t >& data );

private:
  static state_flags_backend* impl;
  std::mutex api_lock;
  string special_slot_zero_tag;
};

state_flags_backend* state_flags_backend::impl = 0;

state_flags_backend&
state_flags_backend
::get_instance()
{
  if ( state_flags_backend::impl == 0 )
  {
    state_flags_backend::impl = new state_flags_backend();
  }
  return *state_flags_backend::impl;
}


pair< size_t, size_t >
state_flags_backend
::set_flag( const string& component, const string& status )
{
  std::lock_guard< std::mutex > lock( this->api_lock );

  // first, locate or assign the component
  string_index_map_t& c = this->component_map;
  sim_it component_probe = c.find( component );
  if ( component_probe == c.end() )
  {
    size_t new_index = c.size();
    pair< sim_it, bool> p = c.insert( make_pair(component, new_index) );
    component_probe = p.first;
    string_index_map_t new_status_map;
    new_status_map[ "invalid_status" ] = 0;
    this->status_maps[ new_index ] = new_status_map;
  }

  // component_probe now points at the list of status strings for
  // this component.  Locate or insert the status.

  string_index_map_t& s = this->status_maps[ component_probe->second ];
  sim_it status_probe = s.find( status );
  if ( status_probe == s.end() )
  {
    size_t new_index = s.size();
    pair< sim_it, bool > p = s.insert( make_pair(status, new_index) );
    status_probe = p.first;
  }

  // return the two indices
  return make_pair( component_probe->second, status_probe->second );
}

map< string, string >
state_flags_backend
::get_map( const vector<size_t>& data )
{
  std::lock_guard< std::mutex > lock( api_lock );

  map< string, string > ret;

  for ( sim_cit component = this->component_map.begin();
        component != this->component_map.end();
        ++component )
  {
    //
    // For each entry (name, index) in the component map,
    // if data[index] is valid, then status_maps[index] is
    // the list of status strings for the index.  Let S = data[index].
    // Then the status string for S is status_maps[index][S].
    //
    // Step one: find status_maps[index].


    size_t component_index = component->second;

    // index zero is always invalid; skip it
    if (component_index == 0) continue;

    if ( data.size() <= component_index )
    {
      // not a problem if this instance of flags (supplying data) doesn't
      // include all the components
      continue;
    }

    // can arise if e.g. there are gaps in the vector we've padded with 0
    size_t status_index = data[ component_index ];
    if (status_index == 0) continue;

    // Get the string/index map at component_index.

    map< size_t, string_index_map_t>::const_iterator s = this->status_maps.find( component_index );
    if (s == this->status_maps.end())
    {
      LOG_ERROR( main_logger, "Trace-flags for " << component->second << " has no status map at index "
                 << component_index );
      return ret;
    }
    const string_index_map_t& s_map = s->second;

    // we now have the status map for the component... iterate until we find
    // the matching status

    sim_cit probe = s_map.end();
    for (sim_cit i = s_map.begin(); (probe == s_map.end() && (i != s_map.end())); ++i)
    {
      if (i->second == status_index) probe = i;
    }
    if (probe == s_map.end())
    {
      LOG_ERROR( main_logger, "Trace-flags for " << component->second << " has no status string for index "
                 << status_index );
      return ret;
    }

    // set the status for this component
    ret[ component->first ] = probe->first;
  }

  return ret;
}

void
state_flag_type
::set_flag( const string& component, const string& status )
{
  if ( (component.find(' ') != string::npos) || (status.find(' ') != string::npos))
  {
    throw runtime_error( "Cannot have spaces in state flag key/value pairs: '"+component+"' ; '"+status+"'" );
  }

  // both members of ci are indices; if ci = (5,7) that means data[5] == 7.
  pair< size_t, size_t > ci = state_flags_backend::get_instance().set_flag( component, status );

  // expand vector if necessary
  // n=3: [0 1 2] ;
  // n' = 6: [0 1 2 3 4 5]  -> ci.first = 5; 5-3+1
  // n=0: []
  // n' = 1: [0] -> ci.first = 0; 0 - 0 +1
  size_t n = this->data.size();
  if ( ci.first >= n )
  {
    for (size_t i=0; i < ci.first-n+1; ++i)
    {
      this->data.push_back( 0 ); // 0 is always invalid
    }
  }

  this->data[ ci.first ] = ci.second;
}

void
state_flag_type
::clear_flag( const string& component )
{
  pair< size_t, size_t> ci = state_flags_backend::get_instance().set_flag( component, "" );
  this->data[ ci.first ] = 0;
}

map< string, string >
state_flag_type
::get_flags() const
{
  return state_flags_backend::get_instance().get_map( this->data );
}

bool
state_flag_type
::operator==( const state_flag_type& rhs ) const
{
  if (this->data.size() != rhs.data.size()) return false;
  for (size_t i=0; i<this->data.size(); ++i)
  {
    if (this->data[i] != rhs.data[i]) return false;
  }
  return true;
}


ostream& operator<<( ostream& os, const state_flag_type& t )
{
  map< string, string > m = t.get_flags();
  size_t n_bars = m.size() - 1;
  for (map<string, string>::const_iterator i = m.begin(); i != m.end(); ++i, n_bars--)
  {
    os << i->first;
    if ( ! i->second.empty() )
    {
      os << ":" << i->second;
    }
    if (n_bars > 0)
    {
      os << "|";
    }
  }
  return os;
}

istream& operator>>( istream& is, state_flag_type& t )
{
  string text;
  if ( is >> text )
  {
    vector<string> tokens;
    kwiver::vital::tokenize( text, tokens, ",|+" );
    for (size_t i=0; i<tokens.size(); ++i)
    {
      string& token = tokens[i];
      kwiver::vital::string_trim(token);
      if (token.empty()) continue;
      size_t c = token.find( ':' );
      if ( c == string::npos )
      {
        // insert the token with just an empty string as the key
        t.set_flag( token, "" );
      }
      else
      {
        t.set_flag( token.substr( 0, c), token.substr( c+1 ));
      }
    }
  }
  return is;
}

template <>
std::ostream&
kwiver_io_base<state_flag_type>::to_stream( std::ostream& os, const state_flag_type& t ) const
{
  os << t;
  return os;
}

template <>
bool
kwiver_io_base<state_flag_type>::from_str( const std::string& s, state_flag_type& t ) const
{
  istringstream iss( s );
  return static_cast<bool>(iss >> t );
}

template <>
bool
kwiver_io_base<state_flag_type>
::read_xml( const TiXmlElement* e, state_flag_type& val ) const
{
  if (! e->GetText()) return false;
  return this->from_str( e->GetText(), val );
}

template <>
void
kwiver_io_base<state_flag_type>
::write_xml( ostream& os, const string& indent, const state_flag_type& val ) const
{
  os << indent << "<" << this->name << "> ";
  this->to_stream( os, val );
  os << " </" << this->name << ">\n";
}

template<>
vector< string>
kwiver_io_base<state_flag_type>
::csv_headers() const
{
  vector< string > h;
  h.push_back( this->name );
  return h;
}

template<>
bool
kwiver_io_base<state_flag_type>
::from_csv( const map< string, string >& header_value_map, state_flag_type& val ) const
{
  const vector< string >& csv_h = this->csv_headers();
  string s("");
  for (size_t i=0; i<csv_h.size(); ++i)
  {
    map< string, string >::const_iterator p = header_value_map.find( csv_h[i] );
    if ( p == header_value_map.end() ) return false;
    s += p->second + " ";
  }
  return this->from_str( s, val );
}

template<>
ostream&
kwiver_io_base<state_flag_type>
::to_csv( ostream& os, const state_flag_type& val ) const
{
  return this->to_stream( os, val );
}


dt::context dt::utility::state_flags::c( dt::utility::state_flags::get_context_name(),
                                         dt::utility::state_flags::get_context_description() );

} // ...track_oracle
} // ...kwiver
