/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#include "kwiver_io_helpers.h"

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::ostream;
using std::streamsize;
using std::string;
using std::istringstream;
using std::ostringstream;
using std::vector;
using std::pair;

namespace kwiver {
namespace track_oracle {

ostream& kwiver_write_highprecision( ostream& os, double d, streamsize new_prec )
{
  streamsize old_prec = os.precision( new_prec );
  os << d;
  os.precision( old_prec );
  return os;
}

//
// vgl_point_2d<double>
//

bool kwiver_read( const string& s, vgl_point_2d<double>& d )
{
  double x, y;
  istringstream iss( s );
  if ( ! (iss >> x >> y ))
  {
    LOG_ERROR( main_logger, "Couldn't parse vgl_point_2d<double> from '" << s << "'" );
    return false;
  }
  d.x() = x;
  d.y() = y;
  return true;
}

ostream& kwiver_write( ostream& os, const vgl_point_2d<double>& d, const string& sep )
{
  os << d.x() << sep << d.y();
  return os;
}

vector<string> kwiver_csv_header_pair( const string& n, const string& p1, const string& p2 )
{
  vector<string> r;
  r.push_back( n+p1 );
  r.push_back( n+p2 );
  return r;
}


//
// vgl_box_2d<double>
//

bool kwiver_read( const string& s, vgl_box_2d<double>& d )
{
  double min_x, min_y, max_x, max_y;
  istringstream iss( s );
  if ( ! ( iss >> min_x >> min_y >> max_x >> max_y ))
  {
    LOG_ERROR( main_logger, "Couldn't parse vgl_box_2d<double> from '" << s << "'" );
    return false;
  }
  d.set_min_x( min_x );
  d.set_min_y( min_y );
  d.set_max_x( max_x );
  d.set_max_y( max_y );
  return true;
}

ostream& kwiver_write( ostream& os, const vgl_box_2d<double>& d, const string& sep )
{
  os << d.min_x() << sep << d.min_y() << sep << d.max_x() << sep << d.max_y();
  return os;
}

vector<string> kwiver_box_2d_headers( const string& s )
{
  vector<string> r;
  r.push_back( s + "_ul_x" );
  r.push_back( s + "_ul_y" );
  r.push_back( s + "_lr_x" );
  r.push_back( s + "_lr_y" );
  return r;
}

//
// vgl_point_3d<double>
//

bool kwiver_read( const string& s, vgl_point_3d<double>& d )
{
  double x, y, z;
  istringstream iss( s );
  if ( ! (iss >> x >> y >> z))
  {
    LOG_ERROR( main_logger, "Couldn't parse vgl_point_3d<double> from '" << s << "'" );
    return false;
  }
  d.x() = x;
  d.y() = y;
  d.z() = z;
  return true;
}

ostream& kwiver_write( ostream& os, const vgl_point_3d<double>& d, const string& sep )
{
  os << d.x() << sep << d.y() << sep << d.z();
  return os;
}

vector<string> kwiver_point_3d_headers( const string& n )
{
  vector<string> r;
  r.push_back( n+"_x" );
  r.push_back( n+"_y" );
  r.push_back( n+"_z" );
  return r;
}

//
// vital::timestamp
//

pair<string, string >
kwiver_ts_to_strings( const vital::timestamp& ts )
{
  string f_str( "none" ), t_str( "none" );
  if (ts.has_valid_frame())
  {
    ostringstream oss;
    oss << ts.get_frame();
    f_str = oss.str();
  }
  if (ts.has_valid_time())
  {
    ostringstream oss;
    kwiver_write_highprecision( oss, ts.get_time_seconds() );
    t_str = oss.str();
  }
  return make_pair( f_str, t_str );
}


bool
kwiver_ts_string_read( const string& frame_str,
                       const string& time_str,
                       vital::timestamp& t )
{
  unsigned fn( static_cast<unsigned int>( -1 ));
  if ( frame_str == "none" )
  {
    // leave as invalid
  }
  else
  {
    istringstream iss( frame_str );
    if ( ! ( iss >> fn ))
    {
      LOG_ERROR( main_logger, "Timestamp: couldn't parse '" << frame_str << "' as a frame number; setting invalid" );
      return false;
    }
  }
  t.set_frame( fn );

  double ts( -1e300 );
  if ( time_str == "none" )
  {
    // leave as invalid
  }
  else
  {
    istringstream iss( time_str );
    if ( ! ( iss >> ts ))
    {
      LOG_ERROR( main_logger, "Timestamp::from_stream: couldn't parse '" << time_str << "' as a time; setting invalid" );
      return false;
    }
  }
  t.set_time_usec( static_cast< vital::time_us_t >( ts * 1.0e6) );
  return true;
}

bool kwiver_read( const string& s, vital::timestamp& ts)
{
  size_t c = s.find(':');
  if ( c == string::npos )
  {
    LOG_ERROR( main_logger, "Improperly formatted timestamp string '" << s << "'" );
    return false;
  }

  return kwiver_ts_string_read( s.substr( 0, c-1 ), s.substr( c ), ts );
}

ostream& kwiver_write( ostream& os, const vital::timestamp& ts )
{
  pair< string, string > ts_strings = kwiver_ts_to_strings( ts );
  os << ts_strings.first << ":" << ts_strings.second;
  return os;
}

bool kwiver_read( const std::string& s, kpf_cset_type& cset )
{
  return false;
}

ostream& kwiver_write( std::ostream& os, const kpf_cset_type& cset )
{
  return os;
}

bool kwiver_read( const std::string& s, kpf_cset_sys_type& cset )
{
  return false;
}

ostream& kwiver_write( std::ostream& os, const kpf_cset_sys_type& cset )
{
  return os;
}

bool kwiver_read( const std::string& s, kpf_cset_s2i_type& cset )
{
  return false;
}

ostream& kwiver_write( std::ostream& os, const kpf_cset_s2i_type& cset )
{
  return os;
}

} // ...track_oracle
} // ...kwiver
