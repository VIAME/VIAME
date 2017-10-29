#include "kpf_bounce_buffer.h"
#include <vital/kpf/kpf_parse_utils.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;
using std::pair;
using std::make_pair;

namespace kwiver {
namespace vital {
namespace kpf {

packet_bounce_t
::packet_bounce_t()
  : is_set( false )
{
}

packet_bounce_t
::packet_bounce_t( const string& s )
  : is_set( false )
{
  this->init( s );
}

packet_bounce_t
::packet_bounce_t( const packet_header_t& h )
  : is_set( false ), header( h )
{
}

void
packet_bounce_t
::init( const string& s )
{
  if (! packet_header_parser( s, this->header, false ))
  {
    LOG_ERROR( main_logger, "Couldn't create a reader for packets of type '" << s << "'");
    return;
  }
}

void packet_bounce_t
::init( const packet_header_t& h )
{
  this->header = h;
}

void
packet_bounce_t
::set_from_buffer( const packet_t& p )
{
  this->is_set = true;
  this->packet = p;
}

packet_header_t
packet_bounce_t
::my_header() const
{
  return this->header;
}

packet_bounce_t&
packet_bounce_t
::set_domain( int d )
{
  this->is_set = false;
  this->header.domain = d;
  return *this;
}

pair< bool, packet_t >
packet_bounce_t
::get_packet()
{
  bool f = this->is_set;
  this->is_set = false;
  return make_pair( f, this->packet );
}

} // ...kpf
} // ...vital
} // ...kwiver
