#include "kpf_parse_utils.h"

#include <utility>
#include <cctype>


#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::string;

namespace kwiver {
namespace vital {
namespace kpf {

header_parse_t
parse_header(const string& s )
{
  auto bad_parse = std::make_tuple( false, string(), packet_header_t::NO_DOMAIN );

  //
  // packet headers are always of the form [a-Z]+[0-9]*:
  //

  // parse from the rear

  if (s.empty())
  {
    LOG_ERROR( main_logger, "Packet header: tying to parse empty string?" );
    return bad_parse;
  }

  size_t i=s.size()-1;
  if (s[i--] != ':')
  {
    LOG_ERROR( main_logger, "Packet header '" << s << "': no trailing colon" );
    return bad_parse;
  }

  //
  // look for the domain
  //
  int domain = packet_header_t::NO_DOMAIN;

  size_t domain_end(i);
  while ((i != 0) && std::isdigit( s[i] ))
  {
    --i;
  }
  // if we've backed up to the front of the string and it's still digits,
  // which is ill-formed
  if ((i == 0) && std::isdigit( s[i] ))
  {
    LOG_ERROR( main_logger, "Packet header '" << s << "': no packet style");
    return bad_parse;
  }
  size_t domain_start = i+1;
  if (domain_start <= domain_end)
  {
    char* end;
    const int base = 10;
    domain = std::strtol( s.substr( domain_start, domain_end - domain_start+1 ).c_str(),
                          &end,
                          base );
  }

  //
  // packet style is everything else
  //

  string style = s.substr( 0, i+1 );

  return std::make_tuple( true, style, domain);
}


} // ...kpf
} // ...vital
} // ...kwiver
