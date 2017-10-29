#include "kpf_yaml_parser.h"

#include <string>
#include <vector>
#include <cctype>

#include <vital/util/tokenize.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );


using std::istream;
using std::string;
using std::vector;
using std::isdigit;

namespace kwiver {
namespace vital {
namespace kpf {

kpf_yaml_parser_t
::kpf_yaml_parser_t( istream& is )
{
  this->root = YAML::Load( is );
  this->current_record = this->root.begin();
}

bool
kpf_yaml_parser_t
::get_status() const
{
  return this->current_record != this->root.end();
}

bool
kpf_yaml_parser_t
::parse_next_record( packet_buffer_t& local_packet_buffer )
{
  if (this->current_record == this->root.end())
  {
    return false;
  }
  const YAML::Node& n = *(this->current_record++);
  vector< string > tokens;

  if (! n.IsMap())
  {
    LOG_ERROR( main_logger, "YAML: type " << n.Type() << " not a map? " << n.as<string>() );
    return false;
  }

  // start parsing the map entries; each must be a packet header
  for (auto it=n.begin(); it != n.end(); ++it)
  {
    string s = it->first.as<string>();
    // if the last character is not a digit, it's a key
    if ( (!s.empty()) && ( isdigit( static_cast< unsigned char>( s.back() ))))
    {
      // it's a KPF packet
      tokens.push_back( s+":" );
    }
    else
    {
      // it's a KPF 'kv:' key-value pair
      tokens.push_back( "kv:" );
      tokens.push_back( s );
    }
    vector< string > sub_tokens;
    ::kwiver::vital::tokenize( it->second.as<string>(), sub_tokens, " ", true );
    tokens.insert( tokens.end(), sub_tokens.begin(), sub_tokens.end() );
  }

  return packet_parser( tokens, local_packet_buffer );
}

} // ...kpf
} // ...vital
} // ...kwiver

