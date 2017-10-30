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

namespace { // anon

void visit( const YAML::Node& n, vector< string >& tokens )
{
  switch (n.Type())
  {
  case YAML::NodeType::Scalar:
    ::kwiver::vital::tokenize( n.as<string>(), tokens, " ", true );
    break;

  case YAML::NodeType::Sequence:
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      visit( *it, tokens );
    }
    break;

  case YAML::NodeType::Map:
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      visit( it->first, tokens );
      visit( it->second, tokens );
    }
    break;

  case YAML::NodeType::Null:  // fall-through
  case YAML::NodeType::Undefined:
    // do nothing
    break;
  }
}

} // ...anon

namespace kwiver {
namespace vital {
namespace kpf {

kpf_yaml_parser_t
::kpf_yaml_parser_t( istream& is )
{
  try
  {
    this->root = YAML::Load( is );
  }
  // This seems not to work on OSX as of 30oct2017
  // see https://stackoverflow.com/questions/21737201/problems-throwing-and-catching-exceptions-on-os-x-with-fno-rtti
  catch (const YAML::ParserException& e )
  {
    LOG_ERROR( main_logger, "Exception parsing KPF YAML: " << e.what() );
    this->root = YAML::Node();
  }
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
    // if the last character is not a digit, it's a key (or a meta)
    bool is_meta = (s == "meta");
    bool ends_in_digit =  (!s.empty()) && ( isdigit( static_cast< unsigned char>( s.back() )));
    if (is_meta || ends_in_digit)
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
    visit( it->second, tokens );
  }

  return packet_parser( tokens, local_packet_buffer );
}

} // ...kpf
} // ...vital
} // ...kwiver

