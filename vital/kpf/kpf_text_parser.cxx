#include "kpf_text_parser.h"

#include <vital/util/tokenize.h>
#include <string>
#include <vector>

using std::istream;
using std::string;
using std::vector;

namespace kwiver {
namespace vital {
namespace kpf {

//
// text parser
//

kpf_text_parser_t
::kpf_text_parser_t( istream& is ): input_stream( is )
{
}

bool
kpf_text_parser_t
::get_status() const
{
  return static_cast<bool>( this->input_stream );
}

bool
kpf_text_parser_t
::parse_next_record( packet_buffer_t& local_packet_buffer )
{
  string s;
  if (! std::getline( this->input_stream, s ))
  {
    return false;
  }
  vector< string > tokens;
  ::kwiver::vital::tokenize( s, tokens, " ", true );

  return packet_parser( tokens, local_packet_buffer );
}


} // ...kpf
} // ...vital
} // ...kwiver

