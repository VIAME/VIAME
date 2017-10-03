#include "kpf_parse.h"

#include <vital/util/tokenize.h>
#include <vector>

#include <vital/kpf/kpf_parse_utils.h>

#include <vital/logger/logger.h>

using std::istream;
using std::istringstream;
using std::string;
using std::vector;
using std::pair;

namespace { // anon
using namespace kwiver::vital::kpf;

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

bool
packet_header_parser( const string& s, packet_header_t& packet_header )
{
  //
  // try to parse the header into a flag / tag / domain
  //

  header_parse_t h = parse_header( s );
  if (! std::get<0>(h) )
  {
    return false;
  }

  string tag_str( std::get<1>(h) );
  packet_style style = str2style( tag_str );
  if ( style == packet_style::INVALID )
  {
    return false;
  }

  int domain( std::get<2>(h) );
  packet_header = packet_header_t( style, domain );
  return true;
}

bool
packet_parser( const vector<string>& tokens,
               packet_buffer_t& packet_buffer )
{
  size_t index(0), n( tokens.size() );

  while ( index < n )
  {
    packet_t p;
    if (! packet_header_parser( tokens[ index++ ],
                                p.header ))
    {
      return false;
    }

    pair< bool, size_t > next = packet_payload_parser( index, tokens, p );
    if (! next.first )
    {
      return false;
    }
    index = next.second;

    packet_buffer.insert( std::make_pair( p.header, p ));
  }
  return true;
}


} // ...anon

namespace kwiver {
namespace vital {
namespace kpf {

text_record_reader_t
::text_record_reader_t( istream& is )
  : packet_buffer( packet_header_cmp), input_stream(is)
{
  this->reader_status = static_cast<bool>( is );
}


text_record_reader_t
::operator bool() const
{
  return this->reader_status;
}


bool
text_record_reader_t
::parse_next_line()
{
  //
  // Read the next line of text from the stream and
  // tokenize it
  //

  string s;
  if (! std::getline( this->input_stream, s ))
  {
    return false;
  }
  vector< string > tokens;
  ::kwiver::vital::tokenize( s, tokens );

  //
  // pass the tokens off to the packet parser
  //

  return packet_parser( tokens, this->packet_buffer );
}

bool
text_record_reader_t
::process_reader( text_record_reader_buffer_base_t& b )
{
  //
  // If the buffer is empty, read a 'record' (a line)
  // from the stream.
  //

  if (this->packet_buffer.empty())
  {
    if ( ! this->parse_next_line() )
    {
      return false;
    }
  }

  //
  // what type of packet is this reader looking for?
  //

  packet_header_t h = b.my_header();

  //
  // if the head is invalid (i.e. the null reader) we're done
  //

  if (h.style == packet_style::INVALID)
  {
    return true;
  }

  //
  // does the packet buffer contain what this reader is looking for?
  // if not, return false
  //

  auto probe = this->packet_buffer.find( h );
  if (probe == this->packet_buffer.end())
  {
    return false;
  }

  //
  // remove the packet from the buffer and set the reader; we're done
  //

  b.set_from_buffer( probe->second );
  this->packet_buffer.erase( probe );
  return true;
}

text_record_reader_t&
operator>>( text_record_reader_t& t,
            text_record_reader_buffer_base_t& b )
{
  if (t.reader_status)
  {
    bool okay = t.process_reader( b );
    t.reader_status = okay && t.reader_status;
  }
  return t;
}

} // ...kpf
} // ...vital
} // ...kwiver
