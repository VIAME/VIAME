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
using std::make_pair;

kwiver::vital::kpf::private_endl_t kwiver::vital::kpf::record_text_writer::endl;

namespace { // anon
using namespace kwiver::vital::kpf;

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

bool
packet_header_parser( const string& s, packet_header_t& packet_header, bool expect_colon )
{
  //
  // try to parse the header into a flag / tag / domain
  //

  header_parse_t h = parse_header( s, expect_colon );
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
                                p.header,
                                true ))
    {
      return false;
    }

    pair< bool, size_t > next = packet_payload_parser( index, tokens, p );
    if (! next.first )
    {
      return false;
    }
    index = next.second;

    packet_buffer.insert( make_pair( p.header, p ));
  }
  return true;
}


} // ...anon

namespace kwiver {
namespace vital {
namespace kpf {

text_reader_t
::text_reader_t()
  : is_set( false )
{
}

text_reader_t
::text_reader_t( const string& s )
  : is_set( false )
{
  this->init( s );
}

text_reader_t
::text_reader_t( const packet_header_t& h )
  : is_set( false ), header( h )
{
}

void
text_reader_t
::init( const string& s )
{
  if (! packet_header_parser( s, this->header, false ))
  {
    LOG_ERROR( main_logger, "Couldn't create a reader for packets of type '" << s << "'");
    return;
  }
}

void text_reader_t
::init( const packet_header_t& h )
{
  this->header = h;
}

void
text_reader_t
::set_from_buffer( const packet_t& p )
{
  this->is_set = true;
  this->packet = p;
}

packet_header_t
text_reader_t
::my_header() const
{
  return this->header;
}

text_reader_t&
text_reader_t
::set_domain( int d )
{
  this->is_set = false;
  this->header.domain = d;
  return *this;
}

pair< bool, packet_t >
text_reader_t
::get_packet()
{
  bool f = this->is_set;
  this->is_set = false;
  return make_pair( f, this->packet );
}

text_parser_t
::text_parser_t( istream& is )
  : packet_buffer( packet_header_cmp), input_stream(is)
{
  this->reader_status = static_cast<bool>( is );
}


text_parser_t
::operator bool() const
{
  return this->reader_status;
}


bool
text_parser_t
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
  ::kwiver::vital::tokenize( s, tokens, " ", true );

  //
  // pass the tokens off to the packet parser
  //

  return packet_parser( tokens, this->packet_buffer );
}

bool
text_parser_t
::process_reader( text_reader_t& b )
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
  LOG_INFO( main_logger, "Reader looking for style " << style2str(h.style) << " domain " << h.domain );

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

bool
text_parser_t
::process( text_reader_t& b )
{
  if ( this->reader_status )
  {
    bool okay = this->process_reader( b );
    this->reader_status = okay && this->reader_status;
  }
  return this->reader_status;
}

text_parser_t&
operator>>( text_parser_t& t,
            text_reader_t& b )
{
  t.process( b );
  return t;
}

bool
text_parser_t
::process( kpf_io_adapter_base& io )
{
  return this->process( io.text_reader );
}

text_parser_t& operator>>( text_parser_t& t,
                           kpf_io_adapter_base& io )
{
  return t >> io.text_reader;
}

record_text_writer&
operator<<( record_text_writer& w, const private_endl_t& )
{
  w.s << std::endl;
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const io< canonical::id_t >& io)
{
  w.s << "id" << io.domain << ": " << io.id.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const io< canonical::bbox_t >& io)
{
  w.s << "g" << io.domain << ": " << io.box.x1 << " " << io.box.y1 << " " << io.box.x2 << " " << io.box.y2 << " ";
  return w;
}

} // ...kpf
} // ...vital
} // ...kwiver
