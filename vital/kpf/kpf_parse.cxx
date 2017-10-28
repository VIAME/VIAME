#include "kpf_parse.h"

#include <vital/util/tokenize.h>
#include <vector>
#include <stdexcept>

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
    //    LOG_ERROR( main_logger, "Bad packet style '" << tag_str << "'" );
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
    if (packet_header_parser( tokens[ index ],
                              p.header,
                              true ))
    {
      // uh-oh, we couldn't parse it; build up an 'unparsed' key-value packet
      // until we get a parse
      ++index;
      pair< bool, size_t > next = packet_payload_parser( index, tokens, p );
      if (! next.first )
      {
        // This indicates a malformed packet error
        return false;
      }
      index = next.second;

      packet_buffer.insert( make_pair( p.header, p ));
    }
    else
    {
      // uh-oh, we couldn't recognize the header-- build up an 'unparsed' key-value
      // packet
      string unparsed_txt = tokens[index];
      LOG_DEBUG( main_logger, "starting unparsed with '" << unparsed_txt << "'" );
      // keep trying until we either parse a header or run out of tokens
      ++index;
      bool keep_going = (index < n);
      while (keep_going)
      {
        if (packet_header_parser( tokens[index], p.header, true ))
        {
          // we found a parsable header-- all done
          LOG_DEBUG( main_logger, "Found a parsable header at index " << index << ": '" << tokens[index] << "'" );
          keep_going = false;
        }
        else
        {
          unparsed_txt += " "+tokens[index++];
          keep_going = (index < n);
        }
      }

      packet_header_t unparsed_header( packet_style::KV );
      packet_t unparsed( unparsed_header );
      LOG_DEBUG( main_logger, "Completing unparsed '" << unparsed_txt << "' ; next index " << index << " of " << n);
      new (&unparsed.kv) canonical::kv_t( "unparsed", unparsed_txt );
      packet_buffer.insert( make_pair( unparsed.header, unparsed ));
    }
  }
  return true;
}


} // ...anon

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

  bool rc = packet_parser( tokens, local_packet_buffer );
  return rc;
}

//
//
//

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

kpf_reader_t
::kpf_reader_t( kpf_parser_base_t& p )
  : packet_buffer( packet_header_cmp ), parser(p)
{
  this->reader_status = this->parser.get_status();
}

//
//
//

kpf_reader_t
::operator bool() const
{
  return this->reader_status;
}

bool
kpf_reader_t
::next()
{
  this->verify_reader_status();
  return this->reader_status;
}

vector< string >
kpf_reader_t
::get_meta_packets() const
{
  return this->meta_buffer;
}


bool
kpf_reader_t
::parse_next_line()
{
  //
  // loop over each line (throwing meta packets into the
  // meta-buffer) until either (a) non-meta packets have
  // been added or (b) getline() fails
  //

  bool non_meta_packets_added = false;
  packet_header_t meta_packet_h( packet_style::META );
  while ( ! non_meta_packets_added )
  {
    packet_buffer_t local_packet_buffer( packet_header_cmp );
    bool rc = this->parser.parse_next_record( local_packet_buffer );
    if (! rc )
    {
      break;
    }
    // pull any meta packets out
    packet_buffer_cit p;
    while ( (p = local_packet_buffer.find( meta_packet_h ))
           != local_packet_buffer.end() )
    {
      this->meta_buffer.push_back( p->second.meta.txt );
      local_packet_buffer.erase( p );
    }

    // if we have any packets left, push them into the
    // global packet buffer and set the flag
    if (! local_packet_buffer.empty())
    {
      this->packet_buffer.insert( local_packet_buffer.begin(), local_packet_buffer.end() );
      non_meta_packets_added = true;
    }
  }
  return non_meta_packets_added;
}

bool
kpf_reader_t
::verify_reader_status()
{
  if (! this->reader_status )
  {
    return false;
  }

  //
  // If the buffer is empty, read a 'record' (a line)
  // from the stream.
  //

  if (this->packet_buffer.empty())
  {
    if ( ! this->parse_next_line() )
    {
      this->reader_status = false;
      return false;
    }
  }
  return true;
}

pair< bool, packet_t >
kpf_reader_t
::transfer_kv_packet_from_buffer( const string& key )
{
  if (! this->verify_reader_status() )
  {
    return make_pair( false, packet_t() );
  }

  //
  // Look for a packet in the buffer which is (a) a kv packet,
  // and (b) its key value matches the parameter
  //

  auto probe =
    std::find_if( this->packet_buffer.cbegin(),
                  this->packet_buffer.cend(),
                  [key]( const std::pair< packet_header_t, packet_t>& p) -> bool {
                    return ((p.first.style == packet_style::KV) &&
                            (p.second.kv.key == key )); });

  if (probe == this->packet_buffer.end())
  {
    return make_pair( false, packet_t() );
  }

  //
  // remove the packet from the buffer and set the reader; we're done
  //

  auto ret = make_pair( true, probe->second );
  this->packet_buffer.erase( probe );
  return ret;
}


pair< bool, packet_t >
kpf_reader_t
::transfer_packet_from_buffer( const packet_header_t& h )
{
  if (! this->verify_reader_status() )
  {
    return make_pair( false, packet_t() );
  }

  //
  // if the head is invalid (i.e. the null reader) we're done
  //

  if (h.style == packet_style::INVALID)
  {
    return make_pair( true, packet_t() );
  }

  //
  // does the packet buffer contain what this reader is looking for?
  // if not, return false
  //

  auto probe = this->packet_buffer.find( h );
  if (probe == this->packet_buffer.end())
  {
    this->reader_status = false;
    return make_pair( false, packet_t() );
  }

  //
  // remove the packet from the buffer and set the reader; we're done
  //

  auto ret = make_pair( true, probe->second );
  this->packet_buffer.erase( probe );
  return ret;
}

bool
kpf_reader_t
::process_reader( packet_bounce_t& b )
{
  auto probe = this->transfer_packet_from_buffer( b.my_header() );
  if (probe.first)
  {
    b.set_from_buffer( probe.second );
  }
  return probe.first;
}

bool
kpf_reader_t
::process( packet_bounce_t& b )
{
  if ( this->reader_status )
  {
    bool okay = this->process_reader( b );
    this->reader_status = okay && this->reader_status;
  }
  return this->reader_status;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            packet_bounce_t& b )
{
  t.process( b );
  return t;
}

bool
kpf_reader_t
::process( kpf_io_adapter_base& io )
{
  return this->process( io.text_reader );
}

kpf_reader_t& operator>>( kpf_reader_t& t,
                           kpf_io_adapter_base& io )
{
  return t >> io.text_reader;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::bbox_t >& r )
{
  t.process( r.box_adapter.set_domain( r.domain ) );
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::poly_t >& r )
{
  t.process( r.poly_adapter.set_domain( r.domain ) );
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::id_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::ID, r.domain ));
  if (probe.first)
  {
    r.id_ref = probe.second.id.d;
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::timestamp_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::TS, r.domain ));
  if (probe.first)
  {
    switch (r.which)
    {
      case reader< canonical::timestamp_t >::to_int:
        r.int_ts = static_cast<int>( probe.second.timestamp.d );
        break;
      case reader< canonical::timestamp_t >::to_unsigned:
        r.unsigned_ts = static_cast<unsigned>( probe.second.timestamp.d );
        break;
      case reader< canonical::timestamp_t >::to_double:
        r.double_ts = probe.second.timestamp.d;
      break;
    }
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::kv_t >& r )
{
  auto probe = t.transfer_kv_packet_from_buffer( r.key );
  if (probe.first)
  {
    r.val = probe.second.kv.val;
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::conf_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::CONF, r.domain ));
  if (probe.first)
  {
    r.conf = probe.second.conf.d;
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::meta_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::META ));
  if (probe.first)
  {
    r.txt = probe.second.meta.txt;
  }
  return t;
}

record_text_writer&
operator<<( record_text_writer& w, const private_endl_t& )
{
  w.s << std::endl;
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::id_t >& io)
{
  w.s << "id" << io.domain << ": " << io.id.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::bbox_t >& io)
{
  w.s << "g" << io.domain << ": " << io.box.x1 << " " << io.box.y1 << " " << io.box.x2 << " " << io.box.y2 << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::timestamp_t >& io)
{
  w.s << "ts" << io.domain << ": " << io.ts.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::kv_t >& io)
{
  w.s << "kv: " << io.kv.key << " " << io.kv.val << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::conf_t >& io)
{
  w.s << "conf" << io.domain << ": " << io.conf.d << " ";
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::poly_t >& io)
{
  w.s << "poly" << io.domain << ": " << io.poly.xy.size() << " ";
  for (const auto& p : io.poly.xy )
  {
    w.s << p.first << " " << p.second << " ";
  }
  return w;
}

record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::meta_t >& io)
{
  w.s << "meta: " << io.meta.txt;
  return w;
}


} // ...kpf
} // ...vital
} // ...kwiver
