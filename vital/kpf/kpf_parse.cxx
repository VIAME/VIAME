#include "kpf_parse.h"

#include <vector>
#include <stdexcept>

#include <vital/logger/logger.h>
#include <vital/kpf/kpf_canonical_io_adapter.h>

using std::istream;
using std::istringstream;
using std::string;
using std::vector;
using std::pair;
using std::make_pair;

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

namespace kwiver {
namespace vital {
namespace kpf {


kpf_reader_t
::kpf_reader_t( kpf_parser_base_t& p )
  : parser(p)
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
    packet_buffer_t local_packet_buffer;
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
::process( kpf_canonical_io_adapter_base& io )
{
  return this->process( io.packet_bounce );
}

kpf_reader_t& operator>>( kpf_reader_t& t,
                          kpf_canonical_io_adapter_base& io )
{
  return t >> io.packet_bounce;
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


} // ...kpf
} // ...vital
} // ...kwiver
