/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief KPF reader.
 *
 * This reads parsed KPF packets out of the packet buffer into user-space.
 *
 */

#include "kpf_reader.h"

#include <vector>
#include <stdexcept>
#include <algorithm>

#include <arrows/kpf/yaml/kpf_canonical_io_adapter.h>

using std::istream;
using std::istringstream;
using std::string;
using std::vector;
using std::pair;
using std::make_pair;

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
::transfer_kv_packet_from_buffer( const string& key, bool set_bad_if_missing )
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
    if (set_bad_if_missing)
    {
      this->reader_status = false;
    }
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
::transfer_packet_from_buffer( const packet_header_t& h, bool set_bad_if_missing )
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

  packet_buffer_cit probe = this->packet_buffer.end();
  if (h.domain == packet_header_t::ANY_DOMAIN)
  {
    for (packet_buffer_cit any_probe = this->packet_buffer.begin();
         any_probe != this->packet_buffer.end();
         ++any_probe )
    {
      if (any_probe->first.style == h.style)
      {
        probe = any_probe;
        break;
      }
    }
  }
  else
  {
    probe = this->packet_buffer.find( h );
  }
  if (probe == this->packet_buffer.end())
  {
    if (set_bad_if_missing)
    {
      this->reader_status = false;
    }
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
  // fail if missing; this is (I think) only called by readers
  auto probe = this->transfer_packet_from_buffer( b.my_header(), true );
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
            const reader< canonical::activity_t >& r )
{
  t.process( r.act_adapter.set_domain( r.domain ) );
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::id_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::ID, r.domain ), true );
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
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::TS, r.domain ), true );
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
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::CONF, r.domain ), true);
  if (probe.first)
  {
    r.conf = probe.second.conf.d;
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::cset_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::CSET, r.domain ), true);
  if (probe.first)
  {
    r.cset = *probe.second.cset;
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::meta_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::META ), true);
  if (probe.first)
  {
    r.txt = probe.second.meta.txt;
  }
  return t;
}

kpf_reader_t&
operator>>( kpf_reader_t& t,
            const reader< canonical::timestamp_range_t >& r )
{
  auto probe = t.transfer_packet_from_buffer( packet_header_t( packet_style::TSR ), true);
  if (probe.first)
  {
    switch (r.which)
    {
      case reader< canonical::timestamp_range_t >::to_int:
        r.int_ts = make_pair( static_cast< int >( probe.second.timestamp_range.start ),
                              static_cast< int >( probe.second.timestamp_range.stop ) );
        break;
      case reader< canonical::timestamp_range_t >::to_unsigned:
        r.unsigned_ts = make_pair( static_cast< unsigned>( probe.second.timestamp_range.start ),
                                   static_cast< unsigned>( probe.second.timestamp_range.stop ) );
        break;
      case reader< canonical::timestamp_range_t >::to_double:
        r.double_ts = make_pair( probe.second.timestamp_range.start, probe.second.timestamp_range.stop );
      break;
    }
  }
  return t;
}


} // ...kpf
} // ...vital
} // ...kwiver
