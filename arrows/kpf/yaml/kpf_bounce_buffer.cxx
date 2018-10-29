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
 * \brief KPF bounce buffer implementation.
 *
 */

#include "kpf_bounce_buffer.h"
#include <arrows/kpf/yaml/kpf_parse_utils.h>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "arrows.kpf.bounce_buffer" ) );

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
  this->header = packet_header_parser( s );
  if (! (this->header.style == packet_style::INVALID))
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
