/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief kwiver I/O for CSET support types; compiles but do nothing for now
 */

#include "kpf_cset_io.h"

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace kwiver {
namespace track_oracle {

using tag2index_t = std::map< std::string, std::size_t >;
using index2val_t = std::map< std::size_t, double >;

std::ostream&
kwiver_io_base<tag2index_t>
::to_stream( std::ostream& os, const tag2index_t& val ) const
{
  return os;
}

bool
kwiver_io_base<tag2index_t>
::from_str( const std::string& s,
            tag2index_t& val ) const
{
  return false;
}

bool
kwiver_io_base<tag2index_t>
::read_xml( const TiXmlElement* e,  tag2index_t& val ) const
{
  return false;
}

void
kwiver_io_base<tag2index_t>
::write_xml( std::ostream& os, const std::string& indent, const tag2index_t& val ) const
{
}

vector<string>
kwiver_io_base<tag2index_t>
::csv_headers() const
{
  return vector<string>();
}

bool
kwiver_io_base<tag2index_t>
::from_csv( const std::map< std::string, std::string >& header_value_map, tag2index_t& val ) const
{
  return false;
}

std::ostream&
kwiver_io_base<tag2index_t>
::to_csv( std::ostream& os, const tag2index_t& val ) const
{
  return os;
}

std::ostream&
kwiver_io_base<index2val_t>
::to_stream( std::ostream& os, const index2val_t& val ) const
{
  return os;
}

bool
kwiver_io_base<index2val_t>
::from_str( const std::string& s, index2val_t& val ) const
{
  return false;
}

bool
kwiver_io_base<index2val_t>
::read_xml( const TiXmlElement* e, index2val_t& val ) const
{
  return false;
}

void
kwiver_io_base<index2val_t>
::write_xml( std::ostream& os, const std::string& indent, const index2val_t& val ) const
{
}

std::vector< std::string >
kwiver_io_base<index2val_t>
::csv_headers() const
{
  return vector<string>();
}

bool
kwiver_io_base<index2val_t>
::from_csv( const std::map< std::string, std::string >& header_value_map, index2val_t& val ) const
{
  return false;
}

std::ostream&
kwiver_io_base<index2val_t>
::to_csv( std::ostream& os, const index2val_t& val ) const
{
  return os;
}

} // ...track_oracle
} // ...kwiver


