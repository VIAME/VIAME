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
 * \brief Template instances of kwiver I/O for the CSET support types
 */

#ifndef INCL_KPF_CSET_IO_H
#define INCL_KPF_CSET_IO_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/kpf_utils/kpf_utils_export.h>
#include <track_oracle/core/kwiver_io_base.h>

#include <map>
#include <string>
#include <iostream>

namespace kwiver {
namespace track_oracle {

using tag2index_t = std::map< std::string, std::size_t >;
using index2val_t = std::map< std::size_t, double >;

template<>
class KPF_UTILS_EXPORT kwiver_io_base< tag2index_t >
{
  virtual ~kwiver_io_base() {}
  virtual std::ostream& to_stream( std::ostream& os, const tag2index_t& val ) const;
  virtual bool from_str( const std::string& s,
                         tag2index_t& val ) const;
  virtual bool read_xml( const TiXmlElement* e,
                         tag2index_t& val ) const;
  virtual void write_xml( std::ostream& os,
                          const std::string& indent,
                          const tag2index_t& val ) const;
  virtual std::vector< std::string > csv_headers() const;
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, tag2index_t& val ) const;
  virtual std::ostream& to_csv( std::ostream& os, const tag2index_t& val ) const;
};

template<>
class KPF_UTILS_EXPORT kwiver_io_base< index2val_t >
{
  virtual ~kwiver_io_base() {}
  virtual std::ostream& to_stream( std::ostream& os, const index2val_t& val ) const;
  virtual bool from_str( const std::string& s,
                         index2val_t& val ) const;
  virtual bool read_xml( const TiXmlElement* e,
                         index2val_t& val ) const;
  virtual void write_xml( std::ostream& os,
                          const std::string& indent,
                          const index2val_t& val ) const;
  virtual std::vector< std::string > csv_headers() const;
  virtual bool from_csv( const std::map< std::string, std::string >& header_value_map, index2val_t& val ) const;
  virtual std::ostream& to_csv( std::ostream& os, const index2val_t& val ) const;
};

} // ...track_oracle
} // ...kwiver

#endif
