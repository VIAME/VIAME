// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
