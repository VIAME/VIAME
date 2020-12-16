// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief kwiver I/O for CSET support types; compiles but do nothing for now
 */

#include "kpf_cset_io.h"

#include <vector>
#include <string>

#include <vital/vital_config.h>

using std::vector;
using std::string;

namespace kwiver {
namespace track_oracle {

using tag2index_t = std::map< std::string, std::size_t >;
using index2val_t = std::map< std::size_t, double >;

std::ostream&
kwiver_io_base<tag2index_t>
::to_stream( std::ostream& os, VITAL_UNUSED const tag2index_t& val ) const
{
  return os;
}

bool
kwiver_io_base<tag2index_t>
::from_str( VITAL_UNUSED const std::string& s,
            VITAL_UNUSED tag2index_t& val ) const
{
  return false;
}

bool
kwiver_io_base<tag2index_t>
::read_xml( VITAL_UNUSED const TiXmlElement* e,
            VITAL_UNUSED tag2index_t& val ) const
{
  return false;
}

void
kwiver_io_base<tag2index_t>
::write_xml( VITAL_UNUSED std::ostream& os,
             VITAL_UNUSED const std::string& indent,
             VITAL_UNUSED const tag2index_t& val ) const
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
::from_csv( VITAL_UNUSED const std::map< std::string,
            std::string >& header_value_map,
            VITAL_UNUSED tag2index_t& val ) const
{
  return false;
}

std::ostream&
kwiver_io_base<tag2index_t>
::to_csv( std::ostream& os,
          VITAL_UNUSED const tag2index_t& val ) const
{
  return os;
}

std::ostream&
kwiver_io_base<index2val_t>
::to_stream( std::ostream& os,
             VITAL_UNUSED const index2val_t& val ) const
{
  return os;
}

bool
kwiver_io_base<index2val_t>
::from_str( VITAL_UNUSED const std::string& s,
            VITAL_UNUSED index2val_t& val ) const
{
  return false;
}

bool
kwiver_io_base<index2val_t>
::read_xml( VITAL_UNUSED const TiXmlElement* e,
            VITAL_UNUSED index2val_t& val ) const
{
  return false;
}

void
kwiver_io_base<index2val_t>
::write_xml( VITAL_UNUSED std::ostream& os,
             VITAL_UNUSED const std::string& indent,
             VITAL_UNUSED const index2val_t& val ) const
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
::from_csv(
  VITAL_UNUSED const std::map< std::string, std::string >& header_value_map,
  VITAL_UNUSED index2val_t& val ) const
{
  return false;
}

std::ostream&
kwiver_io_base<index2val_t>
::to_csv( std::ostream& os,
          VITAL_UNUSED const index2val_t& val ) const
{
  return os;
}

} // ...track_oracle
} // ...kwiver

