/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include "pipe_display.h"

#include <vital/config/config_block.h>
#include <vital/util/string.h>
#include <vital/util/wrap_text_block.h>

#include <sprokit/pipeline_util/pipe_declaration_types.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>

#include <algorithm>
#include <iterator>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <cstdlib>


namespace sprokit {

namespace {

// ==================================================================
class config_printer
{
public:
  config_printer( std::ostream& ostr );
  ~config_printer();

  void operator()( sprokit::config_pipe_block const& config_block );
  void operator()( sprokit::process_pipe_block const& process_block );
  void operator()( sprokit::connect_pipe_block const& connect_block ) const;

private:
  void print_config_value( sprokit::config_value_t const& config_value ) const;

  typedef std::set< sprokit::process::name_t > process_set_t;

  std::ostream& m_ostr;

};


config_printer
::config_printer( std::ostream& ostr )
  : m_ostr( ostr )
{
}


config_printer
::~config_printer()
{
}


// ==================================================================
class key_printer
{
public:
  key_printer( std::ostream& ostr );
  ~key_printer();

  void operator()( sprokit::config_value_t const& config_value ) const;


private:
  std::ostream& m_ostr;
};


// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::config_pipe_block const& config_block )
{
  kwiver::vital::config_block_keys_t const& keys = config_block.key;
  sprokit::config_values_t const& values = config_block.values;

  kwiver::vital::config_block_key_t const key_path = kwiver::vital::join( keys, kwiver::vital::config_block::block_sep );

  // generate pipe level config block
  m_ostr << "config " << key_path << std::endl;

  key_printer const printer( m_ostr );

  std::for_each( values.begin(), values.end(), printer );
}


// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::process_pipe_block const& process_block )
{
  sprokit::process::name_t const& name = process_block.name;
  sprokit::process::type_t const& type = process_block.type;
  sprokit::config_values_t const& values = process_block.config_values;

  m_ostr << "process " << name << std::endl
         << " :: " << type << std::endl;

  key_printer const printer( m_ostr );

  std::for_each( values.begin(), values.end(), printer );
}


// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::connect_pipe_block const& connect_block ) const
{
  sprokit::process::port_addr_t const& upstream_addr = connect_block.from;
  sprokit::process::port_addr_t const& downstream_addr = connect_block.to;

  sprokit::process::name_t const& upstream_name = upstream_addr.first;
  sprokit::process::port_t const& upstream_port = upstream_addr.second;
  sprokit::process::name_t const& downstream_name = downstream_addr.first;
  sprokit::process::port_t const& downstream_port = downstream_addr.second;

  m_ostr << "connect from " << upstream_name << "." << upstream_port << std::endl
         << "        to   " << downstream_name << "." << downstream_port << std::endl
         << std::endl;
}


// ------------------------------------------------------------------
key_printer
::key_printer( std::ostream& ostr )
  : m_ostr( ostr )
{
}


key_printer
::~key_printer()
{
}


// ------------------------------------------------------------------
void
key_printer
::operator()( sprokit::config_value_t const& config_value ) const
{
  const auto& value = config_value.value;
  const auto& keys = config_value.key_path;
  const auto key_path = kwiver::vital::join( keys, kwiver::vital::config_block::block_sep );

  const auto& flags = config_value.flags;

  m_ostr << "  " << key_path;

  if ( ! flags.empty() )
  {
    const auto flag_list = kwiver::vital::join( flags, "," );

    m_ostr << "[" << flag_list << "]";
  }

  m_ostr << " = " << value << std::endl;
}



} // end namespace

// ==================================================================
pipe_display::
pipe_display( std::ostream& ostr )
  : m_ostr( ostr )
{
}


pipe_display::
~pipe_display()
{ }


// ------------------------------------------------------------------
void
pipe_display::
display_pipe_blocks( const sprokit::pipe_blocks blocks )
{
  config_printer printer( m_ostr );

  m_ostr << "Number of blocks in list: " << blocks.size() << std::endl;

  for ( auto b : blocks )
  {
    kwiver::vital::visit( printer, b );
  }
}

} // end namespace
