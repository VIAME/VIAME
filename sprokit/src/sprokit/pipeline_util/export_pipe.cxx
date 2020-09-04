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

#include "export_pipe.h"

#include <vital/config/config_block.h>
#include <vital/util/string.h>
#include <vital/util/wrap_text_block.h>

#include <sprokit/pipeline_util/pipe_declaration_types.h>
#include <sprokit/pipeline_util/pipeline_builder.h>

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

//+ not sure what these are for
static sprokit::process::name_t denormalize_name( sprokit::process::name_t const& name );
static sprokit::process::name_t normalize_name( sprokit::process::name_t const& name );

// ==================================================================
class config_printer
{
public:
  config_printer( std::ostream& ostr, sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& conf );
  ~config_printer();

  void operator()( sprokit::config_pipe_block const& config_block );
  void operator()( sprokit::process_pipe_block const& process_block );
  void operator()( sprokit::connect_pipe_block const& connect_block ) const;

  void output_process_defaults();


private:
  void output_process_by_name( sprokit::process::name_t const& name, bool fatal_if_no_process );
  void output_process_block( sprokit::process_t const& name, std::string const& kind );
  void output_process( sprokit::process_t const& proc );

  typedef std::set< sprokit::process::name_t > process_set_t;

  std::ostream& m_ostr;

  sprokit::pipeline_t const m_pipe;
  kwiver::vital::config_block_sptr const m_config;
  process_set_t m_visited;
};


// ------------------------------------------------------------------
config_printer
::config_printer( std::ostream& ostr, sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& conf )
  : m_ostr( ostr )
  , m_pipe( pipe )
  , m_config( conf )
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

  kwiver::vital::config_block_key_t const key_path =
     kwiver::vital::join( keys, kwiver::vital::config_block::block_sep() );

  // generate pipe level config block
  m_ostr << "config " << key_path << std::endl;

  key_printer const printer( m_ostr );

  std::for_each( values.begin(), values.end(), printer );

  // Not sure what we are doing here
  sprokit::process::name_t const denorm_name = denormalize_name( key_path );

  output_process_by_name( denorm_name, false );
}


// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::process_pipe_block const& process_block )
{
  sprokit::process::name_t const& name = process_block.name;
  sprokit::process::type_t const& type = process_block.type;
  sprokit::config_values_t const& values = process_block.config_values;

  m_ostr << "process " << name << std::endl;
  m_ostr << " :: " << type << std::endl;

  key_printer const printer( m_ostr );

  std::for_each( values.begin(), values.end(), printer );

  output_process_by_name( name, true );
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

  m_ostr << "connect from " << upstream_name << "." << upstream_port << std::endl;
  m_ostr << "        to   " << downstream_name << "." << downstream_port << std::endl;

  m_ostr << std::endl;
}


// ------------------------------------------------------------------
void
config_printer
::output_process_defaults()
{
  sprokit::process::names_t const cluster_names = m_pipe->cluster_names();

  for( sprokit::process::name_t const & name : cluster_names )
  {
    if ( m_visited.count( name ) )
    {
      continue;
    }

    sprokit::process_cluster_t const cluster = m_pipe->cluster_by_name( name );
    sprokit::process_t const proc = std::static_pointer_cast< sprokit::process > ( cluster );

    static std::string const kind = "cluster";

    output_process_block( proc, kind );
  }

  sprokit::process::names_t const process_names = m_pipe->process_names();

  for( sprokit::process::name_t const & name : process_names )
  {
    if ( m_visited.count( name ) )
    {
      continue;
    }

    sprokit::process_t const proc = m_pipe->process_by_name( name );

    static std::string const kind = "process";

    output_process_block( proc, kind );
  }
}


// ------------------------------------------------------------------
void
config_printer
::output_process_by_name( sprokit::process::name_t const& name, bool fatal_if_no_process )
{
  sprokit::process_t proc;

  if ( ! m_visited.count( name ) )
  {
    try
    {
      proc = m_pipe->process_by_name( name );
    }
    catch ( sprokit::no_such_process_exception const& /*e*/ )
    {
      try
      {
        sprokit::process_cluster_t const cluster = m_pipe->cluster_by_name( name );

        proc = std::static_pointer_cast< sprokit::process > ( cluster );
      }
      catch ( sprokit::no_such_process_exception const& /*e*/ )
      {
        if ( fatal_if_no_process )
        {
          std::string const reason = "A process block did not result in a process being "
                                     "added to the pipeline: " + name;

          throw std::logic_error( reason );
        }
      }
    }
  }

  if ( proc )
  {
    output_process( proc );
  }
  else
  {
    m_ostr << std::endl;
  }
}


// ------------------------------------------------------------------
void
config_printer
::output_process_block( sprokit::process_t const& proc, std::string const& kind )
{
  sprokit::process::name_t const name = proc->name();
  sprokit::process::type_t const type = proc->type();
  sprokit::process::name_t const norm_name = normalize_name( name );

  m_ostr << "# Defaults for \'" << name << "\' " << kind << ":" << std::endl
         << "config " << norm_name << std::endl
         << "#:: " << type << std::endl;

  output_process( proc );
}


// ------------------------------------------------------------------
void
config_printer
  ::output_process( sprokit::process_t const& proc )
{
  sprokit::process::name_t const name = proc->name();
  kwiver::vital::config_block_keys_t const keys = proc->available_config();
  kwiver::vital::config_block_keys_t const tunable_keys = proc->available_tunable_config();
  sprokit::process::name_t const norm_name = normalize_name( name );
  kwiver::vital::wrap_text_block wtb;
  wtb.set_indent_string( "  #    " );

  for ( kwiver::vital::config_block_key_t const& key : keys )
  {
    if ( kwiver::vital::starts_with( key, "_" ) )
    {
      continue;
    }

    m_ostr << std::endl;

    sprokit::process::conf_info_t const& info = proc->config_info( key );

    // Process description by converting it to comments.
    kwiver::vital::config_block_description_t const desc = wtb.wrap_text( info->description );

    bool const is_tunable = ( 0 != std::count( tunable_keys.begin(), tunable_keys.end(), key ) );
    std::string const tunable = ( is_tunable ? "yes" : "no" );

    m_ostr  << "  # Key: " << key << std::endl
            << "  # Description:\n" << desc
            << "  # Tunable: " << tunable << std::endl;

    kwiver::vital::config_block_value_t const& def = info->def;

    if ( def.empty() )
    {
      m_ostr << "  # No default value" << std::endl;
    }
    else
    {
      m_ostr << "  # Default value: " << def << std::endl;
    }

    kwiver::vital::config_block_key_t const resolved_key =
      norm_name + kwiver::vital::config_block::block_sep() + key;

    if ( m_config->has_value( resolved_key ) )
    {
      kwiver::vital::config_block_value_t const cur_value =
        m_config->get_value< kwiver::vital::config_block_value_t > ( resolved_key );

      m_ostr << "  # Current value: " << cur_value << std::endl;
    }
    else
    {
      m_ostr << "  # No current value" << std::endl;
    }
  }

  m_ostr << std::endl;

  m_visited.insert( name );
} // config_printer::output_process


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
  const auto & value = config_value.value;
  const auto & keys = config_value.key_path;

  const auto key_path = kwiver::vital::join( keys,
               kwiver::vital::config_block::block_sep() );

  const auto & flags = config_value.flags;

  m_ostr << "  " << key_path;

  if ( ! flags.empty() )
  {
    const auto flag_list = kwiver::vital::join( flags, "," );

    m_ostr << "[" << flag_list << "]";
  }

  m_ostr << " = " << value << std::endl;
}


// ------------------------------------------------------------------
sprokit::process::name_t
denormalize_name( sprokit::process::name_t const& name )
{
  sprokit::process::name_t denorm_name(name);
  // replace ';' with '/'
  std::replace( denorm_name.begin(), denorm_name.end(), ':', '/' );

  return denorm_name;
}


// ------------------------------------------------------------------
sprokit::process::name_t
normalize_name( sprokit::process::name_t const& name )
{
  sprokit::process::name_t norm_name( name );
  // replace '/' with ';'
  std::replace( norm_name.begin(), norm_name.end(), '/', ':' );

  return norm_name;
}


// ==================================================================
export_pipe::
export_pipe( const sprokit::pipeline_builder& builder )
  : m_builder( builder )
{
}


export_pipe::
~export_pipe()
{ }


// ------------------------------------------------------------------
void
export_pipe::
generate( std::ostream& str )
{
  sprokit::pipeline_t const pipe = m_builder.pipeline();
  kwiver::vital::config_block_sptr const config = m_builder.config();
  sprokit::pipe_blocks const blocks = m_builder.pipeline_blocks();

  config_printer printer( str, pipe, config );

  for ( auto b : blocks )
  {
    kwiver::vital::visit( printer, b );
  }

  printer.output_process_defaults();
}

} // end namespace
