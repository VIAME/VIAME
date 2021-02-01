/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#include "bakery_display.h"

#include <vital/util/wrap_text_block.h>

namespace sprokit {

bakery_display
::bakery_display( std::ostream& str )
  : m_ostr( str )
{ }

// ----------------------------------------------------------------------------
void
bakery_display
::print( bakery_base const& b_b )
{

/*
  b_b.m_configs
  b_b.m_processes
  b_b.m_connections
*/

  // display pipeline internals
  m_ostr << m_prefix << "---- Config Items ----\n";

  for ( auto const& obj : b_b.m_configs )
  {
    // m_configs is a vector of config_decl_t
    // config_decl_t is a pair <block_key, config_info_t >
    // config_info_t has: <need separate formatter for this type>
    // - value : string
    // - read_only : bool
    // - relative_path : bool
    // - defined_log : source_loc
    //
    m_ostr << m_prefix << "  Config name:  " << obj.first << std::endl
           << m_prefix << "  Value:        " << obj.second.value << std::endl
           << m_prefix << "  Read-only:    " << ( obj.second.read_only ? "Yes" : "No") << std::endl
           << m_prefix << "  Relativepath: " << ( obj.second.relative_path ? "Yes" : "No") << std::endl
           << std::endl;
  }

  m_ostr << m_prefix << "---- Processes ----\n";
  for ( auto const& obj : b_b.m_processes )
  {
    // process name :: process_type
    m_ostr << m_prefix << "process " << obj.first << "\n"
           << " :: " << obj.second << std::endl
           << std::endl;
    // Maybe print the config that is associated with this process name.
  }

  m_ostr << m_prefix << "---- Connections ----\n";
  for ( auto const& obj : b_b.m_connections )
  {
    m_ostr << m_prefix << "connect from " << obj.first.first << "." << obj.first.second << std::endl
           << m_prefix << "        to   " << obj.second.first << "." << obj.second.second << std::endl
           << std::endl;
  }

}

// ----------------------------------------------------------------------------
void
bakery_display
::print( cluster_bakery const& c_b )
{
  namespace kv = kwiver::vital;

  /*
    c_b.m_type - name of the cluster
    c_b.m_description - descriptive text
    c_b.m_cluster.m_configs
    c_b.m_cluster.m_inputs
    c_b.m_cluster.m_outputs

    then bakery_base fields
   */

  kwiver::vital::wrap_text_block wtb;
  wtb.set_indent_string( m_prefix + "    " );

  m_ostr << m_prefix << "---------------------\n"
         << m_prefix << "Cluster type: " << c_b.m_type << std::endl
         << m_prefix << "Description: " << wtb.wrap_text( c_b.m_description ) << std::endl;

  // display c_b.m_configs
  bool group_empty{ true };
  m_ostr << m_prefix << "Cluster Configuration\n";
  for ( auto const& c : c_b.m_cluster->m_configs )
  {
    auto const flags_str = kv::join( c.config_value.flags, ", " );
    auto const& def = c.config_value.value;

    // description of config value
    m_ostr << m_prefix << "    Name       : " << kv::join( c.config_value.key_path, ":") << std::endl
           << m_prefix << "    Default    : " << def << std::endl
           << m_prefix << "    Description: " << wtb.wrap_text( c.description )
           << m_prefix << "    Flags      : " << flags_str << std::endl
           << std::endl;

    group_empty = false;
  }

  if (group_empty)
  {
    m_ostr << m_prefix << "    None\n\n";
  }

  // display c_b.m_inputs
  group_empty = true;
  m_ostr << m_prefix << "Cluster Input Ports\n";
  for ( auto const& port : c_b.m_cluster->m_inputs )
  {
    group_empty = false;

    // description
    // from port
    // targets vector
    m_ostr << m_prefix << "  Input port map from: " << port.from << std::endl;

    for ( auto const& p : port.targets )
    {
      m_ostr << m_prefix << "                   to: "
             << p.first << "." << p.second << std::endl;
    }

    m_ostr << wtb.wrap_text( port.description )
           << std::endl;
  }

  if (group_empty)
  {
    m_ostr << m_prefix << "    None\n\n";
  }

  // display c_b.m_outputs
  group_empty = true;
  m_ostr << m_prefix << "Cluster Output Poprts\n";
  for ( auto const& port : c_b.m_cluster->m_outputs )
  {
    group_empty = false;

    // description
    // from
    // to
    m_ostr << m_prefix << "  Output port map from: " << port.from.first << "." << port.from.second << std::endl
           << m_prefix << "                    to: " << port.to << std::endl
           << wtb.wrap_text( port.description )
                 << std::endl;
  }

  if (group_empty)
  {
    m_ostr << m_prefix << "    None\n\n";
  }
}

// ----------------------------------------------------------------------------
void
bakery_display
::set_prefix( const std::string& pfx )
{
  m_prefix = pfx;
}

// ----------------------------------------------------------------------------
void
bakery_display
::generate_source_loc( bool opt )
{
  m_gen_source_loc = opt;
}

} // end namespace sprokit
