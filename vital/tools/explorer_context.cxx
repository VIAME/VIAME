/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#include "explorer_plugin.h"
#include "explorer_context_priv.h"
#include <vital/config/config_block.h>
#include <vital/util/string.h>

namespace kwiver {
namespace vital {

// ==================================================================
// --- Explorer context methods ---
// ------------------------------------------------------------------
explorer_context::
explorer_context( explorer_context::priv* pp )
  : p( pp )
{ }


explorer_context::
~explorer_context()
{ }


// ------------------------------------------------------------------
std::ostream&
explorer_context::
output_stream() const
{
  return *p->m_out_stream;
}


// ------------------------------------------------------------------
cxxopts::Options&
explorer_context::
command_line_args()
{
  return *p->m_cmd_options;
}


// ------------------------------------------------------------------
cxxopts::ParseResult&
explorer_context::
command_line_result()
{
  return *p->m_result;
}


// ------------------------------------------------------------------
const std::string&
explorer_context::
formatting_type() const
{
  return p->formatting_type;
}


// ------------------------------------------------------------------
std::string
explorer_context::
wrap_text( const std::string& text ) const
{
  return p->m_wtb.wrap_text( text );
}

// ------------------------------------------------------------------
//
// display full factory list
//
void
explorer_context::
display_attr( const kwiver::vital::plugin_factory_handle_t fact ) const
{
  p->display_attr( fact );
}

// ----------------------------------------------------------------------------
std::string
explorer_context
::format_description( std::string const& text ) const
{
  std::string output;

  // String off the first line. Up to the first new-line.
  auto pos = text.find( '\n' );
  if ( pos == std::string::npos)
  {
    output = wrap_text( text );
    left_trim( output );
  }
  else
  {
    output = wrap_text( text.substr( 0, pos ) );
    string_trim( output );
    output += "\n";

    // If in detail mode, print the rest of the description
    if( if_detail() )
    {
      std::string descr {text.substr( pos )};
      string_trim( descr );
      output += "\n";
      output += wrap_text( descr );
    }

  }
  return output;
}

// ----------------------------------------------------------------------------
void
explorer_context
::print_config( kwiver::vital::config_block_sptr const config ) const
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();
  const std::string indent( "    " );

  output_stream() << indent << "Configuration block contents\n";

  for( auto key : all_keys )
  {
    auto val = config->get_value< kwiver::vital::config_block_value_t > ( key );
    output_stream() << std::endl
                    << indent << "\"" << key << "\" = \"" << val << "\"\n";

    auto descr = config->get_description( key );
    output_stream() << indent << "Description: "
                    << format_description( descr )
                    << std::endl;
  }
}


// ------------------------------------------------------------------
bool explorer_context::if_detail() const { return p->opt_detail; }
bool explorer_context::if_brief() const { return p->opt_brief; }

} } // end namespace
