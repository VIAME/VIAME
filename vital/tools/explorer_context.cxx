// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
