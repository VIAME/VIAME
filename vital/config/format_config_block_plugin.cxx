/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "format_config_block.h"

#include <vital/config/format_config_export.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/wrap_text_block.h>
#include <vital/util/string.h>

namespace kwiver {
namespace vital {

// ============================================================================
/**
 * @brief Formats config block using markdown
 *
 */
class FORMAT_CONFIG_NO_EXPORT format_config_block_markdown
  : public format_config_block
{
public:
  // -- CONSTRUCTORS --
  format_config_block_markdown();
  virtual ~format_config_block_markdown() = default;

  void print( std::ostream& str ) override;

}; // end class format_config_block_markdown

format_config_block_markdown::
format_config_block_markdown()
{ }

// ----------------------------------------------------------------------------
void format_config_block_markdown::
print( std::ostream& str )
{
  wrap_text_block wtb;
  wtb.set_indent_string( opt_prefix + "    " );

  kwiver::vital::config_block_keys_t all_keys = m_config->available_values();

  for ( kwiver::vital::config_block_key_t key : all_keys )
  {
    std::string ro;

    auto const val = m_config->get_value< kwiver::vital::config_block_value_t > ( key );
    if ( m_config->is_read_only( key ) )
    {
      ro = "[RO]";
    }

    str << opt_prefix << "**" << key << "** " << ro << " = " << val << std::endl;
    std::string descrip = m_config->get_description( key );
    if ( ! descrip.empty() )
    {
        str << wtb.wrap_text( descrip );
    }

    if ( opt_gen_source_loc )
    {
      // Add location information if available
      std::string file;
      int line( 0 );
      if ( m_config->get_location( key, file, line ) )
      {
        str << opt_prefix << "    Defined at " << file << ":" << line << std::endl;
      }
    }

    str << std::endl;
  }
}

// ============================================================================
/**
 * @brief Formats config block in a tree structure
 *
 */
class FORMAT_CONFIG_NO_EXPORT format_config_block_tree
  : public format_config_block
{
public:
  // -- CONSTRUCTORS --
  format_config_block_tree();
  virtual ~format_config_block_tree() = default;

  void print( std::ostream& str ) override;

protected:
  void format_block( std::ostream& str,
                     const config_block_sptr config,
                     const std::string& prefix );

}; // end class format_config_block_tree


format_config_block_tree::
format_config_block_tree()
{ }

// ----------------------------------------------------------------------------
void format_config_block_tree::
print( std::ostream& str )
{
  format_block( str, m_config, opt_prefix );
}

// ----------------------------------------------------------------------------
void format_config_block_tree::
format_block( std::ostream& str,
              const config_block_sptr config,
              const std::string& prefix )
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();

  auto ix = all_keys.begin();
  auto ex = all_keys.end();

  for ( ; ix != ex; ++ix )
  {
    // get first component
    auto pos = ix->find_first_of( ":" );
    if ( (pos != std::string::npos) && (pos > 0) )
    {
      // Block does not have trailing ':'
      std::string current_block = ix->substr( 0, pos );

      // extract subblock to process further
      const auto subb = config->subblock( current_block );

      // Create block markers and format subblock
      str << prefix << "block   " << current_block << std::endl;

      // Indent this nested block
      format_block( str, subb, prefix + "  ");

      str << prefix << "endblock     # " << current_block << std::endl;

      // skip over the entries we have processed
      for ( ; ix != ex; ++ix )
      {
        if ( ! starts_with( *ix, current_block + ":" ) )
        {
          --ix; // backup so outer loop will increment
          break;
        }
      } // end for

      if (ix == ex) { break; }
    }
    else
    {
      // current key does not have elements
      // format one element
      wrap_text_block wtb;
      wtb.set_line_length( 100 );
      wtb.set_indent_string( std::string("#") + prefix );

      std::string ro;
      auto const val = config->get_value< kwiver::vital::config_block_value_t > ( *ix );

      if ( config->is_read_only( *ix ) )
      {
        ro = "[RO]";
      }

      str << prefix << *ix << ro << " = " << val << std::endl;
      std::string descrip = config->get_description( *ix );
      if ( ! descrip.empty() )
      {
        str << wtb.wrap_text( descrip );
      }

      if ( opt_gen_source_loc )
      {
        // Add location information if available
        std::string file;
        int line( 0 );
        std::stringstream sstream;
        if ( config->get_location( *ix, file, line ) )
        {
          sstream << "Defined at " << file << ":" << line << "\n";
          str << wtb.wrap_text( sstream.str() );
        }
      }

    }
  } // end for
}

// ============================================================================
  extern "C"
FORMAT_CONFIG_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "format-config-block" );

  if ( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  kwiver::vital::plugin_factory_handle_t
  fact = vpm.ADD_FACTORY( kwiver::vital::format_config_block, kwiver::vital::format_config_block_markdown );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "markdown")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Formats the config block using markdown.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_FACTORY( kwiver::vital::format_config_block, kwiver::vital::format_config_block_tree );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "tree")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Formats the config block in an indented tree format.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - -
  vpm.mark_module_as_loaded( module_name );

}



} } // end namespace
