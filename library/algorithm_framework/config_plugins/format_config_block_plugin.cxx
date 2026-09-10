// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/config/config_helpers.txx>
#include <viame/algorithm_framework/config/format_config_block.h>
#include "format_config_export.h"
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/algorithm_framework/util/wrap_text_block.h>

namespace kwiver::vital {

// ----------------------------------------------------------------------------
/// @brief Formats config block using markdown
///
class FORMAT_CONFIG_NO_EXPORT format_config_block_markdown
  : public format_config_block
{
public:
  PLUGGABLE_IMPL(
    format_config_block_markdown,
    "Format config block with markdown"
  )

  void print( const config_block_sptr config, std::ostream& str ) override;

protected:
  void
  set_configuration_internal( [[maybe_unused]] config_block_sptr ) override {}
  void
  initialize() override {}
}; // end class format_config_block_markdown

// ----------------------------------------------------------------------------
void
format_config_block_markdown
::print( config_block_sptr config, std::ostream& str )
{
  wrap_text_block wtb;

  kwiver::vital::config_block_keys_t all_keys = config->available_values();

  for( kwiver::vital::config_block_key_t key : all_keys )
  {
    std::string ro;

    auto const val =
      config->get_value< kwiver::vital::config_block_value_t >( key );
    if( config->is_read_only( key ) )
    {
      ro = "[RO]";
    }

    str << "**" << key << "** " << ro << " = " << val
        << std::endl;

    std::string descrip = config->get_description( key );
    if( !descrip.empty() )
    {
      str << wtb.wrap_text( descrip );
    }

    str << std::endl;
  }
}

// ----------------------------------------------------------------------------
/// @brief Formats config block in a tree structure
///
class FORMAT_CONFIG_NO_EXPORT format_config_block_tree
  : public format_config_block
{
public:
  PLUGGABLE_IMPL(
    format_config_block_tree,
    "Format config block to output as a tree",
    PARAM( opt_prefix, std::string, "Optional prefix for output" ), \
    PARAM_DEFAULT(
      opt_gen_source_loc, bool,
      "Provide information about config file.", false )
  )

  void print( const config_block_sptr config, std::ostream& str ) override;

protected:
  void format_block(
    std::ostream& str,
    const config_block_sptr config,
    const std::string& prefix );
}; // end class format_config_block_tree

// ----------------------------------------------------------------------------
void
format_config_block_tree
::print( const config_block_sptr config, std::ostream& str )
{
  format_block( str, config, this->get_opt_prefix() );
}

// ----------------------------------------------------------------------------
void
format_config_block_tree
::format_block(
  std::ostream& str,
  const config_block_sptr config,
  const std::string& prefix )
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();

  auto ix = all_keys.begin();
  auto ex = all_keys.end();

  for(; ix != ex; ++ix )
  {
    // get first component
    auto pos = ix->find_first_of( ":" );
    if( ( pos != std::string::npos ) && ( pos > 0 ) )
    {
      // Block does not have trailing ':'
      std::string current_block = ix->substr( 0, pos );

      // extract subblock to process further
      const auto subb = config->subblock( current_block );

      // Create block markers and format subblock
      str << prefix << "block   " << current_block << std::endl;

      // Indent this nested block
      format_block( str, subb, prefix + "  " );

      str << prefix << "endblock     # " << current_block << std::endl;

      // skip over the entries we have processed
      for(; ix != ex; ++ix )
      {
        if( !starts_with( *ix, current_block + ":" ) )
        {
          --ix; // backup so outer loop will increment
          break;
        }
      } // end for

      if( ix == ex ) { break; }
    }
    else
    {
      // current key does not have elements
      // format one element
      wrap_text_block wtb;
      wtb.set_line_length( 100 );
      wtb.set_indent_string( std::string( "#" ) + prefix );

      std::string ro;
      auto const val =
        config->get_value< kwiver::vital::config_block_value_t >( *ix );

      if( config->is_read_only( *ix ) )
      {
        ro = "[RO]";
      }

      str << prefix << *ix << ro << " = " << val << std::endl;

      std::string descrip = config->get_description( *ix );
      if( !descrip.empty() )
      {
        str << wtb.wrap_text( descrip );
      }

      if( this->get_opt_gen_source_loc() )
      {
        // Add location information if available
        std::string file;
        int line( 0 );
        std::stringstream sstream;
        if( config->get_location( *ix, file, line ) )
        {
          sstream << "Defined at " << file << ":" << line << "\n";
          str << wtb.wrap_text( sstream.str() );
        }
      }
    }
  } // end for
}

// ----------------------------------------------------------------------------
extern "C"
FORMAT_CONFIG_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpl )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "format-config-block" );

  // common handle, so we can add attributes to factories as we add them.
  kwiver::vital::plugin_factory_handle_t fact;

  fact = vpl.add_factory<
    kwiver::vital::format_config_block,
    kwiver::vital::format_config_block_markdown >( "markdown" );
  fact->add_attribute(
    kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
    module_name )
    .add_attribute(
    kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "Formats the config block using markdown." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute(
      kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION,
      "Kitware Inc." );

  fact = vpl.add_factory<
    kwiver::vital::format_config_block,
    kwiver::vital::format_config_block_tree >( "tree" );
  fact->add_attribute(
    kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME,
    module_name )
    .add_attribute(
    kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
    "Formats the config block in an indented tree format." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute(
      kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION,
      "Kitware Inc." );
}

} // end namespace
