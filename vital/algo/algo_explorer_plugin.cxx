/*ckwg +29
 * Copyright 2016-2017, 2020 by Kitware, Inc.
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

#include <vital/algo/algo_explorer_plugin_export.h>

#include <vital/tools/explorer_plugin.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/util/wrap_text_block.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Plugin to provide detailed dsplay of algorithm plugins.
 *
 * This class implements a plugin category formatter for the plugin_explorer
 * tool.
 */
class algo_explorer
  : public category_explorer
{
public:
  // -- CONSTRUCTORS --
  algo_explorer();
  virtual ~algo_explorer();

  virtual bool initialize( explorer_context* context );
  virtual void explore( const kwiver::vital::plugin_factory_handle_t fact );

  void display_algo( std::shared_ptr< kwiver::vital::algorithm_factory > fact );

  // instance data
  explorer_context* m_context;
}; // end class algo_explorer


// ==================================================================
algo_explorer::
algo_explorer()
{ }


algo_explorer::
~algo_explorer()
{ }


// ------------------------------------------------------------------
bool
algo_explorer::
initialize( explorer_context* context )
{
  m_context = context;
  return true;
}


// ------------------------------------------------------------------
void
algo_explorer::
  explore( const kwiver::vital::plugin_factory_handle_t fact )
{
  // downcast to correct factory type.
  std::shared_ptr< kwiver::vital::algorithm_factory > pf =
    std::dynamic_pointer_cast< kwiver::vital::algorithm_factory > ( fact );

  if ( ! pf )
  {
    // Wrong type of factory returned.
    m_context->output_stream() << "Factory for algorithm could not be converted to algorithm_factory type.";
    return;
  }

  display_algo( pf );
}


// ------------------------------------------------------------------
void
algo_explorer::
display_algo( std::shared_ptr< kwiver::vital::algorithm_factory > fact )
{
  const std::string indent( "    " );

  std::string type = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::INTERFACE_TYPE, type );

  std::string impl = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_context->format_description( descrip );

  if ( m_context->if_brief() )
  {
    m_context->output_stream() << indent << "Algorithm type: "
                               << type << "   Implementation: " << impl << std::endl;
    return;
  }

  m_context->output_stream()  << "---------------------\n"
                              << "Info on algorithm type \"" << type << "\" implementation \"" << impl << "\""
                              << std::endl;

  m_context->display_attr( fact );

  if ( m_context->if_detail() )
  {
    kwiver::vital::algorithm_sptr ptr = fact->create_object();

    // Get configuration
    auto config = ptr->get_configuration();

    auto all_keys = config->available_values();

    m_context->output_stream() << indent << "-- Configuration --" << std::endl;

    for( auto key : all_keys )
    {
      auto val = config->get_value< kwiver::vital::config_block_value_t > ( key );

      m_context->output_stream() << indent << "\"" << key << "\" = \"" << val << "\"\n";

      kwiver::vital::config_block_description_t descr = config->get_description( key );
      m_context->output_stream() << indent << "Description: " << m_context->format_description( descr )
                                 << std::endl;
    }
  }

} // algo_explorer::explore


// ==================================================================
/**
 * @brief Plugin to provide detailed dsplay of algorithm plugins.
 *
 * This class implements a plugin category formatter for the plugin_explorer
 * tool generating output in pipeline file format.
 */
class algo_explorer_pipe
  : public category_explorer
{
public:
  // -- CONSTRUCTORS --
  algo_explorer_pipe();
  virtual ~algo_explorer_pipe();

  bool initialize( explorer_context* context ) override;
  void explore( const kwiver::vital::plugin_factory_handle_t fact ) override;


  // instance data
  explorer_context* m_context;

  // Need special indent prefix so we can not use normal text wrapper.
  kwiver::vital::wrap_text_block m_wtb;

}; // end class algo_explorer_pipe


// ==================================================================
algo_explorer_pipe::
algo_explorer_pipe()
{
  m_wtb.set_indent_string( "#      " );
}


algo_explorer_pipe::
~algo_explorer_pipe()
{ }


// ------------------------------------------------------------------
bool
algo_explorer_pipe::
initialize( explorer_context* context )
{
  m_context = context;
  return true;
}


// ------------------------------------------------------------------
void
algo_explorer_pipe::
  explore( const kwiver::vital::plugin_factory_handle_t pf )
{
  // downcast to correct factory type.
  std::shared_ptr< kwiver::vital::algorithm_factory > fact =
    std::dynamic_pointer_cast< kwiver::vital::algorithm_factory > ( pf );

  if ( ! fact )
  {
    // Wrong type of factory returned.
    m_context->output_stream() << "Factory for algorithm could not be converted to algorithm_factory type.";
    return;
  }

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_wtb.wrap_text( descrip );

  std::string impl = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl );

  // algo.type = impl
  m_context->output_stream() << "# ---------------------------------" << std::endl
                             << "type = " << impl << std::endl
                             << descrip << std::endl
                             << "block " << impl << std::endl;

  kwiver::vital::algorithm_sptr ptr = fact->create_object();

  // Get configuration
  auto config = ptr->get_configuration();
  auto all_keys = config->available_values();

  for( auto key : all_keys )
  {
    auto val = config->get_value< kwiver::vital::config_block_value_t > ( key );

    m_context->output_stream() << "    " << key << " = " << val << std::endl;

    kwiver::vital::config_block_description_t descr = config->get_description( key );
    m_context->output_stream() << m_wtb.wrap_text( descr ) << std::endl;
  } // end foreach over config

  m_context->output_stream() << "endblock\n" << std::endl;
}

} } // end namespace

// ==================================================================
extern "C"
ALGO_EXPLORER_PLUGIN_EXPORT
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  static std::string module("algo_explorer_plugin" );
  if ( vpm.is_module_loaded( module ) )
  {
    return;
  }

  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::algo_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Plugin explorer for algorithm category." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::algo_explorer_pipe );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "algorithm-pipe" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Plugin explorer for algorithm category. Generates pipeline format output." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

    vpm.mark_module_as_loaded( module );
}
