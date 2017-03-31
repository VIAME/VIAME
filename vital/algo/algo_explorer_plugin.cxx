/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
  std::string indent( "    " );

  // downcast to correct factory type.
  kwiver::vital::algorithm_factory* pf = dynamic_cast< kwiver::vital::algorithm_factory* > ( fact.get() );

  if ( 0 == pf )
  {
    // Wrong type of factory returned.
    m_context->output_stream() << "Factory for algorithm could not be converted to algorithm_factory type.";
    return;
  }

  std::string type = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::INTERFACE_TYPE, type );

  std::string impl = "-- not set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, impl );

  std::string descrip = "-- Not_Set --";
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip );
  descrip = m_context->wrap_text( descrip );

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

  if ( m_context->if_config() )
  {
    kwiver::vital::algorithm_sptr ptr = pf->create_object();

    // Get configuration
    auto config = ptr->get_configuration();

    auto all_keys = config->available_values();

    m_context->output_stream() << indent << "-- Configuration --" << std::endl;

    VITAL_FOREACH( auto  key, all_keys )
    {
      auto  val = config->get_value< kwiver::vital::config_block_value_t > ( key );

      m_context->output_stream() << indent << "\"" << key << "\" = \"" << val << "\"\n";

      kwiver::vital::config_block_description_t descr = config->get_description( key );
      m_context->output_stream() << indent << "Description: " << m_context->wrap_text( descr ) << std::endl;
    }
  }

} // algo_explorer::explore

} } // end namespace

// ==================================================================
extern "C"
ALGO_EXPLORER_PLUGIN_EXPORT
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::algo_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Plugin explorer for algorithm category." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
}
