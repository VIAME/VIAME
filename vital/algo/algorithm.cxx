/*ckwg +29
 * Copyright 2014-2015, 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief base algorithm function implementations
 */

#include "algorithm.h"

#include <vital/logger/logger.h>
#include <vital/algo/algorithm_factory.h>

#include <sstream>
#include <algorithm>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
algorithm
::algorithm()
  : m_logger( kwiver::vital::get_logger( "vital.algorithm" ) )
{
}


// ------------------------------------------------------------------
void
algorithm
::attach_logger( std::string const& name )
{
  m_logger = kwiver::vital::get_logger( name );
}


// ------------------------------------------------------------------
void
algorithm
::set_impl_name( const std::string& name )
{
  m_impl_name = name;
}


// ------------------------------------------------------------------
kwiver::vital::logger_handle_t
algorithm
::logger() const
{
  return m_logger;
}


// ------------------------------------------------------------------
std::string
algorithm
::impl_name() const
{
  return m_impl_name;
}


// ------------------------------------------------------------------
/// Get this alg's \link kwiver::vital::config_block configuration block \endlink
config_block_sptr
algorithm
::get_configuration() const
{
  return config_block::empty_config( this->type_name() );
}


// ------------------------------------------------------------------
/// Helper function for properly getting a nested algorithm's configuration
void
algorithm
::get_nested_algo_configuration( std::string const& type_name,
                                 std::string const& name,
                                 config_block_sptr  config,
                                 algorithm_sptr     nested_algo )
{
  config_block_description_t type_comment =
    "Algorithm to use for '" + name + "'.\n"
    "Must be one of the following options:";

  // Get list of factories for the algo_name
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact_list = vpm.get_factories( type_name );

  for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
  {
    std::string reg_name;
    if ( ! a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, reg_name ) )
    {
      continue;
    }

    type_comment += "\n\t- " + reg_name;
    std::string tmp_d;
    if ( a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, tmp_d ) )
    {
      type_comment += " :: " + tmp_d;
    }
  }

  if ( nested_algo )
  {
    config->set_value( name + config_block::block_sep + "type",
                       nested_algo->impl_name(),
                       type_comment );

    config->subblock_view( name + config_block::block_sep + nested_algo->impl_name() )
      ->merge_config( nested_algo->get_configuration() );
  }
  else if ( ! config->has_value( name + config_block::block_sep + "type" ) )
  {
    config->set_value( name + config_block::block_sep + "type",
                       "",
                       type_comment );
  }
}


// ------------------------------------------------------------------
/// Helper method for properly setting a nested algorithm's configuration
void
algorithm
::set_nested_algo_configuration( std::string const& type_name,
                                 std::string const& name,
                                 config_block_sptr  config,
                                 algorithm_sptr&    nested_algo )
{
  static  kwiver::vital::logger_handle_t logger = kwiver::vital::get_logger( "vital.algorithm" );
  const std::string type_key = name + config_block::block_sep + "type";

  if ( config->has_value( type_key ) )
  {
    const std::string iname = config->get_value< std::string > ( type_key );
    if ( has_algorithm_impl_name( type_name, iname ) )
    {
      nested_algo = create_algorithm( type_name, iname );;
      nested_algo->set_configuration(
        config->subblock_view( name + config_block::block_sep + iname )
                                    );
    }
    else
    {
      std::stringstream msg;
      msg << "Could not find implementation \"" << iname
          << "\" for \"" << type_name << "\"";

      // Add line number if known
      std::string file;
      int line(0);
      if ( config->get_location( type_key, file, line ) )
      {
        msg << " as requested from "
                << file << ":" << line;
      }

      LOG_WARN( logger, msg.str() );
    }
  }
  else
  {
    LOG_WARN( logger, "Config item \"" << type_key
              << "\" not found for \"" << type_name << "\"." );
  }
}

// ------------------------------------------------------------------
/// Helper method for checking that basic nested algorithm configuration is valid
bool
algorithm
::check_nested_algo_configuration( std::string const& type_name,
                                   std::string const& name,
                                   config_block_sptr  config )
{
  static  kwiver::vital::logger_handle_t logger = kwiver::vital::get_logger( "vital.algorithm" );
  const std::string type_key = name + config_block::block_sep + "type";

  if ( ! config->has_value( type_key ) )
  {
    LOG_WARN( logger, "Configuration Failure: missing value: " << type_key );
    return false;
  }

  const std::string instance_name = config->get_value< std::string >( type_key );
  if ( ! has_algorithm_impl_name( type_name, instance_name ) )
  {
    std::stringstream msg;
    msg << "Implementation '" << instance_name << "' for algorithm type "
        << type_key << " could not be found.\nMake sure KWIVER_PLUGIN_PATH is set correctly.";

    // Get list of factories for the algo_name
    kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
    auto fact_list = vpm.get_factories( type_name );
    bool first {true};

    // Find the one that provides the impl_name
    for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
    {
      // Collect a list of all available implementations for this algorithm
      std::string reg_name;
      if ( a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, reg_name ) )
      {
        if (first)
        {
          first = false;
          msg << "   Available implementations are:";
        }

        msg << "\n      " << reg_name;
      }
    }

    if (first)
    {
      msg << "   There are no implementations available.";
    }

    LOG_WARN( logger, msg.str() );
    return false;
  }

  // recursively check the configuration of the sub-algorithm
  const std::string qualified_name = name + config_block::block_sep + instance_name;

  // Need a real algorithm object to check with
  try
  {
    if ( ! create_algorithm( type_name, instance_name )->check_configuration(
      config->subblock_view( qualified_name ) ) )
    {
      LOG_WARN( logger,  "Configuration Failure Backtrace: "
                << qualified_name );
      return false;
    }
  }
  catch ( const kwiver::vital::plugin_factory_not_found& e )
  {
    LOG_WARN( logger, e.what() );
  }
  return true;
}

} }     // end namespace
