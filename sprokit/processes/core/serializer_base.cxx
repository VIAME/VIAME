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

#include "serializer_base.h"

#include <string>
#include <sstream>

namespace kwiver {

// ----------------------------------------------------------------------------
serializer_base::
serializer_base( sprokit::process&              proc,
                 kwiver::vital::logger_handle_t log )
  : m_proc( proc )
  , m_logger( log )
{
}

serializer_base::
~serializer_base()
{
}

// ----------------------------------------------------------------------------
void
serializer_base
::base_init()
{
  // scan through our port groups to make sure it all makes sense.
  for ( auto elem : m_port_group_list )
  {
    auto & pg = elem.second;

    // A group must have at least one port
    if (pg.m_items.size() < 1)
    {
      std::stringstream str;
      str <<  "There are no data items for group \"" << elem.first << "\"";

      VITAL_THROW( sprokit::invalid_configuration_exception, m_proc.name(), str.str() );
    }

    // determine which algorithm we should use. If m_algo_name is set
    // at this point, we are dealing with a multi-item packing
    // serializer.
    //
    // If it is not set, then we are dealing with a single item
    // converter and can use the input port type as the algorithm name.
    if ( pg.m_algo_name.empty() )
    {
      // There should only be one item in the group
      pg.m_algo_name = pg.m_items.cbegin()->second.m_port_type;
    }

    // create config items
    // serialize-protobuf:type = <algo-name>
    // serialize-protobuf:<data type>:foo = bar // possible but not likely

    auto algo_config = kwiver::vital::config_block::empty_config();
    const std::string ser_algo_type( "serialize-" + m_serialization_type );
    const std::string ser_type( ser_algo_type + vital::config_block::block_sep + "type" );

    algo_config->set_value( ser_type, pg.m_algo_name );

    std::stringstream str;
    algo_config->print( str );
    LOG_TRACE( m_logger, "Creating algorithm for (config block):\n" << str.str() << std::endl );

    vital::algorithm_sptr base_nested_algo;

    // create serialization algorithm
    vital::algorithm::set_nested_algo_configuration( ser_algo_type, // data type name
                                                     ser_algo_type, // config block name
                                                     algo_config,
                                                     base_nested_algo );

    pg.m_serializer = std::dynamic_pointer_cast< vital::algo::data_serializer > ( base_nested_algo );
    if ( ! pg.m_serializer )
    {
      std::stringstream str;
      str << "Unable to create serializer for type \""
          << pg.m_algo_name << "\" for " << m_serialization_type;

      VITAL_THROW( sprokit::invalid_configuration_exception, m_proc.name(), str.str() );
    }

    if ( ! vital::algorithm::check_nested_algo_configuration( ser_algo_type,
                                                              ser_algo_type,
                                                              algo_config ) )
    {
      VITAL_THROW( sprokit::invalid_configuration_exception,
                   m_proc.name(), "Configuration check failed." );
    }
  } // end for
}

// ----------------------------------------------------------------------------
bool
serializer_base::
vital_typed_port_info( sprokit::process::port_t const& port_name )
{
  // split port name into algo and item.
  // port_name ::= <group>/<item>
  //
  // Create port_group for <group>
  // Add entry for item.

  // Extract GROUP sub-string from port name
  sprokit::process::ports_t components;
  kwiver::vital::tokenize( port_name, components, "/" );

  if (components.size() > 2 )
  {
    LOG_ERROR( m_logger, "Port \"" << port_name
               << "\" does not have the correct format. "
               "Must be in the form \"<group>/<item>\"." );
    return false;
  }

  std::string algo_name;

  if (components.size() == 1 )
  {
    // No item specified. Assume a single item group
    components.push_back( kwiver::vital::algo::data_serializer::DEFAULT_ELEMENT_NAME );
  }
  else
  {
    // In the case where there are multiple items in the group, set the algo name to be the group name
    algo_name = components[0];
  }

  const std::string group_name = components[0];
  const std::string item_name = components[1];

  // if port has already been added, do nothing
  if (m_port_group_list.count( group_name ) == 0 )
  {
    // create a new empty group
    m_port_group_list[ group_name ] = port_group();
  }

  port_group& pg = m_port_group_list[ group_name ];

  // See if the item already exists in the item list. If so, then
  // the same port was connected more than once.
  if ( pg.m_items.count( item_name ) > 0 )
  {
    LOG_ERROR( m_logger, "Data item \"" << item_name <<"\" from input port \""
               << port_name << "\" has already been connected" );
    return false;
  }

  port_group::data_item di;
  di.m_port_name = port_name;
  di.m_element_name = item_name;

  pg.m_items[item_name] = di;
  pg.m_serialized_port_name = group_name; // expected port name
  pg.m_algo_name = algo_name; // can be empty string if single item group.

  return true;
}

// ----------------------------------------------------------------------------
void
serializer_base::
byte_string_port_info( sprokit::process::port_t const& port_name )
{
    if (m_port_group_list.count( port_name ) == 0 )
    {
      // create a new empty group
      m_port_group_list[ port_name ] = port_group();
    }

    port_group& pg = m_port_group_list[ port_name ];
    pg.m_serialized_port_name = port_name;
}

// ----------------------------------------------------------------------------
void
serializer_base::
set_port_type( sprokit::process::port_t const&      port_name,
               sprokit::process::port_type_t const& port_type )
{
  // Extract GROUP sub-string from port name
  sprokit::process::ports_t components;
  kwiver::vital::tokenize( port_name, components, "/" );

  if (components.size() == 1 )
  {
    // No item specified. Assume a single item group
    components.push_back( kwiver::vital::algo::data_serializer::DEFAULT_ELEMENT_NAME );
  }

  const std::string group_name = components[0];
  const std::string item_name = components[1];

  // update port handler
  port_group& pg = m_port_group_list[group_name];
  port_group::data_item& di = pg.m_items[item_name];

  di.m_port_type = port_type;
}

} // end namespace kwiver
