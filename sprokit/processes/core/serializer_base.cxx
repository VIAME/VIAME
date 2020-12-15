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

#include <vital/util/hex_dump.h>
#include <vital/config/config_block_formatter.h>

#include <string>
#include <sstream>
#include <iostream>
#include <cstdint>

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
  // Scan through our port groups to make sure it all makes sense.
  // The port group name is used as the message-name
  for ( auto msg_elem : m_message_spec_list )
  {
    auto& pg = m_message_spec_list[msg_elem.first];

    // A group must have at least one port
    if (pg.m_elements.size() < 1)
    {
      std::stringstream str;
      str <<  "There are no data elements for group \"" << msg_elem.first << "\"";

      VITAL_THROW( sprokit::invalid_configuration_exception, m_proc.name(), str.str() );
    }

    // Test to see if the output port has been connected to.
    if ( ! pg.m_serialized_port_created )
    {
      VITAL_THROW( sprokit::missing_connection_exception, m_proc.name(), msg_elem.first,
                   "Output port has not been connected" );
    }

    for ( auto it : pg.m_elements )
    {
      auto& elem_spec = pg.m_elements.at(it.first);

      // create config items
      // serialize-protobuf:type = <algo-name>

      auto algo_config = vital::config_block::empty_config();

      const std::string ser_algo_type( "serialize-" + m_serialization_type );
      const std::string ser_type( ser_algo_type +
                                  vital::config_block::block_sep() +
                                  "type" );

      algo_config->set_value( ser_type, elem_spec.m_algo_name );

      std::stringstream str;
      vital::config_block_formatter fmt( algo_config );
      fmt.print( str );
      LOG_TRACE( m_logger, "Creating algorithm for (config block):\n" << str.str() << std::endl );

      vital::algorithm_sptr base_nested_algo;

      // create serialization algorithm
      vital::algorithm::set_nested_algo_configuration( ser_algo_type, // data type name
                                                       ser_algo_type, // config block name
                                                       algo_config,
                                                       base_nested_algo );

      elem_spec.m_serializer = std::dynamic_pointer_cast< vital::algo::data_serializer > ( base_nested_algo );
      if ( ! elem_spec.m_serializer )
      {
        std::stringstream str;
        str << "Unable to create serializer for type \""
            << elem_spec.m_algo_name << "\" for " << m_serialization_type;

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
  } // end for
}

// ----------------------------------------------------------------------------
bool
serializer_base::
vital_typed_port_info( sprokit::process::port_t const& port_name )
{
  // split port name into algo and element.
  // port_name ::= <message-name>/<element>
  //
  // Create message_spec for <message-name>
  // Add entry for element.
  sprokit::process::ports_t components;
  kwiver::vital::tokenize( port_name, components, "/" );

  std::string message_name;
  std::string element_name;

  if ( components.size() != 2 )
  {
    LOG_ERROR( m_logger, "Port \"" << port_name
               << "\" does not have the correct format. "
               "Must be in the form \"<message-name>/<element>\"." );
    return false;
  }

  message_name = components[0];
  element_name = components[1];

  // if port has already been added, do nothing
  if (m_message_spec_list.count( message_name ) == 0 )
  {
    // create a new empty group
    m_message_spec_list[ message_name ] = message_spec();
    LOG_TRACE( m_logger, "Creating new group \"" << message_name << "\" for typed port" );
  }

  message_spec& pg = m_message_spec_list[ message_name ];

  // See if the element already exists in the element list. If so, then
  // the port has already been created.
  if ( pg.m_elements.count( element_name ) != 0 )
  {
    return false;
  }

  message_spec::message_element di;
  di.m_port_name = port_name;
  di.m_element_name = element_name;

  pg.m_elements[element_name] = di;
  pg.m_serialized_port_name = message_name; // expected port name

  LOG_TRACE( m_logger, "Created port element \"" << element_name
             << "\" for message name \"" << message_name << "\"" );

  return true;
}

// ----------------------------------------------------------------------------
bool
serializer_base::
byte_string_port_info( sprokit::process::port_t const& port_name )
{
  if (m_message_spec_list.count( port_name ) == 0 )
  {
    // create a new empty group
    m_message_spec_list[ port_name ] = message_spec();
    LOG_TRACE( m_logger, "Creating new group for byte_string port \"" << port_name << "\"" );
  }

  message_spec& pg = m_message_spec_list[ port_name ];

  if ( ! pg.m_serialized_port_created )
  {
    LOG_TRACE( m_logger, "Creating byte_string port \"" << port_name << "\"" );
    pg.m_serialized_port_name = port_name;
    pg.m_serialized_port_created = true;

    return true;
  }

  LOG_TRACE( m_logger, "byte_string port \"" << port_name
             << "\" has already been created." );

  return false;
}

// ----------------------------------------------------------------------------
void
serializer_base::
set_port_type( sprokit::process::port_t const&      port_name,
               sprokit::process::port_type_t const& port_type )
{
  // Extract sub-strings from port name
  sprokit::process::ports_t components;
  kwiver::vital::tokenize( port_name, components, "/" );
  std::string message_name;
  std::string element_name;

  if ( components.size() != 2 )
  {
    LOG_ERROR( m_logger, "Port \"" << port_name
               << "\" does not have the correct format. "
               "Must be in the form \"<message>/<element>\"" );
    return;
  }

  message_name = components[0];
  element_name = components[1];

  // update port handler
  message_spec& pg = m_message_spec_list[message_name];
  message_spec::message_element& di = pg.m_elements[element_name];
  di.m_port_type = port_type;

  // Currently, the algo name is the same as the data type on that port.
  di.m_algo_name = port_type;

  LOG_TRACE( m_logger, "Setting port type for message \"" << message_name
             << "\" element \"" << element_name << "\" to \"" << port_type << "\"" );
}

// ----------------------------------------------------------------------------
void serializer_base::
decode_message( const std::string& message )
{
  std::istringstream raw_stream( message );
  const int64_t end_offset = message.size();

  // check for expected message type
  std::string msg_type;
  raw_stream >> msg_type;

  // number of bytes left in the buffer
  int64_t payload_size;
  raw_stream >> payload_size;

  std::cout << "Message tag: \"" << msg_type << "\"  Payload size in bytes: " << payload_size << std::endl;

  if ( payload_size != (end_offset - raw_stream.tellg()) )
  {
    std::cout << "WARNING: Payload size does not equal data count in stream.\n";
  }

  while ( true )
  {
    std::string element_name;
    std::string port_type;
    int64_t elem_size;

    raw_stream >> element_name >> port_type >> elem_size;
    raw_stream.get(); // eat delimiter

    std::cout << "   Element name: \"" << element_name
              << "\"  Port type: \"" << port_type
              << "\"  Element size: " << elem_size
              << std::endl;

    const int64_t current_remaining = end_offset - raw_stream.tellg();

    if ( elem_size > current_remaining )
    {
      std::cout << "Message size error. Current element size of "
                << elem_size << " bytes exceeds " << payload_size << " remaining bytes in the payload."
                << std::endl;
      return;
    }


    std::string elem_buffer( elem_size, 0 );
    raw_stream.read( &elem_buffer[0], elem_size );

    // Dump the serialized data element
    kwiver::vital::hex_dump( std::cout, elem_buffer.c_str(), elem_size );
    std::cout << std::endl;

    // check to see if we are at the end
    if ( elem_size == current_remaining )
    {
      break;
    }
  } // end while

}

// ----------------------------------------------------------------------------
void serializer_base::
dump_msg_spec()
{
  for ( const auto it : m_message_spec_list )
  {
    std::cout << "Message tag: " << it.first << std::endl;

    const auto& msg_spec = it.second;

    std::cout << "   Serialized port name: " << msg_spec.m_serialized_port_name << std::endl;

    for ( const auto elem_it : msg_spec.m_elements )
    {
      const auto& msg_elem = elem_it.second;

      std::cout << "    Msg element: " << elem_it.first << std::endl
                << "        Port name: " << msg_elem.m_port_name << std::endl
                << "        Port type: " << msg_elem.m_port_type << std::endl
                << "        Element name: " << msg_elem.m_element_name << std::endl
                << "        Algo name: " << msg_elem.m_algo_name << std::endl;
    } // end for

    std::cout << std::endl;

  } // end for

}



} // end namespace kwiver
