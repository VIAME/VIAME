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

/**
 * \file
 *
 * \brief implementation of the serializer process.
 */

#include "serializer_process.h"

#include <vital/util/string.h>
#include <vital/exceptions.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <sstream>

// ----------------------------------------------------------------------------
namespace kwiver {

/**
 * \class serializer_process
 *
 * \brief Process for serializing data elements into a byte string.
 *
 * \process Serialize incoming data elements into a byte stream. An
 * instance of this process can serialize multiple streams of input
 * data into output byte strings.
 *
 * \iports
 *
 * Input ports are dynamically created as needed. Port name must
 * have the format \iport{message/element}.
 *
 * \oports
 *
 * \oport{algo} Output ports are dynamically created as needed and
 * must correspond to the algo name from an input port.
 *
 * \code
 process ser :: serializer

 # select the serializing method to apply to all data elements.
 serialization_type = json

 # -- connect inputs to algo that generates image and mask messages
 connect from foo.image to ser.imgmask/image  # supplies image to imgmask message
 connect from bar.image to ser.imgmask/mask   # supplies mask to imgmask message

 # -- connect output to transport
 connect ser.imgmask to trans.message

 * \endcode
 *
 */

// (config-key, value-type, default-value, description )
create_config_trait( serialization_type, std::string, "",
                     "Specifies the method used to serialize the data object. "
                     "For example this could be \"json\", or \"protobuf\".");

create_config_trait( dump_message, bool, "false",
                     "Dump printable version of serialized messages of set to true." );

class serializer_process::priv
{
public:
  priv();
  ~priv();

  bool opt_dump_message;
};

// ============================================================================
serializer_process
::serializer_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
  , serializer_base( *this, logger() )
  , d( new priv )
{
  // This process manages its own inputs.
  this->set_data_checking_level( check_none );
  make_config();
}


serializer_process
::~serializer_process()
{
}


// ----------------------------------------------------------------
void
serializer_process
::_configure()
{
  scoped_configure_instrumentation();

  // Examine the configuration
  m_serialization_type = config_value_using_trait( serialization_type );
  d->opt_dump_message =  config_value_using_trait( dump_message );

  if (m_serialization_type.empty())
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "\"serialization_type\" config parameter not specified");
  }
}


// ------------------------------------------------------------------
// Post connection processing
void
serializer_process
::_init()
{
  base_init();

  // Now that we have a "normal" input ports let Sprokit manager them
  this->set_data_checking_level( check_valid );

  if (d->opt_dump_message)
  {
    dump_msg_spec();
  }
}


// ------------------------------------------------------------------
void
serializer_process
::_step()
{
  scoped_step_instrumentation();

  // Loop over all registered messages
  for ( auto msg_spec_it : m_message_spec_list )
  {
    // Each iteration of this loop assembles a multi-element message
    // with the following format
    //
    // message ::= <message-type> <size-of-payload> <element-list>
    // element_list ::= <element> <element_list>
    // element ::= <element-name> <port-type> <length> <serialized-bytes>
    //         |

    std::ostringstream message_str;
    const auto & msg_spec = m_message_spec_list[msg_spec_it.first];

    // Add message type string
    message_str << msg_spec_it.first << " ";

    std::ostringstream ser_elements;

    // loop over all elements that are part of this group.
    // This assembles the output byte string from one or more concrete data types.
    for ( auto msg_elem_it : msg_spec.m_elements )
    {
      const auto& element = msg_elem_it.second;
      LOG_TRACE( logger(), "Processing port: \"" << element.m_port_name
                 << "\" of type \"" << element.m_port_type << "\"" );

      // Convert input datum to a serialized message
      auto datum = grab_datum_from_port( element.m_port_name );
      auto local_datum = datum->get_datum<kwiver::vital::any>();
      std::shared_ptr< std::string > message;
      try
      {
        // Serialize the collected set of inputs
        message = element.m_serializer->serialize( local_datum );
      }
      catch ( const kwiver::vital::vital_exception& e )
      {
        // can be kwiver::vital::serialization_exception or kwiver::vital::bad_any_cast
        LOG_ERROR( logger(), "Error serializing data element \"" << element.m_element_name
                   << "\" for message type \"" << msg_spec_it.first << "\" : " << e.what() );
        break;
      }

      LOG_TRACE( logger(), "Adding element: \"" << element.m_element_name
                 << "\"  Port type: \"" << element.m_port_type << "\"  Size: "
                 << message->size() );

      if ( message->empty() )
      {
        LOG_WARN( logger(), "Serializer for message element \"" << element.m_element_name
                   << "\" for port name \"" << element.m_port_name
                   << "\" returned a null string. This is not expected." );
      }

      // Add element name and port type to output followed by serialized data
      ser_elements << element.m_element_name << " "
                   << element.m_port_type << " "
                   << message->size() << " "
                   << *message;
    } // end for

    // Add payload length to output buffer before the serialized elements.
    message_str << ser_elements.str().size()+1 << " " << ser_elements.str();

    if (d->opt_dump_message)
    {
      decode_message( message_str.str() );
    }

    // Push whole serialized message to port
    auto msg_buffer = std::make_shared< std::string >( message_str.str() );
    push_to_port_as < serialized_message_port_trait::type >( msg_spec.m_serialized_port_name, msg_buffer );

  } // end for

} // serializer_process::_step


// ----------------------------------------------------------------------------
void
serializer_process::
input_port_undefined(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    if ( vital_typed_port_info( port_name ) )
    {
      // Create input port
      port_flags_t required;
      required.insert( flag_required );

      LOG_TRACE( logger(), "Creating input port: \"" << port_name << "\"" );

      // Open an input port for the name
      declare_input_port(
        port_name,                  // port name
        type_flow_dependent,        // port_type
        required,
        port_description_t( "data type to be serialized" ) );
    }
  }
}

// ----------------------------------------------------------------------------
void
serializer_process::
output_port_undefined(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing undefined output port: \"" << port_name << "\"" );

  // Just create an output port
  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    if ( byte_string_port_info( port_name ) )
    {
      port_flags_t required;
      required.insert( flag_required );
      required.insert( flag_output_shared );

      LOG_TRACE( logger(), "Creating output port: \"" << port_name << "\"" );

      // Create output port
      declare_output_port(
        port_name,                                // port name
        serialized_message_port_trait::type_name, // port type
        required,                                 // port flags
        "serialized output" );
    }
  }
}

// ------------------------------------------------------------------
// Intercept input port connection so we can create the requested port
//
// port - name of the input port being connected.
//
// port_type - type of port
bool
serializer_process
::_set_input_port_type( port_t const&       port_name,
                        port_type_t const&  port_type )
{
  LOG_TRACE( logger(), "Processing input port: \"" << port_name
             << "\" of type \"" << port_type
             << "\""  );

  set_port_type( port_name, port_type );

  // pass to base class
  return process::_set_input_port_type( port_name, port_type );
}

// ----------------------------------------------------------------
void serializer_process
::make_config()
{
  declare_config_using_trait( serialization_type );
  declare_config_using_trait( dump_message );
}

// ------------------------------------------------------------------
serializer_process::priv
::priv()
  : opt_dump_message( false )
{
}

serializer_process::priv
::~priv()
{
}


} // end namespace sprokit
