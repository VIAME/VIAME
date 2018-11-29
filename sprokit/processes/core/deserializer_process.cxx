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
 * \brief implementation of the deserializer process.
 */

#include "deserializer_process.h"

#include <vital/algo/data_serializer.h>
#include <vital/util/string.h>
#include <vital/exceptions.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <sstream>
#include <cstdint>

// ----------------------------------------------------------------------------
namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( serialization_type, std::string, "",
                     "Specifies the method used to serialize the data object. "
                     "For example this could be \"json\", or \"protobuf\".");

create_config_trait( dump_message, bool, "false",
                     "Dump printable version of serialized messages of set to true." );

class deserializer_process::priv
{
public:
  priv();
  ~priv();

  bool opt_dump_message;
};

// ============================================================================
deserializer_process
::deserializer_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
  , serializer_base( *this, logger() )
  , d( new priv )
{
  // This process manages its own inputs.
  this->set_data_checking_level( check_none );
  make_config();
}


deserializer_process
::~deserializer_process()
{
}


// ----------------------------------------------------------------
void
deserializer_process
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
deserializer_process
::_init()
{
  base_init();

  // Now that we have a "normal" output port, let Sprokit manage it
  this->set_data_checking_level( check_valid );

  if (d->opt_dump_message)
  {
    dump_msg_spec();
  }
}


// ------------------------------------------------------------------
void
deserializer_process
::_step()
{
  scoped_step_instrumentation();

  // Loop over all registered groups
  for (const auto msg_spec_it : m_message_spec_list )
  {
    // Each iteration of this loop expects a multi-element message
    // with the following format
    //
    // message ::= <message-type> <size-of-payload> <element-list>
    // element_list ::= <element> <element_list>
    // element ::= <element-name> <port-type> <length> <serialized-bytes>
    //         |

    try
    {
      const auto& msg_spec = msg_spec_it.second;

      // Convert input back to vital type
      auto message = grab_from_port_as< serialized_message_port_trait::type >( msg_spec.m_serialized_port_name );

      std::istringstream raw_stream( *message );
      const int64_t end_offset = message->size();
      int64_t current_remaining;

      if (d->opt_dump_message)
      {
        decode_message( *message );
      }

      // check for expected message type
      std::string msg_type;
      raw_stream >> msg_type;

      // This must be the expected message type.
      if ( msg_type != msg_spec_it.first )
      {
        LOG_ERROR( logger(), "Unexpected message type. Expected \""
                   << msg_spec_it.first << "\" but received \"" << msg_type << "\". Message dropped." );
        return;
      }

      // number of bytes left in the buffer
      int64_t payload_size;
      raw_stream >> payload_size;

      if ( payload_size != (end_offset - raw_stream.tellg()) )
      {
        LOG_WARN( logger(), "Payload size does not equal data count in stream." );
      }

      // Loop over all elements in the message
      for ( size_t i = 0; i < msg_spec.m_elements.size(); ++i )
      {
        std::string element_name;
        std::string port_type;

        raw_stream >> element_name >> port_type;

        // Find corresponding entry for element_name
        if ( msg_spec.m_elements.count( element_name ) == 0 )
        {
          LOG_ERROR( logger(), "Message component \"" << element_name
                     << "\" not specified for this process. Message dropped. " );
          break;
        }

        // Get element specification
        const auto& msg_elem = msg_spec.m_elements.at( element_name );

        LOG_TRACE( logger(), "Processing port: \"" << msg_elem.m_port_name
                   << "\" of type \"" << msg_elem.m_port_type << "\"" );

        if ( port_type != msg_elem.m_port_type )
        {
          LOG_ERROR( logger(), "Message element type mismatch in message type \""
                     << msg_type << "\". Expecting element type \""
                     << msg_elem.m_port_type << "\" but received \""
                     << port_type << "\". Remaining message dropped." );
          break;
        }

        // Get size of next element
        int64_t elem_size;
        raw_stream >> elem_size;
        raw_stream.get(); // eat delimiter

        current_remaining = end_offset - raw_stream.tellg();
        if ( elem_size > current_remaining )
        {
          LOG_ERROR( logger(), "Message size error. Current element size of "
                     << elem_size << " bytes exceeds " << current_remaining
                     << " remaining bytes in the payload_size. Remaining message dropped" );
        break;
        }

        std::string elem_buffer( elem_size, 0 );
        // raw_stream.read( elem_buffer.data(), elem_size );
        raw_stream.read( &elem_buffer[0], elem_size );

        current_remaining = end_offset - raw_stream.tellg();

        // deserialize the data
        auto deser_data = msg_elem.m_serializer->deserialize( elem_buffer );

        // test for empty any()
        if ( deser_data.empty() )
        {
          LOG_ERROR( logger(), "Deserializer for type \"" << msg_elem.m_port_type
                     << "\" from message \"" << msg_type
                     << "\" returned a null result. Whole message dropped." );
          break;
        }

        // Push one element from the deserializer to the correct port
        sprokit::datum_t local_datum = sprokit::datum::new_datum( deser_data );
        push_datum_to_port( msg_elem.m_port_name, local_datum );
      } // end for

      // test residual payload size - should be zero
      if ( current_remaining > 0 )
      {
        LOG_ERROR( logger(), "Message contained more elements than expected. Message processed with "
                   << current_remaining << " bytes remaining." );
      }

    }
    catch ( const kwiver::vital::vital_exception& e )
    {
      // can be kwiver::vital::serialization_exception or kwiver::vital::bad_any_cast
      LOG_ERROR( logger(), "Error deserializing data element: " << e.what() );
    }
  } // end for
} // deserializer_process::_step


// ----------------------------------------------------------------------------
void
deserializer_process::
input_port_undefined(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

  // Just create an input port to read byte string from
  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    if ( byte_string_port_info( port_name ) )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
        port_name,                                // port name
        serialized_message_port_trait::type_name, // port type
        required,                                 // port flags
        "serialized input" );
    }
  }
}


// ----------------------------------------------------------------------------
void
deserializer_process::
output_port_undefined(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing output port info: \"" << port_name << "\"" );

  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    if ( vital_typed_port_info( port_name ) )
    {
      // Create input port
      port_flags_t required;
      required.insert( flag_required );

      LOG_TRACE( logger(), "Creating input port: \"" << port_name << "\"" );

      // Open an input port for the name
      declare_output_port(
        port_name,                  // port name
        type_flow_dependent,        // port_type
        required,
        port_description_t( "deserialized data type" ) );
    }
  }
}


// ------------------------------------------------------------------
// Intercept input port connection so we can create the requested port
//
// port - name of the input port being connected.
//
// port_type - type of port to create
//
bool
deserializer_process
::_set_output_port_type( port_t const&      port_name,
                         port_type_t const& port_type )
{
  LOG_TRACE( logger(), "Processing output port: \""  << port_name
             << "\" of type \"" << port_type
             << "\""  );

  set_port_type( port_name, port_type );

  // pass to base class
  return process::_set_output_port_type( port_name, port_type );
}


// ----------------------------------------------------------------
void deserializer_process
::make_config()
{
  declare_config_using_trait( serialization_type );
  declare_config_using_trait( dump_message );
}

// ------------------------------------------------------------------
deserializer_process::priv
::priv()
  : opt_dump_message( false )
{
}

deserializer_process::priv
::~priv()
{
}

} // end namespace sprokit
