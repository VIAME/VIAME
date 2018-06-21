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

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <sstream>

// ----------------------------------------------------------------------------
namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( serialization_type, std::string, "",
                     "Specifies the method used to serialize the data object. "
                     "For example this could be json, or protobuf.");

class deserializer_process::priv
{
public:
  priv();
  ~priv();
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
}


// ------------------------------------------------------------------
// Post connection processing
void
deserializer_process
::_init()
{
  base_init();
}


// ------------------------------------------------------------------
void
deserializer_process
::_step()
{
  scoped_step_instrumentation();

  // Loop over all registered groups
  for (const auto elem : m_port_group_list )
  {
    const auto & pg = elem.second;

    // Convert input back to vital type
    auto message = grab_from_port_as< serialized_message_port_trait::type >( pg.m_serialized_port_name );
    auto deser = pg.m_serializer->deserialize( message );

    // loop over all items that are part of this group.
    // This disassembles the input byte string into one or more concrete data types.
    for ( auto pg_item : pg.m_items )
    {
      auto & item = pg_item.second;
      LOG_TRACE( logger(), "Processing port: \"" << item.m_port_name
                 << "\" of type \"" << item.m_port_type << "\"" );

      // Push one element from the deserializer to the correct port
      sprokit::datum_t local_datum = sprokit::datum::new_datum( deser[ item.m_element_name ] );
      push_datum_to_port( item.m_port_name, local_datum );
    }
  }
} // deserializer_process::_step


// ----------------------------------------------------------------------------
sprokit::process::port_info_t
deserializer_process::
_input_port_info(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

  // Just create an input port to read byte string from
  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    byte_string_port_info( port_name );

    port_flags_t required;
    required.insert( flag_required );

    // Create output port
    declare_input_port(
      port_name,                                // port name
      serialized_message_port_trait::type_name, // port type
      required,                                 // port flags
      "serialized input" );
  }

  return process::_input_port_info( port_name );
}


// ----------------------------------------------------------------------------
sprokit::process::port_info_t
deserializer_process::
_output_port_info(port_t const& port_name)
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

  return process::_output_port_info( port_name );
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
} // deserializer_process::_output_port_info


// ----------------------------------------------------------------
void deserializer_process
::make_config()
{
  declare_config_using_trait( serialization_type );
}

// ------------------------------------------------------------------
deserializer_process::priv
::priv()
{
}

deserializer_process::priv
::~priv()
{
}

} // end namespace sprokit
