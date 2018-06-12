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

  // The canonical name string defining the data type we are converting.
  std::string m_serialization_type;

  /*
   * This class represents the serialization needed for an
   * input/output port pair.
   */
  class port_handler
  {
  public:
    port_t m_port_name;
    port_type_t m_port_type;

    kwiver::vital::algo::data_serializer_sptr m_serializer;
  };

  // map is indexed by port name
  std::map< std::string, port_handler > m_port_handler_list;

};

// ============================================================================
deserializer_process
::deserializer_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
  d( new priv )
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
  d->m_serialization_type = config_value_using_trait( serialization_type );
}


// ------------------------------------------------------------------
// Post connection processing
void
deserializer_process
::_init()
{
}


// ------------------------------------------------------------------
void
deserializer_process
::_step()
{
  scoped_step_instrumentation();

  // Loop over all registered ports
  for (const auto elem : d->m_port_handler_list )
  {
    const auto & ph = elem.second;

    LOG_TRACE( logger(), "Processing port: \"" << ph.m_port_name
               << "\" of type \"" << ph.m_port_type << "\"" );

    // Convert input back to vital type
    auto message = grab_from_port_as< serialized_message_port_trait::type >( ph.m_port_name );
    vital::any local_type = ph.m_serializer->deserialize( message );
    sprokit::datum_t local_datum = sprokit::datum::new_datum( local_type );
    push_datum_to_port( ph.m_port_name, local_datum );
  }

} // deserializer_process::_step


// ----------------------------------------------------------------------------
sprokit::process::port_info_t
deserializer_process::
_input_port_info(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    init_port_handler( port_name );
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
    init_port_handler( port_name );
  }

  return process::_output_port_info( port_name );
}


// ------------------------------------------------------------------
// Intercept input port connection so we can create the requested port
//
// port - name of the input port being connected.
//
// port_type - type of port to create
bool
deserializer_process
::_set_output_port_type( port_t const&      port_name,
                         port_type_t const& port_type )
{
  LOG_TRACE( logger(), "Processing output port: \""  << port_name
             << "\" of type \"" << port_type
             << "\""  );

  // update port handler
  priv::port_handler& ph = d->m_port_handler_list[port_name];

  ph.m_port_type = port_type;

  // create config items
  // serialize-protobuf:type = <data_type>
  // serialize-protobuf:<data type>:foo = bar // possible but not likely

  if ( ! ph.m_serializer )
  {
    auto algo_config = kwiver::vital::config_block::empty_config();
    const std::string ser_algo_type( "serialize-" + d->m_serialization_type );
    const std::string ser_type( ser_algo_type + vital::config_block::block_sep + "type" );

    algo_config->set_value( ser_type, ph.m_port_type );

    std::stringstream str;
    algo_config->print( str );
    LOG_TRACE( logger(), "Creating algorithm for (config block):\n"
               << str.str() << std::endl );

    vital::algorithm_sptr base_nested_algo;

    // create serialization algorithm
    vital::algorithm::set_nested_algo_configuration( ser_algo_type, // data type name
                                                     ser_algo_type, // config blockname
                                                     algo_config,
                                                     base_nested_algo );

    ph.m_serializer = std::dynamic_pointer_cast< vital::algo::data_serializer > ( base_nested_algo );
    if ( ! ph.m_serializer )
    {
      VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                   "Unable to create serializer for type \"" +
                   ph.m_port_type  + "\" for " + d->m_serialization_type );
    }

    if ( ! vital::algorithm::check_nested_algo_configuration( ser_algo_type,
                                                              ser_algo_type,
                                                              algo_config ) )
    {
      throw sprokit::invalid_configuration_exception(
              name(), "Configuration check failed." );
    }
  }

  // pass to base class
  return process::_set_output_port_type( port_name, ph.m_port_type );
} // deserializer_process::_output_port_info


// ----------------------------------------------------------------
void deserializer_process
::make_config()
{
  declare_config_using_trait( serialization_type );
}

// ----------------------------------------------------------------------------
void deserializer_process
::init_port_handler( port_t port_name )
{
  // if port has already been added, do nothing
  if (d->m_port_handler_list.count( port_name ) > 0 )
  {
    return;
  }

    // create new port handler
  priv::port_handler ph;
  ph.m_port_name = port_name;

  d->m_port_handler_list[port_name] = ph;

  port_flags_t required;
  required.insert( flag_required );

  LOG_TRACE( logger(), "Creating input & output port: \"" << port_name << "\"" );

  // Create input port
  declare_input_port(
    port_name,                                // port name
    serialized_message_port_trait::type_name, // port type
    required,                                 // port flags
    "serialized input" );

  // Open an output port for the name
  declare_output_port(
    port_name,                  // port name
    type_flow_dependent,        // port_type
    required,
    port_description_t( "deserialized data type" ) );

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
