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

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <sstream>

// ----------------------------------------------------------------------------
namespace kwiver {

/**
 * \class serializer_process
 *
 * \brief Process for serializing data items into a byte string.
 *
 * \process Serialize incoming data items into a byte stream. An
 * instance of this process can serialize multiple streams of input
 * data into output byte strings.
 *
 * \iports
 *
 * Input ports are dynamically created as needed. Port name must
 * have the format \iport{algo/item} or \iport{algo}.
 *
 * \oports
 *
 * \oport{algo} Output ports are dynamically created as needed and
 * must correspond to the algo name from an input port.
 *
 * The "algo" part of a port name must correspond to a loadable
 * serializer algorithm. The "item" part of the input port name must
 * correspond to an element name expected by the serializer instance.
 *
 * \code
 process ser :: serializer

 # select the serializing method to apply to all data items.
 serialization_type = json

 # -- connect inputs to algo that generates image and mask messages
 connect from foo.image to ser.ImageAndMask/image  # supplies image to ImageAndMask implementation
 connect from bar.image to ser.ImageAndMask/mask   # supplies mask to ImageAndMask implementation

 # -- connect output to transport
 connect ser.ImageAndMask to trans.message

 # -- connect a single input algorithm
 connect from foo.detections to ser.detections  # connect detection set to be serialized
 connect from ser.detections to det_trans.message
 * \endcode
 *
 */

// (config-key, value-type, default-value, description )
create_config_trait( serialization_type, std::string, "",
                     "Specifies the method used to serialize the data object. "
                     "For example this could be json, or protobuf.");

class serializer_process::priv
{
public:
  priv();
  ~priv();
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
}


// ------------------------------------------------------------------
// Post connection processing
void
serializer_process
::_init()
{
  base_init();
}


// ------------------------------------------------------------------
void
serializer_process
::_step()
{
  scoped_step_instrumentation();

  // Loop over all registered groups
  for (const auto elem : m_port_group_list )
  {
    const auto & pg = elem.second;

    vital::algo::data_serializer::serialize_param_t sp;

    // loop over all items that are part of this group.
    // This assembles the output byte string from one or more concrete data types.
    for ( auto pg_item : pg.m_items )
    {
      auto & item = pg_item.second;
      LOG_TRACE( logger(), "Processing port: \"" << item.m_port_name
                 << "\" of type \"" << item.m_port_type << "\"" );

      // Convert input datum to a serialized message
      auto datum = grab_datum_from_port( item.m_port_name );
      auto local_datum = datum->get_datum<kwiver::vital::any>();
      sp[ item.m_element_name ] = local_datum;
    }

    // Serialize the collected set of inputs
    auto message = pg.m_serializer->serialize( sp );
    push_to_port_as < serialized_message_port_trait::type >( pg.m_serialized_port_name, message );
  }

} // serializer_process::_step


// ----------------------------------------------------------------------------
sprokit::process::port_info_t
serializer_process::
_input_port_info(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

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

  return process::_input_port_info( port_name );
}

// ----------------------------------------------------------------------------
sprokit::process::port_info_t
serializer_process::
_output_port_info(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing output port info: \"" << port_name << "\"" );

  // Just create an output port
  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    if ( byte_string_port_info( port_name ) )
    {
      port_flags_t required;
      required.insert( flag_required );

      LOG_TRACE( logger(), "Creating output port: \"" << port_name << "\"" );

      // Create output port
      declare_output_port(
        port_name,                                // port name
        serialized_message_port_trait::type_name, // port type
        required,                                 // port flags
        "serialized output" );
    }
  }

  return process::_output_port_info( port_name );
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
} // serializer_process::_input_port_info

// ----------------------------------------------------------------
void serializer_process
::make_config()
{
  declare_config_using_trait( serialization_type );
}

// ------------------------------------------------------------------
serializer_process::priv
::priv()
{
}

serializer_process::priv
::~priv()
{
}


} // end namespace sprokit
