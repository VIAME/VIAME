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

#include "zmq_transport_send_process.h"

#include <sprokit/pipeline/process_exception.h>

#include <kwiver_type_traits.h>

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( port_name, std::string, "some-port",
                     "Name of port where serialized messages are sent.");


//----------------------------------------------------------------
// Private implementation class
class zmq_transport_send_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_port_name;

  //+ any other connection related data goes here

}; // end priv class

// ================================================================

zmq_transport_send_process
::zmq_transport_send_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new zmq_transport_send_process::priv )
{
  make_ports();
  make_config();
}


zmq_transport_send_process
::~zmq_transport_send_process()
{
}


// ----------------------------------------------------------------
void zmq_transport_send_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get process config entries
  d->m_port_name = config_value_using_trait( port_name );

  // connect to port here or do connect in _init() method if this time
  // is too soon
}


// ----------------------------------------------------------------
void zmq_transport_send_process
::_step()
{
  auto mess = grab_from_port_using_trait( serialized_message );

  scoped_step_instrumentation();

  // We know that the message is a pointer to a std::string
  //+ send mess to the transport
}


// ----------------------------------------------------------------
void zmq_transport_send_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( serialized_message, required );
}


// ----------------------------------------------------------------
void zmq_transport_send_process
::make_config()
{
  declare_config_using_trait( port_name );
}


// ================================================================
zmq_transport_send_process::priv
::priv()
{
}


zmq_transport_send_process::priv
::~priv()
{
}

} // end namespace
