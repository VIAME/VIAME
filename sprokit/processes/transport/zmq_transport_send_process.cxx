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
#include <memory.h>

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( port, int, "5550",
                     "Port number to connect/bind to.");

create_config_trait( expected_subscribers, int, "1",
                     "Number of subscribers to wait for before starting to publish");

/**
 * \class zmq_transport_send_process
 *
 * \brief End cap that publishes incoming data on a ZeroMQ PUB Socket
 *
 * \process This process can be used to connect separate Sprokit pipelines
 * to one another in a publishi/subscribe framework.  This process
 * accepts serialized byte strings from serializer_process and transmits
 * them on a ZeroMQ PUB socket at the specified port.
 *
 * The process can be optionally configured to wait for an expected
 * number of subscribers before it begins publishing or it can begin
 * publishing immediately.  Use these options to control whether
 * subscribers see a consistent amount of data.  Waiting to publish
 * might be used when a data source is a directory of images, while
 * publishing immediately might be used when the data source is a
 * live video feed.
 *
 * The PUB/SUB mechanism makes use of two ports.  The actual
 * PUB/SUB socket pair is created on the port specified in
 * the configuration. One above that port is used to handle
 * subscription handshaking.
 *
 * \iports
 *
 * \iport{serialized_message} the incoming byte string to publish.
 *
 * \configs
 *
 * \config{port} the port number to publish on.
 *
 * \config{expected_subscribers} the number of subscribers that must
 * connect before publication commences.
 */

//----------------------------------------------------------------
// Private implementation class
class zmq_transport_send_process::priv
{
public:
  priv() ;
  ~priv();

  void connect();

  // Configuration values
  int m_port;
  int m_expected_subscribers;

  //+ any other connection related data goes here
  zmq::context_t m_context;
  zmq::socket_t m_pub_socket;
  zmq::socket_t m_sync_socket;

  vital::logger_handle_t m_logger; // for logging in priv methods

}; // end priv class

// ================================================================

zmq_transport_send_process
::zmq_transport_send_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new zmq_transport_send_process::priv )
{
  make_ports();
  make_config();

  d->m_logger = logger();
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
  d->m_port = config_value_using_trait( port );
  d->m_expected_subscribers = config_value_using_trait( expected_subscribers );

  int major, minor, patch;
  zmq_version(&major, &minor, &patch);
  LOG_DEBUG( logger(), "ZeroMQ Version: " << major << "." << minor << "." << patch );
}

// ----------------------------------------------------------------
void zmq_transport_send_process
::_init()
{
  d->connect();
}

// ----------------------------------------------------------------
void zmq_transport_send_process
::_step()
{
  auto mess = grab_from_port_using_trait( serialized_message );

  scoped_step_instrumentation();

  // We know that the message is a pointer to a std::string
  // send mess to the transport
  LOG_TRACE( logger(), "Sending datagram of size " << mess->size() );
  zmq::message_t datagram(mess->size());
  memcpy((void *) datagram.data(), (mess->c_str()), mess->size());
  d->m_pub_socket.send(datagram);
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
  declare_config_using_trait( port );
  declare_config_using_trait( expected_subscribers );
}


// ================================================================
zmq_transport_send_process::priv
::priv()
  : m_context( 1 )
  , m_pub_socket( m_context, ZMQ_PUB )
  , m_sync_socket( m_context, ZMQ_REP )
{
}


zmq_transport_send_process::priv
::~priv()
{
}


// ----------------------------------------------------------------------------
void
zmq_transport_send_process::priv
::connect()
{
  // Bind to the publisher socket
  std::ostringstream pub_connect_string;
  pub_connect_string << "tcp://*:" << m_port;
  LOG_TRACE( m_logger, "PUB Connect for " << pub_connect_string.str() );
  m_pub_socket.bind( pub_connect_string.str() );

  // Wait for replies from expected number of subscribers before sending antying
  std::ostringstream sync_connect_string;
  sync_connect_string << "tcp://*:" << ( m_port + 1 );
  LOG_TRACE( m_logger, "SYNC Connect for " << sync_connect_string.str() );
  m_sync_socket.bind( sync_connect_string.str() );

  int subscribers = 0;
  LOG_TRACE( m_logger, "Entering Sync Loop, waiting for "
             << m_expected_subscribers << " subscribers" );
  while ( subscribers < m_expected_subscribers )
  {
    zmq::message_t datagram;
    m_sync_socket.recv( &datagram );
    LOG_TRACE( m_logger, "SYNC Loop, received reply from subscriber "
               << subscribers << " of " << m_expected_subscribers );

    zmq::message_t datagram_o;
    m_sync_socket.send( datagram_o );
    ++subscribers;
  }
  LOG_TRACE( m_logger, "SYNC Loop done, all " << subscribers << " received" );
}

} // end namespace
