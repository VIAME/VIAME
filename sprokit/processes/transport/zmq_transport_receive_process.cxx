// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "zmq_transport_receive_process.h"

#include <sprokit/pipeline/process_exception.h>

#include <kwiver_type_traits.h>

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( port, int, "5550",
                     "Port number to connect/bind to");

create_config_trait( connect_host, std::string, "localhost",
                     "Hostname (or IP address) to connect to." );

create_config_trait( num_publishers, int, "1",
                     "Number of publishers to subscribe to. ");

/**
 * \class zmq_transport_receive_process
 *
 * \brief End cap that subscribes to incoming data on a ZeroMQ SUB
 * Socket and outputs it on a Sprokit byte string port.
 *
 * \process This process can be used to connect separate Sprokit pipelines
 * to one another in a publishi/subscribe framework.  This process
 * accepts serialized byte strings from a ZeroMQ SUB socket and
 * pushes it to a serializer_process for deserialization into
 * a Sprokit pipeline.
 *
 * The PUB/SUB mechanism makes use of two ports.  The actual
 * PUB/SUB socket pair is created on the port specified in
 * the configuration. One above that port is used to handle
 * subscription handshaking.
 *
 * The process can be optionally configured to listen to multiple
 * publishers. It will connect to each publisher starting at
 * the specified port using port for PUB/SUB and port+1 for
 * synchronization.  A subsequent publisher will be connected
 * at port+2 and so on.
 *
 * \iports
 *
 * \iport{serialized_message} the incoming byte is sent to a
 * serializer process here.
 *
 * \configs
 *
 * \config{port} the port number to subscribe on.
 *
 * \config{num_publishers} the number of publishers that must
 * be connected to subscription commences.
 *
 * \config{connect_host} The name of the host to connect
 * to.  May be a DNS name or an IP address.
 */

//----------------------------------------------------------------
// Private implementation class
class zmq_transport_receive_process::priv
{
public:
  priv();
  ~priv();

  void connect();

  // Configuration values
  int m_port;
  int m_num_publishers;
  std::string m_connect_host;

  // any other connection related data goes here
  zmq::context_t m_context;
  zmq::socket_t m_sub_socket;
  std::vector< std::shared_ptr<zmq::socket_t> > m_sync_sockets;

  vital::logger_handle_t m_logger; // for logging in priv methods

}; // end priv class

// ================================================================

zmq_transport_receive_process
::zmq_transport_receive_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new zmq_transport_receive_process::priv )
{
  make_ports();
  make_config();
  d->m_logger = logger();
}

zmq_transport_receive_process
::~zmq_transport_receive_process()
{
}

// ----------------------------------------------------------------
void zmq_transport_receive_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get process config entries
  d->m_port = config_value_using_trait( port );
  d->m_num_publishers = config_value_using_trait( num_publishers );
  d->m_connect_host = config_value_using_trait( connect_host );

  int major, minor, patch;
  zmq_version(&major, &minor, &patch);
  LOG_DEBUG( logger(), "ZeroMQ Version: " << major << "." << minor << "." << patch );
}

// ----------------------------------------------------------------
void zmq_transport_receive_process
::_init()
{
  d->connect();
}

// ----------------------------------------------------------------
void zmq_transport_receive_process
::_step()
{
  LOG_TRACE( logger(), "Waiting for datagram..." );

  zmq::message_t datagram;
  d->m_sub_socket.recv(&datagram);

  auto msg = std::make_shared< std::string >(static_cast<char *>(datagram.data()), datagram.size());
  LOG_TRACE( logger(), "Received datagram of size " << msg->size() );

  // We know that the message is a pointer to a std::string
  push_to_port_using_trait( serialized_message, msg );
}

// ----------------------------------------------------------------
void zmq_transport_receive_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_output_port_using_trait( serialized_message, required );
}

// ----------------------------------------------------------------
void zmq_transport_receive_process
::make_config()
{
  declare_config_using_trait( port );
  declare_config_using_trait( num_publishers );
  declare_config_using_trait( connect_host );
}

// ================================================================
zmq_transport_receive_process::priv
::priv()
  : m_context( 1 )
  , m_sub_socket( m_context, ZMQ_SUB )
{
}

zmq_transport_receive_process::priv
::~priv()
{
}

// ----------------------------------------------------------------------------
void
zmq_transport_receive_process::priv
::connect()
{
  LOG_DEBUG( m_logger, "Number of publishers " << m_num_publishers );
  m_sub_socket.setsockopt(ZMQ_SUBSCRIBE,"",0);

  // We start with our base port.  Even ports are the pub/sub socket
  // Odd ports (pub/sub + 1) are the sync sockets
  for ( int i = 0; i < m_num_publishers * 2; i+=2 )
  {
    std::shared_ptr< zmq::socket_t > sync_socket = std::make_shared< zmq::socket_t >( m_context, ZMQ_REQ );
    m_sync_sockets.push_back(sync_socket);

    std::ostringstream sub_connect_string;
    sub_connect_string << "tcp://" << m_connect_host << ":" << m_port + i;
    LOG_TRACE( m_logger, "SUB Connect for " << sub_connect_string.str() );
    m_sub_socket.connect( sub_connect_string.str() );

    std::ostringstream sync_connect_string;
    sync_connect_string << "tcp://" << m_connect_host << ":" << ( m_port + i + 1);
    LOG_TRACE( m_logger, "SYNC Connect for " << sync_connect_string.str() );
    sync_socket->connect( sync_connect_string.str() );

    zmq::message_t datagram;
    sync_socket->send(datagram);

    zmq::message_t datagram_i;
    LOG_TRACE( m_logger, "Waiting for SYNC reply, pub: " << i << " at " << m_port + i + 1 );
    sync_socket->recv(&datagram_i);
    LOG_TRACE( m_logger, "SYNC reply received, pub: " << i << " at " <<  m_port + i + 1 );
  }
}

} // end namespace
