// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "input_adapter_process.h"

#include <stdexcept>
#include <sstream>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * \class input_adapter_process
 *
 * \brief Generic input adapter process.
 *
 * This process is a generic source process in a pipeline. It takes
 * input values from an external source and feeds them into the
 * specified ports of the pipeline. The name of the pipeline ports are
 * discovered dynamically as they are configured.
 *
 * The main processing loop (_step()) takes sets of data from the
 * input queue connected to the application side input_adapter
 * object. The elements in a data set are sent to the named output
 * port.
 *
 * \oports
 *
 * \oport{various} Output ports are created dynamically based on
 * pipeline connections.
 */

// ------------------------------------------------------------------
input_adapter_process
::input_adapter_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
{
}

input_adapter_process
::~input_adapter_process()
{ }

// ------------------------------------------------------------------
kwiver::adapter::ports_info_t
input_adapter_process
::get_ports()
{
  kwiver::adapter::ports_info_t p_info;

  // formulate list of current input ports
  auto ports = this->output_ports();
  for( auto port : ports )
  {
    p_info[port] = this->input_port_info( port );
  }

  return p_info;
}

// ------------------------------------------------------------------
void
input_adapter_process
::output_port_undefined( sprokit::process::port_t const& port )
{
  // If we have not created the port, then make a new one.
  if ( m_active_ports.count( port ) == 0 )
  {
    port_flags_t p_flags;
    p_flags.insert( flag_required );

    if ( port[0] != '_' ) // skip special ports (e.g. _heartbeat)
    {
      LOG_TRACE( logger(), "Creating output port: \"" << port << "\" on process \"" << name() << "\"" );

      // create a new port
      declare_output_port( port, // port name
                           type_any, // port type
                           p_flags,
                           port_description_t("Output for " + port)
        );

      // Add to our list of existing ports
      m_active_ports.insert( port );
    }
  }
}

// ------------------------------------------------------------------
void
input_adapter_process
::_step()
{
  LOG_TRACE( logger(), "Processing data set" );
  auto data_set = this->get_interface_queue()->Receive(); // blocking call
  std::set< sprokit::process::port_t > unused_ports = m_active_ports; // copy set of active ports

  // Handle end of input as last data supplied.
  if (data_set->is_end_of_data() )
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );

    // indicate done
    mark_process_as_complete();
    const auto dat( sprokit::datum::complete_datum() );

    for( auto p : m_active_ports )
    {
      // Push each datum to their port
      push_datum_to_port( p, dat );
    }
    return;
  }

  // We have real data to send down the pipeline.
  // Need to assure that all defined ports have a datum, and
  // there are no unconnected ports specified.

  for ( auto ix : *data_set )
  {
    // validate the port name against our list of created ports
    if ( m_active_ports.count( ix.first ) == 1 )
    {
      // Push each datum to their port
      this->push_datum_to_port( ix.first, ix.second );

      // Remove this port from set so we know data has been sent.
      // Any remaining names are considered unused.
      unused_ports.erase( ix.first );
    }
    else
    {
      std::stringstream str;
      str << "Process " << name() << ": Unconnected port \"" << ix.first << "\" specified in data packet. ";
      throw std::runtime_error( str.str() );
    }
  } // end for

  // check to see if all ports have been supplied with a datum
  if ( unused_ports.size() != 0 )
  {
    for ( auto port : unused_ports )
    {
      LOG_ERROR( logger(), "Process: " << name() << ": Port \"" << port
                 << "\" has not been supplied with a datum");
    } // end for

    throw std::runtime_error( "Process " + name() + " has not been supplied data for all ports." );
  }

  return;
}

} // end namespace
