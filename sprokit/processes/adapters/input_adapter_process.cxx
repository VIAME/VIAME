/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
  kwiver::adapter::ports_info_t port_info;

  // formulate list of current input ports
  auto ports = this->output_ports();
  for( auto port : ports )
  {
    port_info[port] = this->input_port_info( port );
  }

  return port_info;
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
