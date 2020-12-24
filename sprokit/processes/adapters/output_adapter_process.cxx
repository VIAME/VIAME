// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for output_adapter_process class.
 */

#include "output_adapter_process.h"

#include <kwiver_type_traits.h>

#include <stdexcept>
#include <sstream>

namespace kwiver {

/**
 * \class output_adapter_process
 *
 * \brief Generic output adapter class
 *
 * \iports
 * \iport{various}
 * Input ports are dynamically created based on pipeline connections.
 */

create_config_trait( wait_on_queue_full, bool, "TRUE",
                     "When the output queue back to the application is full and there is more data to add, "
                     "should new data be dropped or should the pipeline block until the data can be delivered. "
                     "The default action is to wait until the data can be delivered." );

//----------------------------------------------------------------
// Private implementation class
class output_adapter_process::priv
{
public:
  priv();

  ~priv();

  bool m_wait_on_queue_full;

}; // end priv class

// ------------------------------------------------------------------
output_adapter_process
::output_adapter_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
  , d( new output_adapter_process::priv )
{
  declare_config_using_trait( wait_on_queue_full );
}

output_adapter_process
::~output_adapter_process()
{ }

// ------------------------------------------------------------------
void
output_adapter_process::
_configure()
{
  d->m_wait_on_queue_full = config_value_using_trait( wait_on_queue_full );

}

// ------------------------------------------------------------------
kwiver::adapter::ports_info_t
output_adapter_process
::get_ports()
{
  kwiver::adapter::ports_info_t p_info;

  // formulate list of current output ports
  auto ports = this->output_ports();
  for( auto port : ports )
  {
    p_info[port] = this->output_port_info( port );
  }

  return p_info;
}

// ------------------------------------------------------------------
void
output_adapter_process
::input_port_undefined(port_t const& port)
{
  // If we have not created the port, then make a new one.
  if ( m_active_ports.count( port ) == 0 )
  {
    port_flags_t p_flags;
    p_flags.insert( flag_required );

    LOG_TRACE( logger(), "Creating input port: \"" << port << "\" on process \"" << name() << "\"" );

    // create a new port
    declare_input_port( port, // port name
                        type_any, // port data type expected
                        p_flags,
                        port_description_t("Input for " + port)
      );

    // Add to our list of existing ports
    m_active_ports.insert( port );
  }
}

// ------------------------------------------------------------------
void
output_adapter_process
::_step()
{
  LOG_TRACE( logger(), "Processing data set" );

  auto data_set = kwiver::adapter::adapter_data_set::create();

  // The grab call is blocking, so it will wait until data is there.
  for( auto const p : m_active_ports )
  {
    LOG_TRACE( logger(), "Getting data from port \"" << p <<"\"" );

    auto dtm = this->grab_datum_from_port( p );
    data_set->add_datum( p, dtm );
  } // end foreach

  // If we are willing to wait for room in queue
  // or the queue is not full, then send data to application.
  if ( d->m_wait_on_queue_full || ! this->get_interface_queue()->Full() )
  {
    // Send received data to consumer thread
    this->get_interface_queue()->Send( data_set );
  }
}

// ----------------------------------------------------------------
void
output_adapter_process
::_finalize()
{
  LOG_DEBUG( logger(), "End of data detected." );

  // Send end of input into interface queue indicating no more data will be sent.
  auto ds = kwiver::adapter::adapter_data_set::create( kwiver::adapter::adapter_data_set::end_of_input );
  this->get_interface_queue()->Send( ds );
}

// ================================================================
output_adapter_process::priv
::priv()
  : m_wait_on_queue_full( true )
{
}

output_adapter_process::priv
::~priv()
{
}

} // end namespace
