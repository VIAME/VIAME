// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for output_adapter class
 */

#include "output_adapter.h"
#include "output_adapter_process.h"

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>

namespace kwiver {

output_adapter
::output_adapter()
{ }

output_adapter
::~output_adapter()
{ }

// ------------------------------------------------------------------
void
output_adapter
::connect( sprokit::process::name_t proc, sprokit::pipeline_t pipe )
{

  // Find process in pipeline
  sprokit::process_t proc_ptr = pipe->process_by_name( proc ); // throws

  m_process = static_cast< kwiver::output_adapter_process* > ( proc_ptr.get() );
  m_interface_queue = m_process->get_interface_queue();
}

// ------------------------------------------------------------------
  sprokit::process::ports_t
output_adapter
::port_list() const
{
  return m_process->port_list();
}

// ------------------------------------------------------------------
adapter::ports_info_t
output_adapter
::get_ports() const
{
  return m_process->get_ports();
}

// ------------------------------------------------------------------
kwiver::adapter::adapter_data_set_t
output_adapter
::receive()
{
  return m_interface_queue->Receive();
}

// ------------------------------------------------------------------
bool
output_adapter
::empty() const
{
  return m_interface_queue->Empty();
}

} // end namespace
