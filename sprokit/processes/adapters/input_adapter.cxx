// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for input_adapter class
 */

#include "input_adapter.h"
#include "input_adapter_process.h"

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>

namespace kwiver {

input_adapter
::input_adapter()
{ }

input_adapter
::~input_adapter()
{ }

// ------------------------------------------------------------------
void
input_adapter
::connect( sprokit::process::name_t proc, sprokit::pipeline_t pipe )
{

  // Find process in pipeline
  auto proc_ptr = pipe->process_by_name( proc ); // throws

  m_process = static_cast< kwiver::input_adapter_process* > ( proc_ptr.get() );
  m_interface_queue = m_process->get_interface_queue();
}

// ------------------------------------------------------------------
  sprokit::process::ports_t
input_adapter
::port_list() const
{
  return m_process->port_list();
}

// ------------------------------------------------------------------
adapter::ports_info_t
input_adapter
::get_ports() const
{
  return m_process->get_ports();
}

// ------------------------------------------------------------------
void
input_adapter
::send( kwiver::adapter::adapter_data_set_t dat )
{
  m_interface_queue->Send( dat );
}

// ------------------------------------------------------------------
bool
input_adapter
::full() const
{
  return m_interface_queue->Full();
}

} // end namespace
