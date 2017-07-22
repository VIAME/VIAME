/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
