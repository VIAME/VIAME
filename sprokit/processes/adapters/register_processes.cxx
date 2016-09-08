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

#include <sprokit/processes/adapters/kwiver_adapter_processes_export.h>
#include <sprokit/pipeline/process_registry.h>

#include "input_adapter_process.h"
#include "output_adapter_process.h"

// -- list processes to register --
extern "C"
KWIVER_ADAPTER_PROCESSES_EXPORT void register_processes();


// ----------------------------------------------------------------
/*! \brief Regsiter processes
 *
 *
 */
void register_processes()
{
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "kwiver_processes_adapters" );

  sprokit::process_registry_t const registry( sprokit::process_registry::self() );

  if ( registry->is_module_loaded( module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------
  registry->register_process(
    "input_adapter",
    "Source process for pipeline. Pushes data items into pipeline ports. "
    "Ports are dynamically created as needed based on connections specified in the pipeline file.",
    sprokit::create_process< kwiver::input_adapter_process > );

  registry->register_process(
    "output_adapter",
    "Sink process for pipeline. Accepts data items from pipeline ports. "
    "Ports are dynamically created as needed based on connections specified in the pipeline file.",
    sprokit::create_process< kwiver::output_adapter_process > );


  // - - - - - - - - - - - - - - - - - - - - - - -
  registry->mark_module_as_loaded( module_name );
}
