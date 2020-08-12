/*ckwg +29
 * Copyright 2011-2018, 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include <sprokit/processes/flow/processes_flow_export.h>

#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "collate_process.h"
#include "distribute_process.h"
#include "pass_process.h"
#include "sink_process.h"
#include "mux_process.h"

/**
 * \file flow/registration.cxx
 *
 * \brief Register processes for use.
 */

extern "C"
PROCESSES_FLOW_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  sprokit::process_registrar reg( vpm, "flow_processes" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< sprokit::collate_process>();
  reg.register_process< sprokit::distribute_process>();
  reg.register_process< sprokit::pass_process>();
  reg.register_process< sprokit::sink_process>();
  reg.register_process< sprokit::mux_process>();

  // - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
