/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#include <sprokit/pipeline/process_factory.h>

#include "collate_process.h"
#include "distribute_process.h"
#include "pass_process.h"
#include "sink_process.h"

/**
 * \file flow/registration.cxx
 *
 * \brief Register processes for use.
 */

extern "C"
SPROKIT_PROCESSES_FLOW_EXPORT
void
register_factories( kwiver::vital::plugin_manager& vpm )
{
  static process_registry::module_t const module_name = process_registry::module_t("flow_processes");

  process_registry_t const registry = process_registry::self();

  if (sprokit::is_process_loaded(module_name))
  {
    return;
  }

  fact = vpm.ADD_PROCESS( collate_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME, "collate" );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Collates data from multiple worker processes" );

  fact = vpm.ADD_PROCESS( distribute_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME, "distribute" );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Distributes data to multiple worker processes" );

  fact = vpm.ADD_PROCESS( pass_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME, "pass" );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Pass a data stream through" );

  fact = vpm.ADD_PROCESS( sink_process );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_NAME, "sink" );
  fact->add_attribute(  kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Ignores incoming data" );


  sprokit::mark_process_moduleas_loaded(module_name);
}
