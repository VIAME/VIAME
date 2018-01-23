/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include <sprokit/pipeline/process_factory.h>

// -- list processes to register --
#include "template_process.h"
//++ list additional processes here

// ----------------------------------------------------------------
/** \brief Regsiter processes
 *
 *
 */
extern "C"   //++ This needs to have 'C' linkage so the loader can find it.
TEMPLATE_PROCESSES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  // The module name identifies this loadable collection or
  // processes. The name must be unique across all loadable modules,
  // since we check to see if it is already loaded.
  static const auto module_name = kwiver::vital::plugin_manager::module_t( "template_processes" ); //++ <- replace with real name of module

  // Check to see if module is already loaded. If so, then don't do again.
  if ( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------

  auto fact = vpm.ADD_PROCESS( group_ns::template_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "template" ) //+ use your process name
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Description of process. Make as long as necessary to fully explain what the process does "
                    "and how to use it. Explain specific algorithms used, etc." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  //++ Add more additional processes here.

// - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
