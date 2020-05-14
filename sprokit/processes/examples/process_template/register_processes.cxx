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
  // The process registrar does all the hard work of registering the
  // process with the plugin loader.
  sprokit::process_registrar reg( vpm, "template_process" );

  // Check to see if module is already loaded. If so, then don't do again.
  if ( reg.is_module_loaded() )
  {
    return;
  }

  // ----------------------------------------------------------------
  // The process registrar registers the specified process type.
  reg.register_process< group_ns::template_process >();

  //++ Add more additional processes here.

// - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
