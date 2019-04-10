/*ckwg +30
 * Copyright 2019 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#include "kwiver_pipeline_viewer_export.h"

#include "pipeline_viewer.h"

#include <vital/plugin_loader/plugin_factory.h>
#include <vital/plugin_loader/plugin_loader.h>
#include <vital/plugin_loader/plugin_manager.h>

using std::string;

namespace {

static auto const module_name         = string{ "kwiver_pipeline_viewer" };
static auto const module_version      = string{ "1.0" };
static auto const module_organization = string{ "Kitware Inc." };

// ----------------------------------------------------------------------------
template < typename tool_t >
void
register_tool( kwiver::vital::plugin_loader& vpm,
               string const& version = module_version )
{
  using kvpf = kwiver::vital::plugin_factory;

  auto fact = vpm.ADD_APPLET( tool_t );
  fact->add_attribute( kvpf::PLUGIN_NAME,  tool_t::name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,  tool_t::description )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
    .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
    .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
  ;
}

} // namespace (anonymous)

// ----------------------------------------------------------------------------
extern "C"
KWIVER_PIPELINE_VIEWER_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  if ( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_tool< kwiver::tools::pipeline_viewer >( vpm );

  vpm.mark_module_as_loaded( module_name );
}
