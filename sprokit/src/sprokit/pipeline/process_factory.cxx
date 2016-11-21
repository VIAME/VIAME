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

#include "process_factory.h"
#include "process_registry_exception.h"

namespace sprokit {

// ------------------------------------------------------------------
process_t
::create_process(process::type_t const& type,
                 process::name_t const& name,
                 kwiver::vital::config_block_sptr const& config) const
{
  if (!config)
  {
    throw null_process_registry_config_exception();
  }

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact_list = pm.get_factories( typeid( sprokit::process ).name() );

  auto const i = fact_list.find(type);

  if (i == d->registry.end())
  {
    throw no_such_process_type_exception(type);
  }

  config->set_value(process::config_type, kwiver::vital::config_block_value_t(type));
  config->set_value(process::config_name, kwiver::vital::config_block_value_t(name));

  return i->second.create_object(config);
}


// ------------------------------------------------------------------
void mark_process_as_loaded(module_t const& module)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  module_t mod = "process.";
  mod += module;
  vpm->mark_module_as_loaded( mod );
}


// ------------------------------------------------------------------
bool is_process_loaded(module_t const& module)
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  module_t mod = "process.";
  mod += module;

  return vpm->is_module_loaded( mod );
}

} // end namespace
