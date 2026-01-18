/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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

/**
 * \file
 * \brief CppDB plugin algorithm registration interface impl
 */

#include "viame_cppdb_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/query_track_descriptor_set.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/write_object_track_set.h>
#include <vital/algo/write_track_descriptor_set.h>

#include "write_object_track_set_db.h"
#include "write_track_descriptor_set_db.h"
#include "query_track_descriptor_set_db.h"
#include "read_object_track_set_db.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_CPPDB_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.cppdb";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::write_object_track_set, write_object_track_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::write_track_descriptor_set, write_track_descriptor_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::query_track_descriptor_set, query_track_descriptor_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::read_object_track_set, read_object_track_set_db >(
    "db" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
