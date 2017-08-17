/*ckwg +29
 * Copyright 2011-2016 by Kitware, Inc.
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
 * \file examples/registration.cxx
 *
 * \brief Register processes for use.
 */

#include <sprokit/pipeline/process_factory.h>

#include "any_source_process.h"
#include "const_process.h"
#include "const_number_process.h"
#include "data_dependent_process.h"
#include "duplicate_process.h"
#include "expect_process.h"
#include "feedback_process.h"
#include "flow_dependent_process.h"
#include "multiplication_process.h"
#include "multiplier_cluster.h"
#include "mutate_process.h"
#include "number_process.h"
#include "orphan_cluster.h"
#include "orphan_process.h"
#include "print_number_process.h"
#include "shared_process.h"
#include "skip_process.h"
#include "tagged_flow_dependent_process.h"
#include "take_number_process.h"
#include "take_string_process.h"
#include "tunable_process.h"

extern "C"
PROCESSES_EXAMPLES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static const auto module_name = kwiver::vital::plugin_manager::module_t( "example_processes" );

  if ( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  auto fact = vpm.ADD_PROCESS( sprokit::any_source_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "any_source" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which creates arbitrary data" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::const_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "const" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process wth a const flag" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::const_number_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "const_number" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Outputs a constant number" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::data_dependent_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "data_dependent" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process with a data dependent type" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::duplicate_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "duplicate" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which duplicates input" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::expect_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "expect" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which expects some conditions" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::feedback_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "feedback" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which feeds data into itself" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::flow_dependent_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "flow_dependent" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process with a flow dependent type" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::multiplication_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "multiplication" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Multiplies numbers" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::multiplier_cluster );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "multiplier_cluster" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A constant factor multiplier cluster" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::mutate_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "mutate" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process with a mutable flag" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::number_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "numbers" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Outputs numbers within a range" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::orphan_cluster );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "orphan_cluster" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A dummy cluster" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::orphan_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "orphan" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A dummy process" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::print_number_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "print_number" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Print numbers to a file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::shared_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "shared" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process with the shared flag" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::skip_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "skip" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which skips input data" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::tagged_flow_dependent_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "tagged_flow_dependent" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process with a tagged flow dependent types" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::take_number_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "take_number" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Print numbers to a file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::take_string_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "take_string" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Print strings to a file" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS( sprokit::tunable_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "tunable" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process with a tunable parameter" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
