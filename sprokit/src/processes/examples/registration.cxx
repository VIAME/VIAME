/*ckwg +29
 * Copyright 2011-2016, 2020 by Kitware, Inc.
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
  using namespace sprokit;

  process_registrar reg( vpm, "sprokit.example_processes" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< any_source_process >();
  reg.register_process< const_number_process >();
  reg.register_process< const_process >();
  reg.register_process< data_dependent_process >();
  reg.register_process< duplicate_process >();
  reg.register_process< expect_process >();
  reg.register_process< feedback_process >();
  reg.register_process< flow_dependent_process >();
  reg.register_process< multiplication_process >();
  reg.register_process< multiplier_cluster >();
  reg.register_process< mutate_process >();
  reg.register_process< number_process >();
  reg.register_process< orphan_cluster >();
  reg.register_process< orphan_process >();
  reg.register_process< print_number_process >();
  reg.register_process< shared_process >();
  reg.register_process< skip_process >();
  reg.register_process< tagged_flow_dependent_process >();
  reg.register_process< take_number_process >();
  reg.register_process< take_string_process >();
  reg.register_process< tunable_process >();

// - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
