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

#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_factory.h>

#include "processes_test_export.h"

using namespace sprokit;

class PROCESSES_TEST_NO_EXPORT test_process
  : public sprokit::process
{
  public:
  PLUGIN_INFO( "test",
               "A test process" );

    test_process(kwiver::vital::config_block_sptr const& config);
    ~test_process();
};

test_process
::test_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
{
}

test_process
::~test_process()
{
}

// ------------------------------------------------------------------
extern "C"
PROCESSES_TEST_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  process_registrar reg( vpm, "test_processes" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< test_process >();

  reg.mark_module_as_loaded();
}
