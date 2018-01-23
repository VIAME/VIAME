/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>

#include "schedulers_test_export.h"

#include <memory>

using namespace sprokit;

class SCHEDULERS_TEST_NO_EXPORT test_scheduler
  : public sprokit::scheduler
{
  public:
    test_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);
    ~test_scheduler();
  protected:
    void _start();
    void _wait();
    void _pause();
    void _resume();
    void _stop();
};

test_scheduler
::test_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
  : scheduler(pipe, config)
{
}

test_scheduler
::~test_scheduler()
{
}

void
test_scheduler
::_start()
{
}

void
test_scheduler
::_wait()
{
}

void
test_scheduler
::_pause()
{
}

void
test_scheduler
::_resume()
{
}

void
test_scheduler
::_stop()
{
}


extern "C"
SCHEDULERS_TEST_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t("test_schedulers");

  if (sprokit::is_scheduler_module_loaded( vpm, module_name ))
  {
    return;
  }

  auto fact = vpm.ADD_SCHEDULER( test_scheduler );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "test" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A test scheduler" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  sprokit::mark_scheduler_module_as_loaded( vpm, module_name );
}
