/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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

#include <test_common.h>

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>
#include <sprokit/pipeline/scheduler_registry_exception.h>
#include <sprokit/pipeline/types.h>

#include <memory>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_config)
{
  kwiver::vital::config_block_sptr const config;
  sprokit::pipeline_t const pipe;

  EXPECT_EXCEPTION(sprokit::null_scheduler_registry_config_exception,
                   sprokit::create_scheduler(sprokit::scheduler::type_t(), pipe, config),
                   "requesting a NULL config to a scheduler");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(null_pipeline)
{
  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();
  sprokit::pipeline_t const pipe;

  EXPECT_EXCEPTION(sprokit::null_scheduler_registry_pipeline_exception,
                   sprokit::create_scheduler(sprokit::scheduler::type_t(), pipe),
                   "requesting a NULL pipeline to a scheduler with default arguments");

  EXPECT_EXCEPTION(sprokit::null_scheduler_registry_pipeline_exception,
                   sprokit::create_scheduler(sprokit::scheduler::type_t(), pipe, config),
                   "requesting a NULL pipeline to a scheduler");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(load_schedulers)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  auto factories =  kwiver::vital::plugin_manager::instance().get_factories<sprokit::scheduler>();

  sprokit::pipeline_t const pipe = std::make_shared<sprokit::pipeline>();

  for( auto fact : factories )
  {
    sprokit::scheduler_t scheduler;

    sprokit::scheduler::type_t type; // scheduler name
    if ( ! fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type ) )
    {
      TEST_ERROR( "Scheduler factory does not have process name attribute" );
      continue;
    }

    try
    {
      scheduler = sprokit::create_scheduler(type, pipe);
    }
    catch (sprokit::no_such_scheduler_type_exception const& e)
    {
      TEST_ERROR("Failed to create scheduler: " << e.what());
      continue;
    }
    catch (std::exception const& e)
    {
      TEST_ERROR("Unexpected exception when creating scheduler: " << e.what());
      continue;
    }

    if (!scheduler)
    {
      TEST_ERROR("Received NULL scheduler (" << type << ")");
      continue;
    }

    sprokit::scheduler::description_t descrip;
    if ( ! fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descrip ) || descrip.empty() )
    {
      TEST_ERROR("The description for " << type << " is empty");
    }
  }
}


class null_scheduler
  : public sprokit::scheduler
{
public:
  null_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
    : scheduler( pipe, config )
  { }

  virtual ~null_scheduler() {}

  virtual void _start() {}
  virtual void _wait() {}
  virtual void _pause(){}
  virtual void _resume() {}
  virtual void _stop() {}
};


// ------------------------------------------------------------------
IMPLEMENT_TEST(duplicate_types)
{
  sprokit::scheduler::type_t const non_existent_scheduler = sprokit::scheduler::type_t("no_such_scheduler");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.ADD_SCHEDULER( null_scheduler );

  EXPECT_EXCEPTION( kwiver::vital::plugin_already_exists,
                    vpm.ADD_SCHEDULER( null_scheduler),
                    "adding duplicate scheduler type");
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(unknown_types)
{
  sprokit::scheduler::type_t const non_existent_scheduler = sprokit::scheduler::type_t("no_such_scheduler");

  sprokit::pipeline_t const pipe = std::make_shared<sprokit::pipeline>();

  EXPECT_EXCEPTION(sprokit::no_such_scheduler_type_exception,
                   sprokit::create_scheduler(non_existent_scheduler, pipe),
                   "requesting an non-existent scheduler type");
}


// ------------------------------------------------------------------
sprokit::scheduler_t
null_scheduler(sprokit::pipeline_t const& /*pipeline*/, kwiver::vital::config_block_sptr const& /*config*/)
{
  return sprokit::scheduler_t();
}
