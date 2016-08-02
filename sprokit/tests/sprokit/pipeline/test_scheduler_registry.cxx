/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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
#include <vital/vital_foreach.h>

#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_registry.h>
#include <sprokit/pipeline/scheduler_registry_exception.h>
#include <sprokit/pipeline/types.h>

#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

IMPLEMENT_TEST(get_twice)
{
  sprokit::scheduler_registry_t const reg1 = sprokit::scheduler_registry::self();
  sprokit::scheduler_registry_t const reg2 = sprokit::scheduler_registry::self();

  if (reg1 != reg2)
  {
    TEST_ERROR("Received two different registries");
  }
}

IMPLEMENT_TEST(null_config)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  kwiver::vital::config_block_sptr const config;
  sprokit::pipeline_t const pipe;

  EXPECT_EXCEPTION(sprokit::null_scheduler_registry_config_exception,
                   reg->create_scheduler(sprokit::scheduler_registry::type_t(), pipe, config),
                   "requesting a NULL config to a scheduler");
}

IMPLEMENT_TEST(null_pipeline)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  kwiver::vital::config_block_sptr const config = kwiver::vital::config_block::empty_config();
  sprokit::pipeline_t const pipe;

  EXPECT_EXCEPTION(sprokit::null_scheduler_registry_pipeline_exception,
                   reg->create_scheduler(sprokit::scheduler_registry::type_t(), pipe),
                   "requesting a NULL pipeline to a scheduler with default arguments");

  EXPECT_EXCEPTION(sprokit::null_scheduler_registry_pipeline_exception,
                   reg->create_scheduler(sprokit::scheduler_registry::type_t(), pipe, config),
                   "requesting a NULL pipeline to a scheduler");
}

IMPLEMENT_TEST(load_schedulers)
{
  sprokit::load_known_modules();

  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_registry::types_t const types = reg->types();

  sprokit::pipeline_t const pipe = boost::make_shared<sprokit::pipeline>();

  VITAL_FOREACH (sprokit::scheduler_registry::type_t const& type, types)
  {
    sprokit::scheduler_t scheduler;

    try
    {
      scheduler = reg->create_scheduler(type, pipe);
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

    if (reg->description(type).empty())
    {
      TEST_ERROR("The description for "
                 << type << " is empty");
    }
  }
}

IMPLEMENT_TEST(null_ctor)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  EXPECT_EXCEPTION(sprokit::null_scheduler_ctor_exception,
                   reg->register_scheduler(sprokit::scheduler_registry::type_t(), sprokit::scheduler_registry::description_t(), sprokit::scheduler_ctor_t()),
                   "requesting an non-existent scheduler type");
}

static sprokit::scheduler_t null_scheduler(sprokit::pipeline_t const& pipeline, kwiver::vital::config_block_sptr const& config);

IMPLEMENT_TEST(duplicate_types)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_registry::type_t const non_existent_scheduler = sprokit::scheduler_registry::type_t("no_such_scheduler");

  reg->register_scheduler(non_existent_scheduler, sprokit::scheduler_registry::description_t(), null_scheduler);

  EXPECT_EXCEPTION(sprokit::scheduler_type_already_exists_exception,
                   reg->register_scheduler(non_existent_scheduler, sprokit::scheduler_registry::description_t(), null_scheduler),
                   "requesting an non-existent scheduler type");
}

IMPLEMENT_TEST(unknown_types)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_registry::type_t const non_existent_scheduler = sprokit::scheduler_registry::type_t("no_such_scheduler");

  sprokit::pipeline_t const pipe = boost::make_shared<sprokit::pipeline>();

  EXPECT_EXCEPTION(sprokit::no_such_scheduler_type_exception,
                   reg->create_scheduler(non_existent_scheduler, pipe),
                   "requesting an non-existent scheduler type");

  EXPECT_EXCEPTION(sprokit::no_such_scheduler_type_exception,
                   reg->description(non_existent_scheduler),
                   "requesting an non-existent scheduler type");
}

IMPLEMENT_TEST(module_marking)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_registry::module_t const module = sprokit::scheduler_registry::module_t("module");

  if (reg->is_module_loaded(module))
  {
    TEST_ERROR("The module \'" << module << "\' is "
               "already marked as loaded");
  }

  reg->mark_module_as_loaded(module);

  if (!reg->is_module_loaded(module))
  {
    TEST_ERROR("The module \'" << module << "\' is "
               "not marked as loaded");
  }
}

sprokit::scheduler_t
null_scheduler(sprokit::pipeline_t const& /*pipeline*/, kwiver::vital::config_block_sptr const& /*config*/)
{
  return sprokit::scheduler_t();
}
