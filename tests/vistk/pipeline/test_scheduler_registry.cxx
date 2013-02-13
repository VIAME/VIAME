/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_registry.h>
#include <vistk/pipeline/scheduler_registry_exception.h>
#include <vistk/pipeline/types.h>

#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST(get_twice);
DECLARE_TEST(null_config);
DECLARE_TEST(null_pipeline);
DECLARE_TEST(load_schedulers);
DECLARE_TEST(null_ctor);
DECLARE_TEST(duplicate_types);
DECLARE_TEST(unknown_types);
DECLARE_TEST(module_marking);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, get_twice);
  ADD_TEST(tests, null_config);
  ADD_TEST(tests, null_pipeline);
  ADD_TEST(tests, load_schedulers);
  ADD_TEST(tests, null_ctor);
  ADD_TEST(tests, duplicate_types);
  ADD_TEST(tests, unknown_types);
  ADD_TEST(tests, module_marking);

  RUN_TEST(tests, testname);
}

IMPLEMENT_TEST(get_twice)
{
  vistk::scheduler_registry_t const reg1 = vistk::scheduler_registry::self();
  vistk::scheduler_registry_t const reg2 = vistk::scheduler_registry::self();

  if (reg1 != reg2)
  {
    TEST_ERROR("Received two different registries");
  }
}

IMPLEMENT_TEST(null_config)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::config_t const config;
  vistk::pipeline_t const pipe;

  EXPECT_EXCEPTION(vistk::null_scheduler_registry_config_exception,
                   reg->create_scheduler(vistk::scheduler_registry::type_t(), pipe, config),
                   "requesting a NULL config to a scheduler");
}

IMPLEMENT_TEST(null_pipeline)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::config_t const config = vistk::config::empty_config();
  vistk::pipeline_t const pipe;

  EXPECT_EXCEPTION(vistk::null_scheduler_registry_pipeline_exception,
                   reg->create_scheduler(vistk::scheduler_registry::type_t(), pipe),
                   "requesting a NULL pipeline to a scheduler with default arguments");

  EXPECT_EXCEPTION(vistk::null_scheduler_registry_pipeline_exception,
                   reg->create_scheduler(vistk::scheduler_registry::type_t(), pipe, config),
                   "requesting a NULL pipeline to a scheduler");
}

IMPLEMENT_TEST(load_schedulers)
{
  vistk::load_known_modules();

  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::types_t const types = reg->types();

  vistk::pipeline_t const pipe = boost::make_shared<vistk::pipeline>();

  BOOST_FOREACH (vistk::scheduler_registry::type_t const& type, types)
  {
    vistk::scheduler_t scheduler;

    try
    {
      scheduler = reg->create_scheduler(type, pipe);
    }
    catch (vistk::no_such_scheduler_type_exception const& e)
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
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  EXPECT_EXCEPTION(vistk::null_scheduler_ctor_exception,
                   reg->register_scheduler(vistk::scheduler_registry::type_t(), vistk::scheduler_registry::description_t(), vistk::scheduler_ctor_t()),
                   "requesting an non-existent scheduler type");
}

static vistk::scheduler_t null_scheduler(vistk::pipeline_t const& pipeline, vistk::config_t const& config);

IMPLEMENT_TEST(duplicate_types)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const non_existent_scheduler = vistk::scheduler_registry::type_t("no_such_scheduler");

  reg->register_scheduler(non_existent_scheduler, vistk::scheduler_registry::description_t(), null_scheduler);

  EXPECT_EXCEPTION(vistk::scheduler_type_already_exists_exception,
                   reg->register_scheduler(non_existent_scheduler, vistk::scheduler_registry::description_t(), null_scheduler),
                   "requesting an non-existent scheduler type");
}

IMPLEMENT_TEST(unknown_types)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const non_existent_scheduler = vistk::scheduler_registry::type_t("no_such_scheduler");

  vistk::pipeline_t const pipe = boost::make_shared<vistk::pipeline>();

  EXPECT_EXCEPTION(vistk::no_such_scheduler_type_exception,
                   reg->create_scheduler(non_existent_scheduler, pipe),
                   "requesting an non-existent scheduler type");

  EXPECT_EXCEPTION(vistk::no_such_scheduler_type_exception,
                   reg->description(non_existent_scheduler),
                   "requesting an non-existent scheduler type");
}

IMPLEMENT_TEST(module_marking)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::module_t const module = vistk::scheduler_registry::module_t("module");

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

vistk::scheduler_t
null_scheduler(vistk::pipeline_t const& /*pipeline*/, vistk::config_t const& /*config*/)
{
  return vistk::scheduler_t();
}
