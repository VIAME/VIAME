/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include <exception>
#include <iostream>
#include <string>

#include <cstdlib>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return EXIT_FAILURE;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception const& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_get_twice();
static void test_null_config();
static void test_null_pipeline();
static void test_load_schedulers();
static void test_null_ctor();
static void test_duplicate_types();
static void test_unknown_types();
static void test_module_marking();

void
run_test(std::string const& test_name)
{
  if (test_name == "get_twice")
  {
    test_get_twice();
  }
  else if (test_name == "null_config")
  {
    test_null_config();
  }
  else if (test_name == "null_pipeline")
  {
    test_null_pipeline();
  }
  else if (test_name == "load_schedulers")
  {
    test_load_schedulers();
  }
  else if (test_name == "null_ctor")
  {
    test_null_ctor();
  }
  else if (test_name == "duplicate_types")
  {
    test_duplicate_types();
  }
  else if (test_name == "unknown_types")
  {
    test_unknown_types();
  }
  else if (test_name == "module_marking")
  {
    test_module_marking();
  }
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

void
test_get_twice()
{
  vistk::scheduler_registry_t reg1 = vistk::scheduler_registry::self();
  vistk::scheduler_registry_t reg2 = vistk::scheduler_registry::self();

  if (reg1 != reg2)
  {
    TEST_ERROR("Received two different registries");
  }
}

void
test_null_config()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  vistk::config_t config;
  vistk::pipeline_t pipe;

  EXPECT_EXCEPTION(vistk::null_scheduler_registry_config_exception,
                   reg->create_scheduler(vistk::scheduler_registry::type_t(), pipe, config),
                   "requesting a NULL config to a scheduler");
}

void
test_null_pipeline()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  vistk::config_t config = vistk::config::empty_config();
  vistk::pipeline_t pipe;

  EXPECT_EXCEPTION(vistk::null_scheduler_registry_pipeline_exception,
                   reg->create_scheduler(vistk::scheduler_registry::type_t(), pipe),
                   "requesting a NULL pipeline to a scheduler with default arguments");

  EXPECT_EXCEPTION(vistk::null_scheduler_registry_pipeline_exception,
                   reg->create_scheduler(vistk::scheduler_registry::type_t(), pipe, config),
                   "requesting a NULL pipeline to a scheduler");
}

void
test_load_schedulers()
{
  vistk::load_known_modules();

  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::types_t const types = reg->types();

  vistk::pipeline_t pipe = boost::make_shared<vistk::pipeline>();

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

void
test_null_ctor()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  EXPECT_EXCEPTION(vistk::null_scheduler_ctor_exception,
                   reg->register_scheduler(vistk::scheduler_registry::type_t(), vistk::scheduler_registry::description_t(), vistk::scheduler_ctor_t()),
                   "requesting an non-existent scheduler type");
}

static vistk::scheduler_t null_scheduler(vistk::pipeline_t const& pipeline, vistk::config_t const& config);

void
test_duplicate_types()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const non_existent_scheduler = vistk::scheduler_registry::type_t("no_such_scheduler");

  reg->register_scheduler(non_existent_scheduler, vistk::scheduler_registry::description_t(), null_scheduler);

  EXPECT_EXCEPTION(vistk::scheduler_type_already_exists_exception,
                   reg->register_scheduler(non_existent_scheduler, vistk::scheduler_registry::description_t(), null_scheduler),
                   "requesting an non-existent scheduler type");
}

void
test_unknown_types()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const non_existent_scheduler = vistk::scheduler_registry::type_t("no_such_scheduler");

  vistk::pipeline_t pipe = boost::make_shared<vistk::pipeline>();

  EXPECT_EXCEPTION(vistk::no_such_scheduler_type_exception,
                   reg->create_scheduler(non_existent_scheduler, pipe),
                   "requesting an non-existent scheduler type");

  EXPECT_EXCEPTION(vistk::no_such_scheduler_type_exception,
                   reg->description(non_existent_scheduler),
                   "requesting an non-existent scheduler type");
}

void
test_module_marking()
{
  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

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
