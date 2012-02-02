/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/schedule_registry_exception.h>
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
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static void test_null_config();
static void test_null_pipeline();
static void test_load_schedules();
static void test_null_ctor();
static void test_duplicate_types();
static void test_unknown_types();
static void test_module_marking();

void
run_test(std::string const& test_name)
{
  if (test_name == "null_config")
  {
    test_null_config();
  }
  else if (test_name == "null_pipeline")
  {
    test_null_pipeline();
  }
  else if (test_name == "load_schedules")
  {
    test_load_schedules();
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
test_null_config()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::config_t config;
  vistk::pipeline_t pipe;

  EXPECT_EXCEPTION(vistk::null_schedule_registry_config_exception,
                   reg->create_schedule(vistk::schedule_registry::type_t(), config, pipe),
                   "requesting a NULL config to a schedule");
}

void
test_null_pipeline()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::config_t config = vistk::config::empty_config();
  vistk::pipeline_t pipe;

  EXPECT_EXCEPTION(vistk::null_schedule_registry_pipeline_exception,
                   reg->create_schedule(vistk::schedule_registry::type_t(), config, pipe),
                   "requesting a NULL pipeline to a schedule");
}

void
test_load_schedules()
{
  vistk::load_known_modules();

  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_registry::types_t const types = reg->types();

  vistk::config_t config = vistk::config::empty_config();
  vistk::pipeline_t pipe = boost::make_shared<vistk::pipeline>(config);

  BOOST_FOREACH (vistk::schedule_registry::type_t const& type, types)
  {
    vistk::schedule_t schedule;

    try
    {
      schedule = reg->create_schedule(type, config, pipe);
    }
    catch (vistk::no_such_schedule_type_exception& e)
    {
      TEST_ERROR("Failed to create schedule: " << e.what());

      continue;
    }
    catch (std::exception& e)
    {
      TEST_ERROR("Unexpected exception when creating schedule: " << e.what());

      continue;
    }

    if (!schedule)
    {
      TEST_ERROR("Received NULL schedule (" << type << ")");

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
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  EXPECT_EXCEPTION(vistk::null_schedule_ctor_exception,
                   reg->register_schedule(vistk::schedule_registry::type_t(), vistk::schedule_registry::description_t(), vistk::schedule_ctor_t()),
                   "requesting an non-existent schedule type");
}

static vistk::schedule_t null_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipeline);

void
test_duplicate_types()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const non_existent_schedule = vistk::schedule_registry::type_t("no_such_schedule");

  reg->register_schedule(non_existent_schedule, vistk::schedule_registry::description_t(), null_schedule);

  EXPECT_EXCEPTION(vistk::schedule_type_already_exists_exception,
                   reg->register_schedule(non_existent_schedule, vistk::schedule_registry::description_t(), null_schedule),
                   "requesting an non-existent schedule type");
}

void
test_unknown_types()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const non_existent_schedule = vistk::schedule_registry::type_t("no_such_schedule");

  vistk::config_t config = vistk::config::empty_config();
  vistk::pipeline_t pipe = boost::make_shared<vistk::pipeline>(config);

  EXPECT_EXCEPTION(vistk::no_such_schedule_type_exception,
                   reg->create_schedule(non_existent_schedule, config, pipe),
                   "requesting an non-existent schedule type");

  EXPECT_EXCEPTION(vistk::no_such_schedule_type_exception,
                   reg->description(non_existent_schedule),
                   "requesting an non-existent schedule type");
}

void
test_module_marking()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_registry::module_t const module = vistk::schedule_registry::module_t("module");

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

vistk::schedule_t
null_schedule(vistk::config_t const& /*config*/, vistk::pipeline_t const& /*pipeline*/)
{
  return vistk::schedule_t();
}
