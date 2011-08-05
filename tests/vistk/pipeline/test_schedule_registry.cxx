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

#include <exception>
#include <iostream>
#include <string>

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Error: Expected one argument" << std::endl;

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: Unexpected exception: " << e.what() << std::endl;

    return 1;
  }

  return 0;
}

static void test_null_config();
static void test_null_pipeline();
static void test_load_schedules();
static void test_duplicate_types();
static void test_unknown_types();

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
  else if (test_name == "duplicate_types")
  {
    test_duplicate_types();
  }
  else if (test_name == "unknown_types")
  {
    test_unknown_types();
  }
  else
  {
    std::cerr << "Error: Unknown test: " << test_name << std::endl;
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
  std::cerr << "Error: Not implemented" << std::endl;
}

void
test_load_schedules()
{
  vistk::load_known_modules();

  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_registry::types_t const types = reg->types();

  vistk::config_t config = vistk::config::empty_config();
  vistk::pipeline_t pipe = vistk::pipeline_t(new vistk::pipeline(config));

  BOOST_FOREACH (vistk::schedule_registry::type_t const& type, types)
  {
    vistk::schedule_t schedule;

    try
    {
      schedule = reg->create_schedule(type, config, pipe);
    }
    catch (vistk::no_such_schedule_type_exception& e)
    {
      std::cerr << "Error: Failed to create schedule: " << e.what() << std::endl;

      continue;
    }
    catch (std::exception& e)
    {
      std::cerr << "Error: Unexpected exception when creating schedule: " << e.what() << std::endl;

      continue;
    }

    if (!schedule)
    {
      std::cerr << "Error: Received NULL schedule (" << type << ")" << std::endl;

      continue;
    }

    if (reg->description(type).empty())
    {
      std::cerr << "Error: The description for "
                << type << " is empty" << std::endl;
    }
  }
}

void
test_duplicate_types()
{
  std::cerr << "Error: Not implemented" << std::endl;
}

void
test_unknown_types()
{
  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const non_existent_schedule = vistk::schedule_registry::type_t("no_such_schedule");

  vistk::config_t config = vistk::config::empty_config();
  vistk::pipeline_t pipe = vistk::pipeline_t(new vistk::pipeline(config));

  EXPECT_EXCEPTION(vistk::no_such_schedule_type_exception,
                   reg->create_schedule(non_existent_schedule, config, pipe),
                   "requesting an non-existent schedule type");

  EXPECT_EXCEPTION(vistk::no_such_schedule_type_exception,
                   reg->description(non_existent_schedule),
                   "requesting an non-existent schedule type");
}
