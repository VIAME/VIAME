/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_exception.h>
#include <vistk/pipeline/schedule_registry.h>

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
  else
  {
    TEST_ERROR("Unknown test: " << test_name);
  }
}

class null_schedule
  : public vistk::schedule
{
  public:
    null_schedule(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~null_schedule();

    void _start();
    void _wait();
    void _stop();
};

class null_config_schedule
  : public null_schedule
{
  public:
    null_config_schedule(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~null_config_schedule();
};

class null_pipeline_schedule
  : public null_schedule
{
  public:
    null_pipeline_schedule(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~null_pipeline_schedule();
};

static vistk::schedule_t create_schedule(vistk::schedule_registry::type_t const& type);

void
test_null_config()
{
  vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const sched_type = vistk::schedule_registry::type_t("null_config");

  reg->register_schedule(sched_type, vistk::schedule_registry::description_t(), vistk::create_schedule<null_config_schedule>);

  EXPECT_EXCEPTION(vistk::null_schedule_config_exception,
                   create_schedule(sched_type),
                   "passing NULL as the configuration for a schedule");
}
void
test_null_pipeline()
{
  vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const sched_type = vistk::schedule_registry::type_t("null_pipeline");

  reg->register_schedule(sched_type, vistk::schedule_registry::description_t(), vistk::create_schedule<null_pipeline_schedule>);

  EXPECT_EXCEPTION(vistk::null_schedule_pipeline_exception,
                   create_schedule(sched_type),
                   "passing NULL as the pipeline for a schedule");
}

vistk::schedule_t
create_schedule(vistk::schedule_registry::type_t const& type)
{
  static vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>();

  return reg->create_schedule(type, pipeline);
}

null_schedule
::null_schedule(vistk::pipeline_t const& pipe, vistk::config_t const& config)
  : vistk::schedule(pipe, config)
{
}

null_schedule
::~null_schedule()
{
}

void
null_schedule
::_start()
{
}

void
null_schedule
::_wait()
{
}

void
null_schedule
::_stop()
{
}

null_config_schedule
::null_config_schedule(vistk::pipeline_t const& pipe, vistk::config_t const& /*config*/)
  : null_schedule(pipe, vistk::config_t())
{
}

null_config_schedule
::~null_config_schedule()
{
}

null_pipeline_schedule
::null_pipeline_schedule(vistk::pipeline_t const& /*pipe*/, vistk::config_t const& config)
  : null_schedule(vistk::pipeline_t(), config)
{
}

null_pipeline_schedule
::~null_pipeline_schedule()
{
}
