/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_exception.h>
#include <vistk/pipeline/scheduler_registry.h>

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

class null_scheduler
  : public vistk::scheduler
{
  public:
    null_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~null_scheduler();

    void _start();
    void _wait();
    void _stop();
};

class null_config_scheduler
  : public null_scheduler
{
  public:
    null_config_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~null_config_scheduler();
};

class null_pipeline_scheduler
  : public null_scheduler
{
  public:
    null_pipeline_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    ~null_pipeline_scheduler();
};

static vistk::scheduler_t create_scheduler(vistk::scheduler_registry::type_t const& type);

void
test_null_config()
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const sched_type = vistk::scheduler_registry::type_t("null_config");

  reg->register_scheduler(sched_type, vistk::scheduler_registry::description_t(), vistk::create_scheduler<null_config_scheduler>);

  EXPECT_EXCEPTION(vistk::null_scheduler_config_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the configuration for a scheduler");
}
void
test_null_pipeline()
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const sched_type = vistk::scheduler_registry::type_t("null_pipeline");

  reg->register_scheduler(sched_type, vistk::scheduler_registry::description_t(), vistk::create_scheduler<null_pipeline_scheduler>);

  EXPECT_EXCEPTION(vistk::null_scheduler_pipeline_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the pipeline for a scheduler");
}

vistk::scheduler_t
create_scheduler(vistk::scheduler_registry::type_t const& type)
{
  static vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>();

  return reg->create_scheduler(type, pipeline);
}

null_scheduler
::null_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config)
  : vistk::scheduler(pipe, config)
{
}

null_scheduler
::~null_scheduler()
{
}

void
null_scheduler
::_start()
{
}

void
null_scheduler
::_wait()
{
}

void
null_scheduler
::_stop()
{
}

null_config_scheduler
::null_config_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& /*config*/)
  : null_scheduler(pipe, vistk::config_t())
{
}

null_config_scheduler
::~null_config_scheduler()
{
}

null_pipeline_scheduler
::null_pipeline_scheduler(vistk::pipeline_t const& /*pipe*/, vistk::config_t const& config)
  : null_scheduler(vistk::pipeline_t(), config)
{
}

null_pipeline_scheduler
::~null_pipeline_scheduler()
{
}
