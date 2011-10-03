/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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

static void run_test(std::string const& test_name);

int
main(int argc, char* argv[])
{
  if (argc != 2)
  {
    TEST_ERROR("Expected one argument");

    return 1;
  }

  std::string const test_name = argv[1];

  try
  {
    run_test(test_name);
  }
  catch (std::exception& e)
  {
    TEST_ERROR("Unexpected exception: " << e.what());

    return 1;
  }

  return 0;
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

static vistk::schedule_t create_schedule(vistk::schedule_registry::type_t const& type);
static vistk::schedule_t create_null_config_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipeline);
static vistk::schedule_t create_null_pipeline_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipeline);

void
test_null_config()
{
  vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const sched_type = vistk::schedule_registry::type_t("null_config");

  reg->register_schedule(sched_type, vistk::schedule_registry::description_t(), create_null_config_schedule);

  EXPECT_EXCEPTION(vistk::null_schedule_config_exception,
                   create_schedule(sched_type),
                   "passing NULL as the configuration for a schedule");
}
void
test_null_pipeline()
{
  vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

  vistk::schedule_registry::type_t const sched_type = vistk::schedule_registry::type_t("null_pipeline");

  reg->register_schedule(sched_type, vistk::schedule_registry::description_t(), create_null_pipeline_schedule);

  EXPECT_EXCEPTION(vistk::null_schedule_pipeline_exception,
                   create_schedule(sched_type),
                   "passing NULL as the pipeline for a schedule");
}

vistk::schedule_t
create_schedule(vistk::schedule_registry::type_t const& type)
{
  static vistk::schedule_registry_t const reg = vistk::schedule_registry::self();

  vistk::config_t const config = vistk::config::empty_config();

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>(config);

  return reg->create_schedule(type, config, pipeline);
}

class null_schedule
  : public vistk::schedule
{
  public:
    null_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe);
    ~null_schedule();

    void start();
    void wait();
    void stop();
};

class null_config_schedule
  : public null_schedule
{
  public:
    null_config_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe);
    ~null_config_schedule();
};

class null_pipeline_schedule
  : public null_schedule
{
  public:
    null_pipeline_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe);
    ~null_pipeline_schedule();
};

vistk::schedule_t
create_null_config_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipeline)
{
  return boost::make_shared<null_config_schedule>(config, pipeline);
}

vistk::schedule_t
create_null_pipeline_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipeline)
{
  return boost::make_shared<null_pipeline_schedule>(config, pipeline);
}

null_schedule
::null_schedule(vistk::config_t const& config, vistk::pipeline_t const& pipe)
  : vistk::schedule(config, pipe)
{
}

null_schedule
::~null_schedule()
{
}

void
null_schedule
::start()
{
}

void
null_schedule
::wait()
{
}

void
null_schedule
::stop()
{
}

null_config_schedule
::null_config_schedule(vistk::config_t const& /*config*/, vistk::pipeline_t const& pipe)
  : null_schedule(vistk::config_t(), pipe)
{
}

null_config_schedule
::~null_config_schedule()
{
}

null_pipeline_schedule
::null_pipeline_schedule(vistk::config_t const& config, vistk::pipeline_t const& /*pipe*/)
  : null_schedule(config, vistk::pipeline_t())
{
}

null_pipeline_schedule
::~null_pipeline_schedule()
{
}
