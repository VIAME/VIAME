/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_exception.h>
#include <vistk/pipeline/scheduler_registry.h>

#include <boost/make_shared.hpp>

#define TEST_ARGS ()

DECLARE_TEST(null_config);
DECLARE_TEST(null_pipeline);
DECLARE_TEST(start_scheduler);
DECLARE_TEST(pause_scheduler);
DECLARE_TEST(resume_scheduler);
DECLARE_TEST(stop_scheduler);
DECLARE_TEST(stop_paused_scheduler);
DECLARE_TEST(restart_scheduler);
DECLARE_TEST(repause_scheduler);
DECLARE_TEST(pause_before_start_scheduler);
DECLARE_TEST(wait_before_start_scheduler);
DECLARE_TEST(stop_before_start_scheduler);
DECLARE_TEST(resume_before_start_scheduler);
DECLARE_TEST(resume_unpaused_scheduler);

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  DECLARE_TEST_MAP(tests);

  ADD_TEST(tests, null_config);
  ADD_TEST(tests, null_pipeline);
  ADD_TEST(tests, start_scheduler);
  ADD_TEST(tests, pause_scheduler);
  ADD_TEST(tests, resume_scheduler);
  ADD_TEST(tests, stop_scheduler);
  ADD_TEST(tests, stop_paused_scheduler);
  ADD_TEST(tests, restart_scheduler);
  ADD_TEST(tests, repause_scheduler);
  ADD_TEST(tests, pause_before_start_scheduler);
  ADD_TEST(tests, wait_before_start_scheduler);
  ADD_TEST(tests, stop_before_start_scheduler);
  ADD_TEST(tests, resume_before_start_scheduler);
  ADD_TEST(tests, resume_unpaused_scheduler);

  RUN_TEST(tests, testname);
}

class null_scheduler
  : public vistk::scheduler
{
  public:
    null_scheduler(vistk::pipeline_t const& pipe, vistk::config_t const& config);
    virtual ~null_scheduler();
  protected:
    void _start();
    void _wait();
    void _pause();
    void _resume();
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

IMPLEMENT_TEST(null_config)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const sched_type = vistk::scheduler_registry::type_t("null_config");

  reg->register_scheduler(sched_type, vistk::scheduler_registry::description_t(), vistk::create_scheduler<null_config_scheduler>);

  EXPECT_EXCEPTION(vistk::null_scheduler_config_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the configuration for a scheduler");
}

IMPLEMENT_TEST(null_pipeline)
{
  vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::scheduler_registry::type_t const sched_type = vistk::scheduler_registry::type_t("null_pipeline");

  reg->register_scheduler(sched_type, vistk::scheduler_registry::description_t(), vistk::create_scheduler<null_pipeline_scheduler>);

  EXPECT_EXCEPTION(vistk::null_scheduler_pipeline_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the pipeline for a scheduler");
}

static vistk::scheduler_t create_minimal_scheduler();

IMPLEMENT_TEST(start_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
}

IMPLEMENT_TEST(pause_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
}

IMPLEMENT_TEST(resume_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
  sched->resume();
}

IMPLEMENT_TEST(stop_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->stop();
}

IMPLEMENT_TEST(stop_paused_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
  sched->stop();
}

IMPLEMENT_TEST(restart_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();

  EXPECT_EXCEPTION(vistk::restart_scheduler_exception,
                   sched->start(),
                   "calling start on a scheduler a second time");
}

IMPLEMENT_TEST(repause_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();

  EXPECT_EXCEPTION(vistk::repause_scheduler_exception,
                   sched->pause(),
                   "pausing a scheduler a second time");
}

IMPLEMENT_TEST(pause_before_start_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(vistk::pause_before_start_exception,
                   sched->pause(),
                   "pausing a scheduler before it is started");
}

IMPLEMENT_TEST(wait_before_start_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(vistk::wait_before_start_exception,
                   sched->wait(),
                   "waiting on a scheduler before it is started");
}

IMPLEMENT_TEST(stop_before_start_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(vistk::stop_before_start_exception,
                   sched->stop(),
                   "stopping a scheduler before it is started");
}

IMPLEMENT_TEST(resume_before_start_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(vistk::resume_before_start_exception,
                   sched->resume(),
                   "resuming an unstarted scheduler");
}

IMPLEMENT_TEST(resume_unpaused_scheduler)
{
  vistk::scheduler_t const sched = create_minimal_scheduler();

  sched->start();

  EXPECT_EXCEPTION(vistk::resume_unpaused_scheduler_exception,
                   sched->resume(),
                   "resuming an unpaused scheduler");
}

vistk::scheduler_t
create_scheduler(vistk::scheduler_registry::type_t const& type)
{
  static vistk::scheduler_registry_t const reg = vistk::scheduler_registry::self();

  vistk::pipeline_t const pipeline = boost::make_shared<vistk::pipeline>();

  return reg->create_scheduler(type, pipeline);
}

vistk::scheduler_t
create_minimal_scheduler()
{
  vistk::load_known_modules();

  static vistk::process::type_t const type = vistk::process::type_t("orphan");
  static vistk::process::name_t const name = vistk::process::name_t("name");

  vistk::process_registry_t const reg = vistk::process_registry::self();
  vistk::process_t const proc = reg->create_process(type, name);

  vistk::pipeline_t const pipe = boost::make_shared<vistk::pipeline>();

  pipe->add_process(proc);
  pipe->setup_pipeline();

  vistk::config_t const conf = vistk::config::empty_config();

  vistk::scheduler_t const sched = boost::make_shared<null_scheduler>(pipe, conf);

  return sched;
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
::_pause()
{
}

void
null_scheduler
::_resume()
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
