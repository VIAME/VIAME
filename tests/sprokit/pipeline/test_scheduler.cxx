/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_exception.h>
#include <sprokit/pipeline/scheduler_registry.h>

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
DECLARE_TEST(restart);

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
  ADD_TEST(tests, restart);

  RUN_TEST(tests, testname);
}

class null_scheduler
  : public sprokit::scheduler
{
  public:
    null_scheduler(sprokit::pipeline_t const& pipe, sprokit::config_t const& config);
    virtual ~null_scheduler();

    void reset_pipeline() const;
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
    null_config_scheduler(sprokit::pipeline_t const& pipe, sprokit::config_t const& config);
    ~null_config_scheduler();
};

class null_pipeline_scheduler
  : public null_scheduler
{
  public:
    null_pipeline_scheduler(sprokit::pipeline_t const& pipe, sprokit::config_t const& config);
    ~null_pipeline_scheduler();
};

static sprokit::scheduler_t create_scheduler(sprokit::scheduler_registry::type_t const& type);

IMPLEMENT_TEST(null_config)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_registry::type_t const sched_type = sprokit::scheduler_registry::type_t("null_config");

  reg->register_scheduler(sched_type, sprokit::scheduler_registry::description_t(), sprokit::create_scheduler<null_config_scheduler>);

  EXPECT_EXCEPTION(sprokit::null_scheduler_config_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the configuration for a scheduler");
}

IMPLEMENT_TEST(null_pipeline)
{
  sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_registry::type_t const sched_type = sprokit::scheduler_registry::type_t("null_pipeline");

  reg->register_scheduler(sched_type, sprokit::scheduler_registry::description_t(), sprokit::create_scheduler<null_pipeline_scheduler>);

  EXPECT_EXCEPTION(sprokit::null_scheduler_pipeline_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the pipeline for a scheduler");
}

static sprokit::scheduler_t create_minimal_scheduler();

IMPLEMENT_TEST(start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
}

IMPLEMENT_TEST(pause_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
}

IMPLEMENT_TEST(resume_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
  sched->resume();
}

IMPLEMENT_TEST(stop_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->stop();
}

IMPLEMENT_TEST(stop_paused_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
  sched->stop();
}

IMPLEMENT_TEST(restart_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();

  EXPECT_EXCEPTION(sprokit::restart_scheduler_exception,
                   sched->start(),
                   "calling start on a scheduler a second time");
}

IMPLEMENT_TEST(repause_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();

  EXPECT_EXCEPTION(sprokit::repause_scheduler_exception,
                   sched->pause(),
                   "pausing a scheduler a second time");
}

IMPLEMENT_TEST(pause_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::pause_before_start_exception,
                   sched->pause(),
                   "pausing a scheduler before it is started");
}

IMPLEMENT_TEST(wait_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::wait_before_start_exception,
                   sched->wait(),
                   "waiting on a scheduler before it is started");
}

IMPLEMENT_TEST(stop_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::stop_before_start_exception,
                   sched->stop(),
                   "stopping a scheduler before it is started");
}

IMPLEMENT_TEST(resume_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::resume_before_start_exception,
                   sched->resume(),
                   "resuming an unstarted scheduler");
}

IMPLEMENT_TEST(resume_unpaused_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();

  EXPECT_EXCEPTION(sprokit::resume_unpaused_scheduler_exception,
                   sched->resume(),
                   "resuming an unpaused scheduler");
}

IMPLEMENT_TEST(restart)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->stop();

  boost::shared_ptr<null_scheduler> const null_sched = boost::dynamic_pointer_cast<null_scheduler>(sched);

  null_sched->reset_pipeline();

  sched->start();
}

sprokit::scheduler_t
create_scheduler(sprokit::scheduler_registry::type_t const& type)
{
  static sprokit::scheduler_registry_t const reg = sprokit::scheduler_registry::self();

  sprokit::pipeline_t const pipeline = boost::make_shared<sprokit::pipeline>();

  return reg->create_scheduler(type, pipeline);
}

sprokit::scheduler_t
create_minimal_scheduler()
{
  sprokit::load_known_modules();

  static sprokit::process::type_t const type = sprokit::process::type_t("orphan");
  static sprokit::process::name_t const name = sprokit::process::name_t("name");

  sprokit::process_registry_t const reg = sprokit::process_registry::self();
  sprokit::process_t const proc = reg->create_process(type, name);

  sprokit::pipeline_t const pipe = boost::make_shared<sprokit::pipeline>();

  pipe->add_process(proc);
  pipe->setup_pipeline();

  sprokit::config_t const conf = sprokit::config::empty_config();

  sprokit::scheduler_t const sched = boost::make_shared<null_scheduler>(pipe, conf);

  return sched;
}

null_scheduler
::null_scheduler(sprokit::pipeline_t const& pipe, sprokit::config_t const& config)
  : sprokit::scheduler(pipe, config)
{
}

null_scheduler
::~null_scheduler()
{
}

void
null_scheduler
::reset_pipeline() const
{
  pipeline()->reset();
  pipeline()->setup_pipeline();
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
::null_config_scheduler(sprokit::pipeline_t const& pipe, sprokit::config_t const& /*config*/)
  : null_scheduler(pipe, sprokit::config_t())
{
}

null_config_scheduler
::~null_config_scheduler()
{
}

null_pipeline_scheduler
::null_pipeline_scheduler(sprokit::pipeline_t const& /*pipe*/, sprokit::config_t const& config)
  : null_scheduler(sprokit::pipeline_t(), config)
{
}

null_pipeline_scheduler
::~null_pipeline_scheduler()
{
}
