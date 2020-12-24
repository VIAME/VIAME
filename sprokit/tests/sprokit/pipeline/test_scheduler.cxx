// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <vital/config/config_block.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_exception.h>
#include <sprokit/pipeline/scheduler_factory.h>

#include <memory>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

class null_scheduler
  : public sprokit::scheduler
{
  public:
    null_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);
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
    null_config_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);
    ~null_config_scheduler();
};

class null_pipeline_scheduler
  : public null_scheduler
{
  public:
    null_pipeline_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config);
    ~null_pipeline_scheduler();
};

static sprokit::scheduler_t create_scheduler(sprokit::scheduler::type_t const& type);

// ------------------------------------------------------------------
IMPLEMENT_TEST(null_config)
{
  const auto sched_type = sprokit::scheduler::type_t("null_config");
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  auto fact = vpm.ADD_SCHEDULER( null_config_scheduler );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, sched_type )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "" );

  EXPECT_EXCEPTION(sprokit::null_scheduler_config_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the configuration for a scheduler");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(null_pipeline)
{
  const auto sched_type = sprokit::scheduler::type_t("null_pipeline");

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  auto fact = vpm.ADD_SCHEDULER( null_pipeline_scheduler );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, sched_type )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "" );

  EXPECT_EXCEPTION(sprokit::null_scheduler_pipeline_exception,
                   create_scheduler(sched_type),
                   "passing NULL as the pipeline for a scheduler");
}

static sprokit::scheduler_t create_minimal_scheduler();

// ------------------------------------------------------------------
IMPLEMENT_TEST(start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(pause_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(resume_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
  sched->resume();
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(stop_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->stop();
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(stop_paused_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();
  sched->stop();
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(restart_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();

  EXPECT_EXCEPTION(sprokit::restart_scheduler_exception,
                   sched->start(),
                   "calling start on a scheduler a second time");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(repause_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->pause();

  EXPECT_EXCEPTION(sprokit::repause_scheduler_exception,
                   sched->pause(),
                   "pausing a scheduler a second time");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(pause_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::pause_before_start_exception,
                   sched->pause(),
                   "pausing a scheduler before it is started");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(wait_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::wait_before_start_exception,
                   sched->wait(),
                   "waiting on a scheduler before it is started");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(stop_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::stop_before_start_exception,
                   sched->stop(),
                   "stopping a scheduler before it is started");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(resume_before_start_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  EXPECT_EXCEPTION(sprokit::resume_before_start_exception,
                   sched->resume(),
                   "resuming an unstarted scheduler");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(resume_unpaused_scheduler)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();

  EXPECT_EXCEPTION(sprokit::resume_unpaused_scheduler_exception,
                   sched->resume(),
                   "resuming an unpaused scheduler");
}

// ------------------------------------------------------------------
IMPLEMENT_TEST(restart)
{
  sprokit::scheduler_t const sched = create_minimal_scheduler();

  sched->start();
  sched->stop();

  std::shared_ptr<null_scheduler> const null_sched = std::dynamic_pointer_cast<null_scheduler>(sched);

  null_sched->reset_pipeline();

  sched->start();
}

// ------------------------------------------------------------------
sprokit::scheduler_t
create_scheduler(sprokit::scheduler::type_t const& type)
{
  sprokit::pipeline_t const pipeline = std::make_shared<sprokit::pipeline>();

  return sprokit::create_scheduler(type, pipeline);
}

// ------------------------------------------------------------------
sprokit::scheduler_t
create_minimal_scheduler()
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();

  static const auto type = sprokit::process::type_t("orphan");
  static const auto name = sprokit::process::name_t("name");

  sprokit::process_t const proc = sprokit::create_process(type, name);

  sprokit::pipeline_t const pipe = std::make_shared<sprokit::pipeline>();

  pipe->add_process(proc);
  pipe->setup_pipeline();

  kwiver::vital::config_block_sptr const conf = kwiver::vital::config_block::empty_config();

  sprokit::scheduler_t const sched = std::make_shared<null_scheduler>(pipe, conf);

  return sched;
}

// ------------------------------------------------------------------
null_scheduler
::null_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
  : sprokit::scheduler(pipe, config)
{
}

// ------------------------------------------------------------------
null_scheduler
::~null_scheduler()
{
  shutdown();
}

// ------------------------------------------------------------------
void
null_scheduler
::reset_pipeline() const
{
  pipeline()->reset();
  pipeline()->setup_pipeline();
}

// ------------------------------------------------------------------
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
::null_config_scheduler(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& /*config*/)
  : null_scheduler(pipe, kwiver::vital::config_block_sptr())
{
}

null_config_scheduler
::~null_config_scheduler()
{
}

null_pipeline_scheduler
::null_pipeline_scheduler(sprokit::pipeline_t const& /*pipe*/, kwiver::vital::config_block_sptr const& config)
  : null_scheduler(sprokit::pipeline_t(), config)
{
}

null_pipeline_scheduler
::~null_pipeline_scheduler()
{
}
