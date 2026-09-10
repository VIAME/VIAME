// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "scheduler.h"
#include "scheduler_exception.h"

#include "pipeline.h"

#include <mutex>
#include <shared_mutex>

/**
 * \file scheduler.cxx
 *
 * \brief Implementation of the base class for \link sprokit::scheduler schedulers\endlink.
 */

namespace sprokit {

class scheduler::priv
{
  public:
    priv(scheduler* sched, pipeline_t const& pipe);
    ~priv();

    void stop();

    scheduler* const q;
    pipeline_t const p;
    bool paused;
    bool running;

    typedef std::shared_mutex mutex_t;
    typedef std::unique_lock<mutex_t> unique_lock_t;

    mutex_t mut;
};

// ============================================================================
scheduler
::~scheduler()
{
}

scheduler
::scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
  : m_logger( kwiver::vital::get_logger( "scheduler.base" ) )
  , d()
{
  if (!config)
  {
    VITAL_THROW( null_scheduler_config_exception );
  }

  if (!pipe)
  {
    VITAL_THROW( null_scheduler_pipeline_exception );
  }

  d.reset(new priv(this, pipe));
}

// ----------------------------------------------------------------------------
void
scheduler
::start()
{
  priv::unique_lock_t const lock(d->mut);

  (void)lock;

  if (d->running)
  {
    VITAL_THROW( restart_scheduler_exception );
  }

  d->p->start();

  _start();

  d->running = true;
}

// ----------------------------------------------------------------------------
void
scheduler
::wait()
{
  priv::unique_lock_t lock(d->mut);

  if (!d->running)
  {
    VITAL_THROW( wait_before_start_exception );
  }

  // Allow many threads to wait on the scheduler.
  {
    lock.unlock();

    _wait();

    lock.lock();
  }

  // After each thread, only one should call stop. Let threads through
  // one-at-a-time to see if the pipeline needs to be stopped yet.
  if (d->running)
  {
    d->stop();
  }
}

// ----------------------------------------------------------------------------
void
scheduler
::pause()
{
  priv::unique_lock_t lock(d->mut);

  if (!d->running)
  {
    VITAL_THROW( pause_before_start_exception );
  }

  if (d->paused)
  {
    VITAL_THROW( repause_scheduler_exception );
  }

  _pause();

  d->paused = true;
}

// ----------------------------------------------------------------------------
void
scheduler
::resume()
{
  priv::unique_lock_t const lock(d->mut);

  (void)lock;

  if (!d->running)
  {
    VITAL_THROW( resume_before_start_exception );
  }

  if (!d->paused)
  {
    VITAL_THROW( resume_unpaused_scheduler_exception );
  }

  _resume();

  d->paused = false;
}

// ----------------------------------------------------------------------------
void
scheduler
::stop()
{
  priv::unique_lock_t const lock(d->mut);

  (void)lock;

  if (!d->running)
  {
    VITAL_THROW( stop_before_start_exception );
  }

  d->stop();
}

// ----------------------------------------------------------------------------
void
scheduler
::shutdown()
{
  priv::unique_lock_t const lock(d->mut);

  (void)lock;

  if (d->running)
  {
    d->stop();
  }
}

// ----------------------------------------------------------------------------
pipeline_t
scheduler
::pipeline() const
{
  return d->p;
}

// ----------------------------------------------------------------------------
kwiver::vital::logger_handle_t
scheduler
::logger()
{
  return m_logger;
}

// ============================================================================
scheduler::priv
::priv(scheduler* sched, pipeline_t const& pipe)
  : q(sched)
  , p(pipe)
  , paused(false)
  , running(false)
  , mut()
{
}

scheduler::priv
::~priv()
{
}

// ----------------------------------------------------------------------------
void
scheduler::priv
::stop()
{
  // Tell the subclass that we want to stop.
  q->_stop();

  // Unpause the scheduler.
  if (paused)
  {
    q->_resume();

    paused = false;
  }

  // Stop the pipeline.
  p->stop();
  running = false;
}

}
