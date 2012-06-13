/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scheduler.h"
#include "scheduler_exception.h"

#include "pipeline.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

/**
 * \file scheduler.cxx
 *
 * \brief Implementation of the base class for \link vistk::scheduler schedulers\endlink.
 */

namespace vistk
{

class scheduler::priv
{
  public:
    priv(pipeline_t const& pipeline);
    ~priv();

    pipeline_t const p;
    bool running;
    boost::mutex mut;
};

scheduler
::~scheduler()
{
}

scheduler
::scheduler(pipeline_t const& pipe, config_t const& config)
  : d(new priv(pipe))
{
  if (!config)
  {
    throw null_scheduler_config_exception();
  }

  if (!pipe)
  {
    throw null_scheduler_pipeline_exception();
  }
}

void
scheduler
::start()
{
  {
    boost::mutex::scoped_lock const lock(d->mut);

    (void)lock;

    d->p->start();
    d->running = true;
  }

  _start();
}

void
scheduler
::wait()
{
  {
    boost::mutex::scoped_lock const lock(d->mut);

    (void)lock;

    if (!d->running)
    {
      /// \todo Throw an exception.
    }
  }

  _wait();

  stop();
}

void
scheduler
::stop()
{
  boost::mutex::scoped_lock const lock(d->mut);

  (void)lock;

  if (!d->running)
  {
    /// \todo Throw an exception.
  }

  _stop();

  d->p->stop();
  d->running = false;
}

pipeline_t
scheduler
::pipeline() const
{
  return d->p;
}

scheduler::priv
::priv(pipeline_t const& pipe)
  : p(pipe)
  , running(false)
{
}

scheduler::priv
::~priv()
{
}

}
