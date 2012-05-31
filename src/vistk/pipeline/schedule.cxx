/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "schedule.h"
#include "schedule_exception.h"

#include "pipeline.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

/**
 * \file schedule.cxx
 *
 * \brief Implementation of the base class for \link vistk::schedule schedules\endlink.
 */

namespace vistk
{

class schedule::priv
{
  public:
    priv(pipeline_t const& pipeline);
    ~priv();

    pipeline_t const p;
    bool running;
    boost::mutex mut;
};

schedule
::~schedule()
{
}

schedule
::schedule(pipeline_t const& pipe, config_t const& config)
  : d(new priv(pipe))
{
  if (!config)
  {
    throw null_schedule_config_exception();
  }

  if (!pipe)
  {
    throw null_schedule_pipeline_exception();
  }
}

void
schedule
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
schedule
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
schedule
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
schedule
::pipeline() const
{
  return d->p;
}

schedule::priv
::priv(pipeline_t const& pipe)
  : p(pipe)
  , running(false)
{
}

schedule::priv
::~priv()
{
}

}
