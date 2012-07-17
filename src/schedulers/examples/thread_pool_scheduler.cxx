/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_pool_scheduler.h"

#include <vistk/pipeline/config.h>

#include <boost/thread/thread.hpp>

/**
 * \file thread_pool_scheduler.cxx
 *
 * \brief Implementation of the thread pool scheduler.
 */

namespace vistk
{

class thread_pool_scheduler::priv
{
  public:
    priv(size_t num_threads_);
    ~priv();

    size_t const num_threads;

    bool complete;

    boost::thread_group thread_pool;

    static config::key_t const config_num_threads;
};

config::key_t const thread_pool_scheduler::priv::config_num_threads = config::key_t("num_threads");

thread_pool_scheduler
::thread_pool_scheduler(pipeline_t const& pipe, config_t const& config)
  : scheduler(pipe, config)
{
  unsigned const hardware_concurrency = boost::thread::hardware_concurrency();
  size_t const num_threads = config->get_value<size_t>(priv::config_num_threads, hardware_concurrency - 1);

  d.reset(new priv(num_threads));
}

thread_pool_scheduler
::~thread_pool_scheduler()
{
}

void
thread_pool_scheduler
::_start()
{
  /// \todo Map processes to upstream edges.
  /// \todo Map processes to downstream edges.
  /// \todo Initialize all process statuses to 'ready'.

  while (!d->complete)
  {
    /// \todo Queue processes while upstreams have data and downstreams aren't full.
  }
}

void
thread_pool_scheduler
::_wait()
{
  d->thread_pool.join_all();
}

void
thread_pool_scheduler
::_pause()
{
  /// \todo Implement.
}

void
thread_pool_scheduler
::_resume()
{
  /// \todo Implement.
}

void
thread_pool_scheduler
::_stop()
{
  d->complete = true;
  d->thread_pool.interrupt_all();
}

thread_pool_scheduler::priv
::priv(size_t num_threads_)
  : num_threads(num_threads_)
  , complete(false)
{
}

thread_pool_scheduler::priv
::~priv()
{
}

}
