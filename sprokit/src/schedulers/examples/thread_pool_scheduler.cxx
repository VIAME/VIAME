// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "thread_pool_scheduler.h"

#include <vital/config/config_block.h>

#include <boost/thread/thread.hpp>

/**
 * \file thread_pool_scheduler.cxx
 *
 * \brief Implementation of the thread pool scheduler.
 */

namespace sprokit
{

class thread_pool_scheduler::priv
{
  public:
    priv(size_t num_threads_);
    ~priv();

    size_t const num_threads;

    bool complete;

    boost::thread_group thread_pool;

    static kwiver::vital::config_block_key_t const config_num_threads;
};

kwiver::vital::config_block_key_t const thread_pool_scheduler::priv::config_num_threads = kwiver::vital::config_block_key_t("num_threads");

thread_pool_scheduler
::thread_pool_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
  : scheduler(pipe, config)
  , d()
{
  unsigned const hardware_concurrency = boost::thread::hardware_concurrency();
  size_t const num_threads = config->get_value<size_t>(priv::config_num_threads, hardware_concurrency - 1);

  d.reset(new priv(num_threads));
}

thread_pool_scheduler
::~thread_pool_scheduler()
{
  shutdown();
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
  , thread_pool()
{
}

thread_pool_scheduler::priv
::~priv()
{
}

}
