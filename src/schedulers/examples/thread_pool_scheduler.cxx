/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
