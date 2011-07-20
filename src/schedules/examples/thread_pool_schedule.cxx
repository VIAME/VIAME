/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_pool_schedule.h"

#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>

namespace vistk
{

class thread_pool_schedule::priv
{
  public:
    priv();
    ~priv();

    boost::thread_group thread_pool;
};

thread_pool_schedule
::thread_pool_schedule(config_t const& config, pipeline_t const& pipe, size_t num_threads)
  : schedule(config, pipe)
  , m_num_threads(num_threads)
  , d(new priv)
{
}

thread_pool_schedule
::~thread_pool_schedule()
{
}

void
thread_pool_schedule
::start()
{
  /// \todo Map processes to upstream edges.
  /// \todo Map processes to downstream edges.
  /// \todo Initialize all process statuses to 'ready'.
  /// \todo Queue processes while upstreams have data and downstreams aren't full.
}

void
thread_pool_schedule
::stop()
{
  /// \todo Shut the schedule down.
}

thread_pool_schedule::priv
::priv()
{
}

thread_pool_schedule::priv
::~priv()
{
}

} // end namespace vistk
