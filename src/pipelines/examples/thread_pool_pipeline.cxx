/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_pool_pipeline.h"

#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>

namespace vistk
{

class thread_pool_pipeline::priv
{
  public:
    priv();
    ~priv();

    boost::thread_group thread_pool;
};

thread_pool_pipeline
::thread_pool_pipeline(config_t const& config, size_t num_threads)
  : pipeline(config)
  , m_num_threads(num_threads)
  , d(new priv)
{
}

thread_pool_pipeline
::~thread_pool_pipeline()
{
}

void
thread_pool_pipeline
::run()
{
  /// \todo Map processes to upstream edges.
  /// \todo Map processes to downstream edges.
  /// \todo Initialize all process statuses to 'ready'.
  /// \todo Queue processes while upstreams have data and downstreams aren't full.
}

void
thread_pool_pipeline
::shutdown()
{
  /// \todo Shut the pipeline down.
}

thread_pool_pipeline::priv
::priv()
{
}

thread_pool_pipeline::priv
::~priv()
{
}

} // end namespace vistk
