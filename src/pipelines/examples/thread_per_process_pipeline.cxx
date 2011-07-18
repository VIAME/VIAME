/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_per_process_pipeline.h"

#include <vistk/pipeline/utils.h>

#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>

namespace vistk
{

static void run_process(process_t process);

class thread_per_process_pipeline::priv
{
  public:
    priv();
    ~priv();

    boost::thread_group process_threads;
};

thread_per_process_pipeline
::thread_per_process_pipeline(config_t const& config)
  : pipeline(config)
{
}

thread_per_process_pipeline
::~thread_per_process_pipeline()
{
}

void
thread_per_process_pipeline
::run()
{
  BOOST_FOREACH (process_map_t::value_type& i, m_process_map)
  {
    process_t process = i.second;

    d->process_threads.create_thread(boost::bind(run_process, process));
  }
}

void
thread_per_process_pipeline
::shutdown()
{
  /// \todo Shut the pipeline down.
}

thread_per_process_pipeline::priv
::priv()
{
}

thread_per_process_pipeline::priv
::~priv()
{
}

void
run_process(process_t process)
{
  name_thread(process->name());

  /// \todo Run the process until it is complete.
}

} // end namespace vistk
