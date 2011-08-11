/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_per_process_schedule.h"

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/utils.h>

#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>

namespace vistk
{

static void run_process(process_t process);

class thread_per_process_schedule::priv
{
  public:
    priv();
    ~priv();

    boost::thread_group process_threads;
};

thread_per_process_schedule
::thread_per_process_schedule(config_t const& config, pipeline_t const& pipe)
  : schedule(config, pipe)
{
  d = boost::shared_ptr<priv>(new priv);
}

thread_per_process_schedule
::~thread_per_process_schedule()
{
}

void
thread_per_process_schedule
::start()
{
  BOOST_FOREACH (process::name_t const& name, pipeline()->process_names())
  {
    process_t process = pipeline()->process_by_name(name);

    d->process_threads.create_thread(boost::bind(run_process, process));
  }
}

void
thread_per_process_schedule
::stop()
{
  /// \todo Shut the schedule down.
}

thread_per_process_schedule::priv
::priv()
{
}

thread_per_process_schedule::priv
::~priv()
{
}

void
run_process(process_t process)
{
  name_thread(process->name());

  /// \todo Run the process until it is complete.
}

}
