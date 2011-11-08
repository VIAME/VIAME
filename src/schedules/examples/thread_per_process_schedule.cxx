/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_per_process_schedule.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/schedule_exception.h>
#include <vistk/pipeline/utils.h>

#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

/**
 * \file thread_per_process_schedule.cxx
 *
 * \brief Implementation of the thread-per-process schedule.
 */

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
  , d(new priv)
{
  pipeline_t const p = pipeline();
  process::names_t const names = p->process_names();

  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = p->process_by_name(name);
    process::constraints_t const consts = proc->constraints();

    process::constraints_t::const_iterator const i = consts.find(process::constraint_no_threads);

    if (i != consts.end())
    {
      static std::string const reason = "The process \'" + name + "\' does "
                                        "not support being in its own thread";

      throw incompatible_pipeline_exception(reason);
    }
  }
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
::wait()
{
  d->process_threads.join_all();
}

void
thread_per_process_schedule
::stop()
{
  d->process_threads.interrupt_all();
}

thread_per_process_schedule::priv
::priv()
{
}

thread_per_process_schedule::priv
::~priv()
{
}

static config_t monitor_edge_config();

void
run_process(process_t process)
{
  static config_t edge_conf = monitor_edge_config();

  name_thread(process->name());
  edge_t monitor_edge = boost::make_shared<edge>(edge_conf);

  process->connect_output_port(process::port_heartbeat, monitor_edge);

  bool complete = false;

  while (!complete)
  {
    process->step();

    while (monitor_edge->has_data())
    {
      edge_datum_t const edat = monitor_edge->get_datum();
      datum_t const dat = edat.get<0>();

      if (dat->type() == datum::complete)
      {
        complete = true;
      }
    }

    boost::this_thread::interruption_point();
  }
}

config_t
monitor_edge_config()
{
  config_t conf = config::empty_config();

  return conf;
}

}
