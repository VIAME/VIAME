/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "thread_per_process_scheduler.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/scheduler_exception.h>
#include <vistk/pipeline/utils.h>

#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

/**
 * \file thread_per_process_scheduler.cxx
 *
 * \brief Implementation of the thread-per-process scheduler.
 */

namespace vistk
{

static void run_process(process_t process);

class thread_per_process_scheduler::priv
{
  public:
    priv();
    ~priv();

    boost::thread_group process_threads;
};

thread_per_process_scheduler
::thread_per_process_scheduler(pipeline_t const& pipe, config_t const& config)
  : scheduler(pipe, config)
  , d(new priv)
{
  pipeline_t const p = pipeline();
  process::names_t const names = p->process_names();

  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = p->process_by_name(name);
    process::constraints_t const consts = proc->constraints();

    process::constraints_t::const_iterator i;

    i = consts.find(process::constraint_no_threads);

    if (i != consts.end())
    {
      std::string const reason = "The process \'" + name + "\' does "
                                 "not support being in its own thread";

      throw incompatible_pipeline_exception(reason);
    }
  }
}

thread_per_process_scheduler
::~thread_per_process_scheduler()
{
}

void
thread_per_process_scheduler
::_start()
{
  pipeline_t const p = pipeline();
  process::names_t const names = p->process_names();

  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const process = pipeline()->process_by_name(name);

    d->process_threads.create_thread(boost::bind(run_process, process));
  }
}

void
thread_per_process_scheduler
::_wait()
{
  d->process_threads.join_all();
}

void
thread_per_process_scheduler
::_stop()
{
  d->process_threads.interrupt_all();
}

thread_per_process_scheduler::priv
::priv()
{
}

thread_per_process_scheduler::priv
::~priv()
{
}

static config_t monitor_edge_config();

void
run_process(process_t process)
{
  config_t const edge_conf = monitor_edge_config();

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
      datum_t const dat = edat.datum;

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
