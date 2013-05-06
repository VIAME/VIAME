/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "sync_scheduler.h"

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/scheduler_exception.h>
#include <sprokit/pipeline/utils.h>

#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <deque>
#include <iterator>
#include <map>
#include <queue>

/**
 * \file sync_scheduler.cxx
 *
 * \brief Implementation of the synchronized scheduler.
 */

namespace sprokit
{

static thread_name_t const thread_name = thread_name_t("sync_scheduler");

class sync_scheduler::priv
{
  public:
    priv();
    ~priv();

    void run(pipeline_t const& pipe);

    boost::thread thread;

    typedef boost::shared_mutex mutex_t;
    typedef boost::shared_lock<mutex_t> shared_lock_t;

    mutable mutex_t mut;
};

sync_scheduler
::sync_scheduler(pipeline_t const& pipe, config_t const& config)
  : scheduler(pipe, config)
  , d(new priv)
{
  pipeline_t const p = pipeline();
  process::names_t const names = p->process_names();

  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = p->process_by_name(name);
    process::properties_t const consts = proc->properties();

    if (consts.count(process::property_unsync_output))
    {
      std::string const reason = "The process \'" + name + "\' does not output "
                                 "consistent data across all its output ports";

      throw incompatible_pipeline_exception(reason);
    }

    if (consts.count(process::property_unsync_input))
    {
      std::string const reason = "The process \'" + name + "\' does not expect "
                                 "consistent data across all its input ports";

      throw incompatible_pipeline_exception(reason);
    }
  }
}

sync_scheduler
::~sync_scheduler()
{
}

void
sync_scheduler
::_start()
{
  d->thread = boost::thread(boost::bind(&priv::run, d.get(), pipeline()));
}

void
sync_scheduler
::_wait()
{
  d->thread.join();
}

void
sync_scheduler
::_pause()
{
  d->mut.lock();
}

void
sync_scheduler
::_resume()
{
  d->mut.unlock();
}

void
sync_scheduler
::_stop()
{
  d->thread.interrupt();
}

sync_scheduler::priv
::priv()
  : thread()
  , mut()
{
}

sync_scheduler::priv
::~priv()
{
}

static process::names_t sorted_names(pipeline_t const& pipe);
static config_t monitor_edge_config();

void
sync_scheduler::priv
::run(pipeline_t const& pipe)
{
  name_thread(thread_name);

  process::names_t const names = sorted_names(pipe);
  std::queue<process_t> processes;
  std::map<process::name_t, edge_t> monitor_edges;

  config_t const edge_conf = monitor_edge_config();

  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = pipe->process_by_name(name);
    edge_t const monitor_edge = boost::make_shared<edge>(edge_conf);

    proc->connect_output_port(process::port_heartbeat, monitor_edge);
    monitor_edges[name] = monitor_edge;

    processes.push(proc);
  }

  while (!processes.empty())
  {
    shared_lock_t const lock(mut);

    (void)lock;

    boost::this_thread::interruption_point();

    process_t proc = processes.front();
    processes.pop();

    edge_t monitor_edge = monitor_edges[proc->name()];

    proc->step();

    bool proc_complete = false;

    while (monitor_edge->has_data())
    {
      edge_datum_t const edat = monitor_edge->get_datum();
      datum_t const dat = edat.datum;

      if (dat->type() == datum::complete)
      {
        proc_complete = true;
      }
    }

    if (proc_complete)
    {
      monitor_edges.erase(proc->name());
    }
    else
    {
      processes.push(proc);
    }
  }
}

namespace
{

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, process::name_t> pipeline_graph_t;
typedef boost::graph_traits<pipeline_graph_t>::vertex_descriptor vertex_t;
typedef std::deque<vertex_t> vertices_t;
typedef std::map<process::name_t, vertex_t> vertex_map_t;

}

process::names_t
sorted_names(pipeline_t const& pipe)
{
  pipeline_graph_t graph;

  // Create the graph.
  {
    vertex_map_t vertex_map;

    process::names_t const names = pipe->process_names();

    BOOST_FOREACH (process::name_t const& name, names)
    {
      vertex_t s = boost::add_vertex(graph);
      graph[s] = name;
      vertex_map[name] = s;
    }

    BOOST_FOREACH (process::name_t const& name, names)
    {
      process_t const proc = pipe->process_by_name(name);
      process::ports_t const iports = proc->input_ports();

      vertex_t const t = vertex_map[name];

      BOOST_FOREACH (process::port_t const& port, iports)
      {
        process::port_addr_t const sender = pipe->sender_for_port(name, port);

        if (sender == process::port_addr_t())
        {
          continue;
        }

        process::name_t const& sender_name = sender.first;
        process::port_t const& sender_port = sender.second;
        edge_t const edge = pipe->edge_for_connection(sender_name, sender_port,
                                                      name, port);

        if (!edge)
        {
          /// \todo Throw an exception.
          continue;
        }

        if (!edge->makes_dependency())
        {
          continue;
        }

        vertex_t const s = vertex_map[sender_name];

        boost::add_edge(s, t, graph);
      }
    }
  }

  vertices_t vertices;

  try
  {
    boost::topological_sort(graph, std::front_inserter(vertices));
  }
  catch (boost::not_a_dag const&)
  {
    /// \todo Throw an exception.
  }

  process::names_t names;

  BOOST_FOREACH (vertex_t const& vertex, vertices)
  {
    names.push_back(graph[vertex]);
  }

  return names;
}

config_t
monitor_edge_config()
{
  config_t conf = config::empty_config();

  return conf;
}

}
