/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "sync_schedule.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/schedule_exception.h>
#include <vistk/pipeline/utils.h>

#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <deque>
#include <iterator>
#include <map>
#include <queue>

/**
 * \file sync_schedule.cxx
 *
 * \brief Implementation of the synchronized schedule.
 */

namespace vistk
{

static thread_name_t const thread_name = thread_name_t("sync_schedule");

class sync_schedule::priv
{
  public:
    priv();
    ~priv();

    boost::thread thread;
};

sync_schedule
::sync_schedule(config_t const& config, pipeline_t const& pipe)
  : schedule(config, pipe)
  , d(new priv)
{
  pipeline_t const p = pipeline();
  process::names_t const names = p->process_names();

  BOOST_FOREACH (process::name_t const& name, names)
  {
    process_t const proc = p->process_by_name(name);
    process::constraints_t const consts = proc->constraints();

    {
      process::constraints_t::const_iterator const i = consts.find(process::constraint_unsync_output);

      if (i != consts.end())
      {
        std::string const reason = "The process \'" + name + "\' does not output "
                                   "consistent data across all its output ports";

        throw incompatible_pipeline_exception(reason);
      }
    }

    {
      process::constraints_t::const_iterator const i = consts.find(process::constraint_unsync_input);

      if (i != consts.end())
      {
        std::string const reason = "The process \'" + name + "\' does not expect "
                                   "consistent data across all its input ports";

        throw incompatible_pipeline_exception(reason);
      }
    }
  }
}

sync_schedule
::~sync_schedule()
{
}

static void run_sync(pipeline_t const& pipe);

void
sync_schedule
::_start()
{
  d->thread = boost::thread(boost::bind(run_sync, pipeline()));
}

void
sync_schedule
::_wait()
{
  d->thread.join();
}

void
sync_schedule
::_stop()
{
  d->thread.interrupt();
}

sync_schedule::priv
::priv()
{
}

sync_schedule::priv
::~priv()
{
}

static process::names_t sorted_names(pipeline_t const& pipe);
static config_t monitor_edge_config();

void
run_sync(pipeline_t const& pipe)
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
    process_t proc = processes.front();
    processes.pop();

    edge_t monitor_edge = monitor_edges[proc->name()];

    proc->step();

    bool proc_complete = false;

    while (monitor_edge->has_data())
    {
      edge_datum_t const edat = monitor_edge->get_datum();
      datum_t const dat = edat.get<0>();

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

    boost::this_thread::interruption_point();
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
  catch (boost::not_a_dag&)
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
