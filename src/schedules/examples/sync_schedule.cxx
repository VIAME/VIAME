/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "sync_schedule.h"

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/utils.h>

#include <boost/graph/directed_graph.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <algorithm>
#include <deque>
#include <iterator>
#include <map>
#include <queue>

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
{
  d = boost::make_shared<priv>();
}

sync_schedule
::~sync_schedule()
{
}

static void run_sync(pipeline_t const& pipe);

void
sync_schedule
::start()
{
  d->thread = boost::thread(boost::bind(run_sync, pipeline()));
}

void
sync_schedule
::wait()
{
  d->thread.join();
}

void
sync_schedule
::stop()
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

      if (dat->type() == datum::DATUM_COMPLETE)
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

struct vertex_name_t
{
  typedef boost::vertex_property_tag kind;
};
typedef boost::property<vertex_name_t, process::name_t> name_property_t;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, name_property_t> pipeline_graph_t;
typedef boost::graph_traits<pipeline_graph_t>::vertex_descriptor vertex_t;
typedef std::deque<vertex_t> vertices_t;
typedef std::map<process::name_t, vertex_t> vertex_map_t;

}

process::names_t
sorted_names(pipeline_t const& pipe)
{
  pipeline_graph_t graph;

  boost::property_map<pipeline_graph_t, vertex_name_t>::type key_prop = boost::get(vertex_name_t(), graph);

  // Create the graph.
  {
    vertex_map_t vertex_map;

    process::names_t const names = pipe->process_names();

    BOOST_FOREACH (process::name_t const& name, names)
    {
      vertex_t s = boost::add_vertex(graph);
      key_prop[s] = name;
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
        edge_t const edge = pipe->edge_for_connection(sender.first, sender.second,
                                                      name, port);

        if (edge && edge->makes_dependency())
        {
          vertex_t const s = vertex_map[sender.first];

          boost::add_edge(s, t, graph);
        }
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
    names.push_back(boost::get(key_prop, vertex));
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
