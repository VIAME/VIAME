/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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

#include "sync_scheduler.h"

#include <vital/config/config_block.h>

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

#include <deque>
#include <iterator>
#include <map>
#include <queue>
#include <memory>

/**
 * \file sync_scheduler.cxx
 *
 * \brief Implementation of the synchronized scheduler.
 */

namespace sprokit {

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


// ============================================================================
sync_scheduler
::sync_scheduler(pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
  : scheduler(pipe, config)
  , d(new priv)
{
  m_logger = kwiver::vital::get_logger( "scheduler.sync" );

  pipeline_t const p = pipeline();

  processes_t procs = p->get_python_processes();
  if( ! procs.empty() )
  {
    std::stringstream str;
    str << "This pipeline contains the following python processes which are not supported by this scheduler.\n";
    for ( auto proc : procs )
    {
      str << "      \"" << proc->name() << "\" of type \"" << proc->type() << "\"\n";
    }

    VITAL_THROW( incompatible_pipeline_exception, str.str());
  }

  process::names_t const names = p->process_names();

  for (process::name_t const& name : names)
  {
    auto proc = p->process_by_name(name);
    process::properties_t const consts = proc->properties();

    if (consts.count(process::property_unsync_output))
    {
      std::string const reason = "The process \'" + name + "\' does not output "
                                 "consistent data across all its output ports";

      VITAL_THROW( incompatible_pipeline_exception,
                   reason);
    }

    if (consts.count(process::property_unsync_input))
    {
      std::string const reason = "The process \'" + name + "\' does not expect "
                                 "consistent data across all its input ports";

      VITAL_THROW( incompatible_pipeline_exception,
                   reason);
    }
  }
}


sync_scheduler
::~sync_scheduler()
{
  shutdown();
}


// ----------------------------------------------------------------------------
void
sync_scheduler
::_start()
{
  d->thread = boost::thread(std::bind(&priv::run, d.get(), pipeline()));
}


// ----------------------------------------------------------------------------
void
sync_scheduler
::_wait()
{
  d->thread.join();
}


// ----------------------------------------------------------------------------
void
sync_scheduler
::_pause()
{
  d->mut.lock();
}


// ----------------------------------------------------------------------------
void
sync_scheduler
::_resume()
{
  d->mut.unlock();
}


// ----------------------------------------------------------------------------
void
sync_scheduler
::_stop()
{
  d->thread.interrupt();
}


// ============================================================================
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
static kwiver::vital::config_block_sptr monitor_edge_config();

// ----------------------------------------------------------------------------
void
sync_scheduler::priv
::run(pipeline_t const& pipe)
{
  name_thread(thread_name);

  process::names_t const names = sorted_names(pipe);
  std::queue<process_t> processes;
  std::map<process::name_t, edge_t> monitor_edges;

  kwiver::vital::config_block_sptr const edge_conf = monitor_edge_config();

  // Loop over all processes and make a connection to the heartbeat output port.
  // This port is connected to an edge that is monitored while running the pipeline.

  for (process::name_t const& name : names)
  {
    process_t const proc = pipe->process_by_name(name);
    edge_t const monitor_edge = std::make_shared<edge>(edge_conf);

    proc->connect_output_port(process::port_heartbeat, monitor_edge);
    monitor_edges[name] = monitor_edge;

    processes.push(proc);
  } // end for

  // Run the pipeline
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

    // Check the monitor edge to see how the processes is doing.  If a
    // "complete" packet is found, then this process has terminated.
    while (monitor_edge->has_data())
    {
      edge_datum_t const edat = monitor_edge->get_datum();
      datum_t const dat = edat.datum;

      if (dat->type() == datum::complete)
      {
        proc_complete = true;
      }
    }

    // If the process is complete, remove the monitor edge and don't
    // put the process back on the active list.
    if (proc_complete)
    {
      monitor_edges.erase(proc->name());
    }
    else
    {
      // Still active, goes back on list.
      processes.push(proc);
    }
  }
}


// ----------------------------------------------------------------------------
namespace {

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, process::name_t> pipeline_graph_t;
typedef boost::graph_traits<pipeline_graph_t>::vertex_descriptor vertex_t;
typedef std::deque<vertex_t> vertices_t;
typedef std::map<process::name_t, vertex_t> vertex_map_t;

} // end anonymous


// ----------------------------------------------------------------------------
process::names_t
sorted_names(pipeline_t const& pipe)
{
  kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "scheduler.sync" ) );

  pipeline_graph_t graph;

  // Create the graph.
  {
    vertex_map_t vertex_map;

    process::names_t const names = pipe->process_names();

    for (process::name_t const& name : names)
    {
      vertex_t s = boost::add_vertex(graph);
      graph[s] = name;
      vertex_map[name] = s;
    }

    for (process::name_t const& name : names)
    {
      process_t const proc = pipe->process_by_name(name);
      process::ports_t const iports = proc->input_ports();

      vertex_t const t = vertex_map[name];

      for (process::port_t const& port : iports)
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
          // This means that there is no edge connecting the two processes.
          LOG_ERROR( logger, "Edge not found from " << sender_name << "." << sender_port
                    << " to " << name << "." << port );
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
    LOG_ERROR( logger, "Pipeline is not a DAG" );
  }

  process::names_t names;

  for (vertex_t const& vertex : vertices)
  {
    names.push_back(graph[vertex]);
  }

  return names;
}


// ----------------------------------------------------------------------------
kwiver::vital::config_block_sptr
monitor_edge_config()
{
  kwiver::vital::config_block_sptr conf = kwiver::vital::config_block::empty_config();

  // empty config created a default edge.

  return conf;
}

}
