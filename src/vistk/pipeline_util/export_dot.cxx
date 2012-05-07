/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "export_dot.h"
#include "export_dot_exception.h"

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/pipeline_exception.h>
#include <vistk/pipeline/process.h>

#include <boost/foreach.hpp>

#include <ostream>

/**
 * \file export_dot.cxx
 *
 * \brief Implementation of dot exporting.
 */

namespace vistk
{

namespace
{

static std::string const node_suffix_main = "_main";
static std::string const node_prefix_input = "_input_";
static std::string const node_prefix_output = "_output_";

static std::string const style_process_subgraph = "color=lightgray;style=filled;";
static std::string const style_process = "shape=ellipse,rank=same";
static std::string const style_port = "shape=none,height=0,width=0,fontsize=7";
static std::string const style_port_edge = "arrowhead=none,color=black";
static std::string const style_input_port = style_port;
static std::string const style_input_port_edge = style_port_edge;
static std::string const style_output_port = style_port;
static std::string const style_output_port_edge = style_port_edge;
static std::string const style_connection_edge = "minlen=1,color=black,weight=1";

}

static void output_process(std::ostream& ostr, process_t const& process);

void
export_dot(std::ostream& ostr, pipeline_t const pipe, std::string const& graph_name)
{
  if (!pipe)
  {
    throw null_pipeline_export_dot_exception();
  }

  ostr << "strict digraph \"" << graph_name << "\" {" << std::endl;
  ostr << std::endl;

  process::names_t const proc_names = pipe->process_names();
  process::names_t const groups = pipe->groups();

  // Output groups
  BOOST_FOREACH (process::name_t const& group, groups)
  {
    ostr << "subgraph \"cluster_" << group << "\" {" << std::endl;
    ostr << std::endl;

    ostr << "label = \"" << group << "\";" << std::endl;
    ostr << "labelloc = \"t\";" << std::endl;
    ostr << "labeljust = \"l\";" << std::endl;

    process::ports_t const iports = pipe->input_ports_for_group(group);

    BOOST_FOREACH (process::port_t const& port, iports)
    {
      std::string const node_port_name = group + node_prefix_input + port;

      ostr << "\"" << node_port_name << "\" ["
              "label=\"" << port << "\","
           << style_input_port
           << "];" << std::endl;
      ostr << "\"" << node_port_name << "\" ["
              "\"" << group << "\" -> "
           << style_input_port_edge
           << "];" << std::endl;

      // Connect mapped ports.
      process::port_addrs_t const addrs = pipe->mapped_group_input_ports(group, port);

      BOOST_FOREACH (process::port_addr_t const& addr, addrs)
      {
        process::name_t const& mapped_name = addr.first;
        process::port_t const& mapped_port = addr.second;

        std::string const mapped_node_name = mapped_name + node_prefix_input + mapped_port;

        ostr << "\"" << node_port_name << "\" ["
                "\"" << mapped_node_name << "\" -> "
             << style_input_port_edge
             << "];" << std::endl;
      }
    }

    process::ports_t const oports = pipe->output_ports_for_group(group);

    BOOST_FOREACH (process::port_t const& port, oports)
    {
      std::string const node_port_name = group + node_prefix_output + port;

      ostr << "\"" << node_port_name << "\" ["
              "label=\"" << port << "\","
           << style_output_port
           << "];" << std::endl;
      ostr << "\"" << group << "\" -> "
              "\"" << node_port_name << "\" ["
           << style_output_port_edge
           << "];" << std::endl;

      // Connect mapped port.
      process::port_addr_t const addr = pipe->mapped_group_output_port(group, port);

      if (addr == process::port_addr_t())
      {
        continue;
      }

      process::name_t const& mapped_name = addr.first;
      process::port_t const& mapped_port = addr.second;

      std::string const mapped_node_name = mapped_name + node_prefix_output + mapped_port;

      ostr << "\"" << mapped_node_name << "\" -> "
              "\"" << node_port_name << "\" ["
           << style_output_port_edge
           << "];" << std::endl;
    }

    ostr << std::endl;

    ostr << "}" << std::endl;

    ostr << std::endl;
  }

  // Output free processes
  BOOST_FOREACH (process::name_t const& name, proc_names)
  {
    process_t const proc = pipe->process_by_name(name);

    output_process(ostr, proc);
  }

  // Output connections
  BOOST_FOREACH (process::name_t const& name, proc_names)
  {
    process_t const proc = pipe->process_by_name(name);

    process::ports_t const oports = proc->output_ports();
    BOOST_FOREACH (process::port_t const& port, oports)
    {
      std::string const node_from_port_name = name + node_prefix_output + port;

      process::port_addrs_t const addrs = pipe->connections_from_addr(name, port);

      BOOST_FOREACH (process::port_addr_t const& addr, addrs)
      {
        process::name_t const& recv_name = addr.first;
        process::port_t const& recv_port = addr.second;

        std::string const node_to_port_name = recv_name + node_prefix_input + recv_port;

        ostr << "\"" << node_from_port_name << "\" -> "
                "\"" << node_to_port_name << "\" ["
             << style_connection_edge
             << "];" << std::endl;
      }
    }
  }

  ostr << std::endl;

  ostr << "}" << std::endl;
}

void
export_dot_setup(std::ostream& ostr, pipeline_t const pipe, std::string const& graph_name)
{
  if (!pipe)
  {
    throw null_pipeline_export_dot_exception();
  }

  if (!pipe->is_setup())
  {
    throw pipeline_not_setup_exception();
  }

  if (!pipe->setup_successful())
  {
    throw pipeline_not_ready_exception();
  }

  ostr << "strict digraph \"" << graph_name << "\" {" << std::endl;
  ostr << std::endl;

  process::names_t const proc_names = pipe->process_names();

  // Output nodes
  BOOST_FOREACH (process::name_t const& name, proc_names)
  {
    process_t const proc = pipe->process_by_name(name);

    output_process(ostr, proc);
  }

  // Output connections
  BOOST_FOREACH (process::name_t const& name, proc_names)
  {
    process_t const proc = pipe->process_by_name(name);

    process::ports_t const oports = proc->output_ports();
    BOOST_FOREACH (process::port_t const& port, oports)
    {
      std::string const node_from_port_name = name + node_prefix_output + port;

      process::port_addrs_t const addrs = pipe->receivers_for_port(name, port);

      BOOST_FOREACH (process::port_addr_t const& addr, addrs)
      {
        process::name_t const& recv_name = addr.first;
        process::port_t const& recv_port = addr.second;

        std::string const node_to_port_name = recv_name + node_prefix_input + recv_port;

        ostr << "\"" << node_from_port_name << "\" -> "
                "\"" << node_to_port_name << "\" ["
             << style_connection_edge
             << "];" << std::endl;
      }
    }
  }

  ostr << std::endl;

  ostr << "}" << std::endl;
}

void
output_process(std::ostream& ostr, process_t const& process)
{
  process::name_t const& name = process->name();
  process_registry::type_t const& type = process->type();

  ostr << "subgraph \"cluster_" << name << "\" {" << std::endl;

  ostr << style_process_subgraph << std::endl;

  ostr << std::endl;

  std::string const node_name = name + node_suffix_main;

  // Central node
  ostr << "\"" << node_name << "\" ["
          "label=\"" << name << "\\n:: " << type << "\","
       << style_process
       << "];" << std::endl;

  ostr << std::endl;

  // Input ports
  process::ports_t const iports = process->input_ports();
  BOOST_FOREACH (process::port_t const& port, iports)
  {
    process::port_type_t const ptype = process->input_port_info(port)->type;

    std::string const node_port_name = name + node_prefix_input + port;

    ostr << "\"" << node_port_name << "\" ["
            "label=\"" << port << "\\n:: " << ptype << "\","
         << style_input_port
         << "];" << std::endl;
    ostr << "\"" << node_port_name << "\" -> "
            "\"" << node_name << "\" ["
         << style_input_port_edge
         << "];" << std::endl;
  }

  ostr << std::endl;

  // Output ports
  process::ports_t const oports = process->output_ports();
  BOOST_FOREACH (process::port_t const& port, oports)
  {
    process::port_type_t const ptype = process->output_port_info(port)->type;

    std::string const node_port_name = name + node_prefix_output + port;

    ostr << "\"" << node_port_name << "\" ["
            "label=\"" << port << "\\n:: " << ptype << "\","
         << style_output_port
         << "];" << std::endl;
    ostr << "\"" << node_name << "\" -> "
            "\"" << node_port_name << "\" ["
         << style_output_port_edge
         << "];" << std::endl;
  }

  ostr << std::endl;

  ostr << "}" << std::endl;

  ostr << std::endl;
}

}
