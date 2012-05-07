/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "export_dot.h"
#include "export_dot_exception.h"

#include <vistk/pipeline/pipeline.h>
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

  // Output nodes
  BOOST_FOREACH (process::name_t const& name, proc_names)
  {
    process_t const proc = pipe->process_by_name(name);

    ostr << "subgraph \"cluster_" << name << "\" {" << std::endl;

    ostr << style_process_subgraph << std::endl;

    ostr << std::endl;

    std::string const node_name = name + node_suffix_main;

    // Central node
    ostr << "\"" << node_name << "\" ["
            "label=\"" << name << "\\n:: " << proc->type() << "\","
         << style_process
         << "];" << std::endl;

    ostr << std::endl;

    // Input ports
    process::ports_t const iports = proc->input_ports();
    BOOST_FOREACH (process::port_t const& port, iports)
    {
      process::port_type_t const ptype = proc->input_port_info(port)->type;

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
    process::ports_t const oports = proc->output_ports();
    BOOST_FOREACH (process::port_t const& port, oports)
    {
      process::port_type_t const ptype = proc->output_port_info(port)->type;

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

}
