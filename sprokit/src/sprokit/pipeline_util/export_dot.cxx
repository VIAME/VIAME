// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "export_dot.h"
#include "export_dot_exception.h"

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>

#include <map>
#include <ostream>
#include <queue>
#include <set>
#include <utility>
#include <vector>
#include <memory>
#include <functional>

/// \todo Implement a depth option to suppress recursion into too many clusters.
/// \todo Improve the color scheme.

/**
 * \file export_dot.cxx
 *
 * \brief Implementation of dot exporting.
 */

namespace sprokit {
namespace {

static std::string const node_suffix_main = "_main";
static std::string const node_prefix_input = "_input_";
static std::string const node_prefix_output = "_output_";

static std::string const style_global = "clusterrank=local;";

static std::string const style_process_subgraph =
  "color=lightgray;style=filled;fillcolor=lightgray;";
static std::string const style_process = "shape=ellipse,rank=same";
static std::string const style_process_subgraph_rst = "color=lightgray;";
static std::string const style_process_rst =
  "shape=ellipse,rank=same,fontcolor=blue,fontsize=16,href=\"";

static std::string const style_cluster =
  "labelloc=t;labeljust=l;color=black;style=filled;fillcolor=gray;";

static std::string const style_port = "shape=none,height=0,width=0,fontsize=7";
static std::string const style_port_rst = "shape=none,height=0,width=0,fontsize=12";
static std::string const style_port_edge = "arrowhead=none,color=black";

static std::string const style_map_edge = "color=\"#808080\"";

static std::string const style_input_subgraph = "clusterrank=local;rankdir=BT;";
static std::string const style_input_port = style_port;
static std::string const style_input_port_rst = style_port_rst;
static std::string const style_input_port_edge = style_port_edge;
static std::string const style_input_map_edge = style_map_edge;

static std::string const style_output_subgraph = "clusterrank=local;rankdir=BT;";
static std::string const style_output_port = style_port;
static std::string const style_output_port_rst = style_port_rst;
static std::string const style_output_port_edge = style_port_edge;
static std::string const style_output_map_edge = style_map_edge;
static std::string const style_connection_edge = "minlen=1,color=black,weight=1";

typedef enum
{
  name_process,
  name_cluster
} name_type_t;

typedef std::pair< process::name_t, name_type_t > name_info_t;
typedef std::vector< name_info_t > name_infos_t;
typedef std::map< process::name_t, name_infos_t > parent_names_t;

typedef std::function< void ( ) > callback_t;

}

static void output_cluster( std::ostream&           ostr,
                            process::name_t const&  name,
                            pipeline_t const&       pipe,
                            parent_names_t const&   parent_map,
                            std::string const&      link_prefix );

// ----------------------------------------------------------------------------
void
export_dot( std::ostream&       ostr,
            pipeline_t const&   pipe,
            std::string const&  graph_name,
            std::string const&  link_prefix )
{
  if ( ! pipe )
  {
    VITAL_THROW( null_pipeline_export_dot_exception );
  }

  ostr << "strict digraph \"" << graph_name << "\" {" << std::endl;
  ostr << style_global << std::endl;

  process::names_t const proc_names = pipe->process_names();
  process::names_t const cluster_names = pipe->cluster_names();

  parent_names_t parent_map;

  for ( process::name_t const& name : proc_names )
  {
    if ( name.empty() )
    {
      VITAL_THROW( empty_name_export_dot_exception );
    }

    process::name_t const parent = pipe->parent_cluster( name );

    name_info_t const info = name_info_t( name, name_process );

    parent_map[parent].push_back( info );
  }

  for ( process::name_t const& name : cluster_names )
  {
    if ( name.empty() )
    {
      VITAL_THROW( empty_name_export_dot_exception );
    }

    process::name_t const parent = pipe->parent_cluster( name );

    name_info_t const info = name_info_t( name, name_cluster );

    parent_map[parent].push_back( info );
  }

  ostr << std::endl;

  output_cluster( ostr, process::name_t(), pipe, parent_map, link_prefix );

  // Output connections
  for ( process::name_t const& name : proc_names )
  {
    process_t const proc = pipe->process_by_name( name );

    process::ports_t const oports = proc->output_ports();
    for ( process::port_t const& port : oports )
    {
      std::string const node_from_port_name = name + node_prefix_output + port;

      process::port_addrs_t const addrs = pipe->connections_from_addr( name, port );

      for ( process::port_addr_t const& addr : addrs )
      {
        process::name_t const& recv_name = addr.first;
        process::port_t const& recv_port = addr.second;

        std::string const node_to_port_name = recv_name + node_prefix_input + recv_port;

        ostr  << "\"" << node_from_port_name << "\" -> "
              << "\"" << node_to_port_name << "\" ["
              << style_connection_edge
              << "];" << std::endl;
      }
    }
  }

  for ( process::name_t const& name : cluster_names )
  {
    process_t const proc = pipe->cluster_by_name( name );

    process::ports_t const oports = proc->output_ports();
    for ( process::port_t const& port : oports )
    {
      std::string const node_from_port_name = name + node_prefix_output + port;

      process::port_addrs_t const addrs = pipe->connections_from_addr( name, port );

      for ( process::port_addr_t const& addr : addrs )
      {
        process::name_t const& recv_name = addr.first;
        process::port_t const& recv_port = addr.second;

        std::string const node_to_port_name = recv_name + node_prefix_input + recv_port;

        ostr  << "\"" << node_from_port_name << "\" -> "
              << "\"" << node_to_port_name << "\" ["
              << style_connection_edge
              << "];" << std::endl;
      }
    }
  }

  ostr << std::endl;

  ostr << "}" << std::endl;
} // export_dot

// ----------------------------------------------------------------------------
void
export_dot( std::ostream& ostr, pipeline_t const& pipe, std::string const& graph_name )
{
  export_dot( ostr, pipe, graph_name, "" );
}

// ----------------------------------------------------------------------------
void
export_dot( std::ostream& ostr, process_cluster_t const& cluster, std::string const& graph_name )
{
  if ( ! cluster )
  {
    VITAL_THROW( null_cluster_export_dot_exception );
  }

  pipeline_t const pipe = std::make_shared< pipeline >();

  pipe->add_process( cluster );

  export_dot( ostr, pipe, graph_name );
}

static void output_process( std::ostream&       ostr,
                            process_t const&    process,
                            std::string const&  link_prefix );

static void output_process_cluster( std::ostream&             ostr,
                                    process_cluster_t const&  cluster,
                                    callback_t const&         output_children );

// ----------------------------------------------------------------------------
void
output_cluster( std::ostream&           ostr,
                process::name_t const&  name,
                pipeline_t const&       pipe,
                parent_names_t const&   parent_map,
                std::string const&      link_prefix )
{
  parent_names_t::const_iterator const i = parent_map.find( name );

  if ( i == parent_map.end() )
  {
    std::string const reason = "Internal: Failed to keep track of parent "
                               "of " + name;

    throw std::logic_error( reason );
  }

  name_infos_t const& name_infos = i->second;

  for ( name_info_t const& info : name_infos )
  {
    process::name_t const& child_name = info.first;
    name_type_t const& type = info.second;

    switch ( type )
    {
    case name_process:
    {
      process_t const proc = pipe->process_by_name( child_name );

      output_process( ostr, proc, link_prefix );

      break;
    }
    case name_cluster:
    {
      process_cluster_t const proc = pipe->cluster_by_name( child_name );

      callback_t const callback = std::bind( &output_cluster,
                                             std::ref( ostr ),
                                             child_name,
                                             pipe,
                                             parent_map,
                                             link_prefix );

      output_process_cluster( ostr, proc, callback );

      break;
    }
    default:
      break;
    }
  }
} // output_cluster

// ----------------------------------------------------------------------------
void
output_process( std::ostream& ostr, process_t const& process, std::string const& link_prefix )
{
  process::name_t const& name = process->name();
  process::type_t const& type = process->type();

  bool const has_link = link_prefix != "";

  ostr << "subgraph \"cluster_" << name << "\" {" << std::endl
       << ( has_link ? style_process_subgraph_rst : style_process_subgraph ) << std::endl
       << std::endl;

  std::string const node_name = name + node_suffix_main;

  // Central node
  if ( has_link )
  {
    ostr  << "\"" << node_name << "\" [label=<<u>" << name << "<br/>:: " << type << "</u>>,"
          << style_process_rst << link_prefix << type << ".html"
          << "\"];" << std::endl;
  }
  else
  {
    ostr  << "\"" << node_name << "\" [label=\"" << name << "\\n:: " << type << "\","
          << style_process
          << "];" << std::endl;
  }

  ostr << std::endl;

  // Input ports
  process::ports_t const iports = process->input_ports();
  for ( process::port_t const& port : iports )
  {
    process::port_type_t const ptype = process->input_port_info( port )->type;

    std::string const node_port_name = name + node_prefix_input + port;

    ostr  << "\"" << node_port_name << "\" [label=\"" << port << "\\n:: " << ptype << "\","
          << ( has_link ? style_input_port_rst : style_input_port )
          << "];" << std::endl
          << "\"" << node_port_name << "\" -> "
          << "\"" << node_name << "\" ["
          << style_input_port_edge
          << "];" << std::endl;
  }

  ostr << std::endl;

  // Output ports
  process::ports_t const oports = process->output_ports();
  for ( process::port_t const& port : oports )
  {
    process::port_type_t const ptype = process->output_port_info( port )->type;

    std::string const node_port_name = name + node_prefix_output + port;

    ostr  << "\"" << node_port_name << "\" [label=\"" << port << "\\n:: " << ptype << "\","
          << ( has_link ? style_output_port_rst : style_output_port )
          << "];" << std::endl
          << "\"" << node_name << "\" -> "
          << "\"" << node_port_name << "\" ["
          << style_output_port_edge
          << "];" << std::endl;
  } // end for

  ostr  << std::endl
        << "}" << std::endl
        << std::endl;
} // output_process

// ----------------------------------------------------------------------------
void
output_process_cluster( std::ostream&             ostr,
                        process_cluster_t const&  cluster,
                        callback_t const&         output_children )
{
  process::name_t const& name = cluster->name();
  process::type_t const& type = cluster->type();

  ostr << "subgraph \"cluster_" << name << "\" {" << std::endl
       << style_cluster << std::endl
       << std::endl;

  typedef std::set< process::port_t > unique_ports_t;

  ostr << "subgraph \"cluster_" << name << "_inputs\" {" << std::endl
       << style_input_subgraph << std::endl
       << std::endl;

  // Input ports
  process::connections_t const input_mappings = cluster->input_mappings();
  unique_ports_t input_ports;
  for ( process::connection_t const& input_mapping : input_mappings )
  {
    process::port_addr_t const& upstream_addr = input_mapping.first;

    process::port_t const& port = upstream_addr.second;

    std::string const node_port_name = name + node_prefix_input + port;

    if ( input_ports.count( port ) )
    {
      continue;
    }

    ostr  << "\"" << node_port_name << "\" [label=\"" << port << "\","
          << style_input_port
          << "];" << std::endl;

    input_ports.insert( port );
  }

  ostr << std::endl
       << "}" << std::endl
       << std::endl;

  for ( process::connection_t const& input_mapping : input_mappings )
  {
    process::port_addr_t const& upstream_addr = input_mapping.first;
    process::port_addr_t const& downstream_addr = input_mapping.second;

    process::port_t const& port = upstream_addr.second;

    std::string const node_port_name = name + node_prefix_input + port;

    // Connect mapped ports.
    process::name_t const& mapped_name = downstream_addr.first;
    process::port_t const& mapped_port = downstream_addr.second;

    std::string const mapped_node_name = mapped_name + node_prefix_input + mapped_port;

    ostr  << "\"" << node_port_name << "\" -> "
          << "\"" << mapped_node_name << "\" ["
          << style_input_map_edge
          << "];" << std::endl
          << std::endl;
  } // end for

  ostr  << "subgraph \"cluster_" << name << "_outputs\" {" << std::endl
        << style_output_subgraph << std::endl
        << std::endl;

  // Output ports
  process::connections_t const output_mappings = cluster->output_mappings();
  unique_ports_t output_ports;
  for ( process::connection_t const& output_mapping : output_mappings )
  {
    process::port_addr_t const& downstream_addr = output_mapping.second;

    process::port_t const& port = downstream_addr.second;

    std::string const node_port_name = name + node_prefix_output + port;

    if ( output_ports.count( port ) )
    {
      continue;
    }

    ostr  << "\"" << node_port_name << "\" ["
          << "label=\"" << port << "\","
          << style_output_port
          << "];" << std::endl;

    output_ports.insert( port );
  }

  ostr  << std::endl
        << "}" << std::endl
        << std::endl;

  for ( process::connection_t const& output_mapping : output_mappings )
  {
    process::port_addr_t const& upstream_addr = output_mapping.first;
    process::port_addr_t const& downstream_addr = output_mapping.second;

    process::port_t const& port = downstream_addr.second;

    std::string const node_port_name = name + node_prefix_output + port;

    // Connect mapped port.
    process::name_t const& mapped_name = upstream_addr.first;
    process::port_t const& mapped_port = upstream_addr.second;

    std::string const mapped_node_name = mapped_name + node_prefix_output + mapped_port;

    ostr  << "\"" << mapped_node_name << "\" -> "
          << "\"" << node_port_name << "\" ["
          << style_output_map_edge
          << "];" << std::endl
          << std::endl;
  }

  // Output cluster children.
  if ( output_children )
  {
    output_children();

    // Output cluster connections
    process::connections_t const connections = cluster->internal_connections();

    for ( process::connection_t const& connection : connections )
    {
      process::port_addr_t const& upstream_addr = connection.first;
      process::port_addr_t const& downstream_addr = connection.second;

      process::name_t const& send_name = upstream_addr.first;
      process::port_t const& send_port = upstream_addr.second;

      process::name_t const& recv_name = downstream_addr.first;
      process::port_t const& recv_port = downstream_addr.second;

      std::string const node_from_port_name = send_name + node_prefix_output + send_port;
      std::string const node_to_port_name = recv_name + node_prefix_input + recv_port;

      ostr  << "\"" << node_from_port_name << "\" -> "
            << "\"" << node_to_port_name << "\" ["
            << style_connection_edge
            << "];" << std::endl;
    }
  }

  ostr  << std::endl
        << "label = \"" << name << "\\n:: " << type << "\";" << std::endl
        << std::endl
        << "}" << std::endl
        << std::endl;
} // output_process_cluster

} // end namespace
