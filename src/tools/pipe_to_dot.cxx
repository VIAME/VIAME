/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_util/load_pipe.h>

#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>

namespace po = boost::program_options;

static po::options_description make_options();
static void usage(po::options_description const& options);

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

int main(int argc, char* argv[])
{
  vistk::load_known_modules();

  po::options_description const desc = make_options();

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    usage(desc);
  }

  if (!vm.count("input"))
  {
    std::cerr << "Error: input not set" << std::endl;
    usage(desc);
  }

  vistk::pipeline_t pipe;

  {
    std::istream* pistr;
    std::ifstream fin;

    boost::filesystem::path const ipath = vm["input"].as<boost::filesystem::path>();

    if (ipath.native() == "-")
    {
      pistr = &std::cin;
    }
    else
    {
      fin.open(ipath.native().c_str());

      if (fin.bad())
      {
        std::cerr << "Error: Unable to open input file" << std::endl;

        return 1;
      }

      pistr = &fin;
    }

    std::istream& istr = *pistr;

    /// \todo Include paths?

    pipe = vistk::bake_pipe(istr, boost::filesystem::current_path());
  }

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return 1;
  }

  std::ostream* postr;
  std::ofstream fout;

  boost::filesystem::path const opath = vm["output"].as<boost::filesystem::path>();

  if (opath.native() == "-")
  {
    postr = &std::cout;
  }
  else
  {
    fout.open(opath.native().c_str());

    if (fout.bad())
    {
      std::cerr << "Error: Unable to open output file" << std::endl;

      return 1;
    }

    postr = &fout;
  }

  std::ostream& ostr = *postr;

  std::string const graph_name = vm["name"].as<std::string>();

  ostr << "strict digraph " << graph_name << " {" << std::endl;
  ostr << std::endl;

  vistk::process::names_t const proc_names = pipe->process_names();

  // Output nodes
  BOOST_FOREACH (vistk::process::name_t const& name, proc_names)
  {
    vistk::process_t const proc = pipe->process_by_name(name);

    ostr << "subgraph cluster_" << name << " {" << std::endl;

    ostr << style_process_subgraph << std::endl;

    ostr << std::endl;

    std::string const node_name = name + node_suffix_main;

    // Central node
    ostr << node_name << " ["
         << "label=\"" << name << "\\n:: " << proc->type() << "\","
         << style_process
         << "];" << std::endl;

    ostr << std::endl;

    // Input ports
    vistk::process::ports_t const iports = proc->input_ports();
    BOOST_FOREACH (vistk::process::port_t const& port, iports)
    {
      vistk::process::port_type_name_t const ptype = proc->input_port_type(port).get<0>();

      std::string const node_port_name = name + node_prefix_input + port;

      ostr << node_port_name << " ["
           << "label=\"" << port << "\\n:: " << ptype << "\","
           << style_input_port
           << "];" << std::endl;
      ostr << node_port_name << " -> "
           << node_name << " ["
           << style_input_port_edge
           << "];" << std::endl;
      ostr << std::endl;
    }

    ostr << std::endl;

    // Output ports
    vistk::process::ports_t const oports = proc->output_ports();
    BOOST_FOREACH (vistk::process::port_t const& port, oports)
    {
      vistk::process::port_type_name_t const ptype = proc->output_port_type(port).get<0>();

      std::string const node_port_name = name + node_prefix_output + port;

      ostr << node_port_name << " ["
           << "label=\"" << port << "\\n:: " << ptype << "\","
           << style_output_port
           << "];" << std::endl;
      ostr << node_name << " -> "
           << node_port_name << " ["
           << style_output_port_edge
           << "];" << std::endl;
      ostr << std::endl;
    }

    ostr << std::endl;

    ostr << "}" << std::endl;
  }

  ostr << std::endl;

  // Output connections
  BOOST_FOREACH (vistk::process::name_t const& name, proc_names)
  {
    vistk::process_t const proc = pipe->process_by_name(name);

    vistk::process::ports_t const oports = proc->output_ports();
    BOOST_FOREACH (vistk::process::port_t const& port, oports)
    {
      std::string const node_from_port_name = name + node_prefix_output + port;

      vistk::process::port_addrs_t const addrs = pipe->receivers_for_port(name, port);

      BOOST_FOREACH (vistk::process::port_addr_t const& addr, addrs)
      {
        std::string const node_to_port_name = addr.first + node_prefix_input + addr.second;

        ostr << node_from_port_name << " -> "
             << node_to_port_name << " ["
             << style_connection_edge
             << "];" << std::endl;
      }
    }

    ostr << std::endl;
  }

  ostr << std::endl;

  ostr << "}" << std::endl;

  return 0;
}

po::options_description
make_options()
{
  po::options_description desc;

  desc.add_options()
    ("help,h", "output help message and quit")
    ("input,i", po::value<boost::filesystem::path>(), "input path")
    ("output,o", po::value<boost::filesystem::path>()->default_value("-"), "output path")
    ("include,I", po::value<std::vector<boost::filesystem::path> >(), "configuration include path")
    ("name,n", po::value<std::string>()->default_value("(unnamed)"), "name of the graph")
  ;

  return desc;
}

void
usage(po::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(1);
}
