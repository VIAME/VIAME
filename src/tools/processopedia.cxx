/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#include <iostream>

namespace po = boost::program_options;

static po::options_description make_options();
static void usage(po::options_description const& options);

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

  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process_registry::types_t types;

  if (vm.count("type"))
  {
    types = vm["type"].as<vistk::process_registry::types_t>();
  }
  else
  {
    types = reg->types();
  }

  vistk::config_t const conf = vistk::config::empty_config();

  BOOST_FOREACH (vistk::process_registry::type_t const& type, types)
  {
    std::cout << "Process type: " << type << std::endl;
    std::cout << "  Description: " << reg->description(type) << std::endl;

    vistk::process_t const proc = reg->create_process(type, conf);

    std::cout << "  Configuration:" << std::endl;

    vistk::config::keys_t const keys = proc->available_config();

    BOOST_FOREACH (vistk::config::key_t const key, keys)
    {
      vistk::config::value_t const def = proc->config_default(key);
      vistk::config::description_t const desc = proc->config_description(key);

      std::cout << "    Name       : " << key << std::endl;
      std::cout << "    Default    : " << def << std::endl;
      std::cout << "    Description: " << desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Input ports:" << std::endl;

    vistk::process::ports_t const iports = proc->input_ports();

    BOOST_FOREACH (vistk::process::port_t const port, iports)
    {
      vistk::process::port_type_t const type = proc->input_port_type(port);
      vistk::process::port_description_t const desc = proc->input_port_description(port);

      vistk::process::port_type_name_t const type_name = type.get<0>();
      vistk::process::port_flags_t const flags = type.get<1>();

      std::string const flags_str = boost::join(flags, ", ");

      std::cout << "    Name       : " << port << std::endl;
      std::cout << "    Type       : " << type_name << std::endl;
      std::cout << "    Flags      : " << flags_str << std::endl;
      std::cout << "    Description: " << desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Output ports:" << std::endl;

    vistk::process::ports_t const oports = proc->output_ports();

    BOOST_FOREACH (vistk::process::port_t const port, oports)
    {
      vistk::process::port_type_t const type = proc->output_port_type(port);
      vistk::process::port_description_t const desc = proc->output_port_description(port);

      vistk::process::port_type_name_t const type_name = type.get<0>();
      vistk::process::port_flags_t const flags = type.get<1>();

      std::string const flags_str = boost::join(flags, ", ");

      std::cout << "    Name       : " << port << std::endl;
      std::cout << "    Type       : " << type_name << std::endl;
      std::cout << "    Flags      : " << flags_str << std::endl;
      std::cout << "    Description: " << desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;
  }

  return 0;
}

po::options_description
make_options()
{
  po::options_description desc;

  desc.add_options()
    ("help,h", "output help message and quit")
    ("type,t", po::value<vistk::process_registry::types_t>(), "type to describe")
  ;

  return desc;
}

void
usage(po::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(1);
}
