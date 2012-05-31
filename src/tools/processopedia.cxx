/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>

#include <vistk/config.h>

#include <tools/helpers/typed_value_desc.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <string>

#include <cstdlib>

namespace po = boost::program_options;

static po::options_description make_options();
static void VISTK_NO_RETURN usage(po::options_description const& options);

int
main(int argc, char* argv[])
{
  vistk::load_known_modules();

  po::options_description const desc = make_options();

  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  }
  catch (po::unknown_option& e)
  {
    std::cerr << "Error: unknown option " << e.get_option_name() << std::endl;

    usage(desc);
  }
  po::notify(vm);

  if (vm.count("help"))
  {
    usage(desc);
  }

  vistk::process_registry_t reg = vistk::process_registry::self();

  vistk::process::types_t types;

  if (vm.count("type"))
  {
    types = vm["type"].as<vistk::process::types_t>();
  }
  else
  {
    types = reg->types();
  }

  if (vm.count("list"))
  {
    BOOST_FOREACH (vistk::process::type_t const& type, types)
    {
      std::cout << type << std::endl;
    }

    return EXIT_SUCCESS;
  }

  BOOST_FOREACH (vistk::process::type_t const& proc_type, types)
  {
    if (!vm.count("detail"))
    {
      try
      {
        std::cout << proc_type << ": " << reg->description(proc_type) << std::endl;
      }
      catch (vistk::no_such_process_type_exception& e)
      {
        std::cerr << "Error: " << e.what() << std::endl;
      }

      continue;
    }

    std::cout << "Process type: " << proc_type << std::endl;
    std::cout << "  Description: " << reg->description(proc_type) << std::endl;

    vistk::process_t proc_m;

    try
    {
      proc_m = reg->create_process(proc_type);
    }
    catch (vistk::no_such_process_type_exception& e)
    {
      std::cerr << "Error: " << e.what() << std::endl;

      continue;
    }

    vistk::process_t const proc = proc_m;

    vistk::process::constraints_t const constraints = proc->constraints();
    std::string const constraints_str = boost::join(constraints, ", ");

    std::cout << "  Constraints: " << constraints_str << std::endl;

    std::cout << "  Configuration:" << std::endl;

    vistk::config::keys_t const keys = proc->available_config();

    BOOST_FOREACH (vistk::config::key_t const& key, keys)
    {
      vistk::process::conf_info_t const info = proc->config_info(key);

      vistk::config::value_t const& def = info->def;
      vistk::config::description_t const& conf_desc = info->description;

      std::cout << "    Name       : " << key << std::endl;
      std::cout << "    Default    : " << def << std::endl;
      std::cout << "    Description: " << conf_desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Input ports:" << std::endl;

    vistk::process::ports_t const iports = proc->input_ports();

    BOOST_FOREACH (vistk::process::port_t const& port, iports)
    {
      vistk::process::port_info_t const info = proc->input_port_info(port);

      vistk::process::port_type_t const& type = info->type;
      vistk::process::port_flags_t const& flags = info->flags;
      vistk::process::port_description_t const& port_desc = info->description;

      std::string const flags_str = boost::join(flags, ", ");

      std::cout << "    Name       : " << port << std::endl;
      std::cout << "    Type       : " << type << std::endl;
      std::cout << "    Flags      : " << flags_str << std::endl;
      std::cout << "    Description: " << port_desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Output ports:" << std::endl;

    vistk::process::ports_t const oports = proc->output_ports();

    BOOST_FOREACH (vistk::process::port_t const& port, oports)
    {
      vistk::process::port_info_t const info = proc->output_port_info(port);

      vistk::process::port_type_t const& type = info->type;
      vistk::process::port_flags_t const& flags = info->flags;
      vistk::process::port_description_t const& port_desc = info->description;

      std::string const flags_str = boost::join(flags, ", ");

      std::cout << "    Name       : " << port << std::endl;
      std::cout << "    Type       : " << type << std::endl;
      std::cout << "    Flags      : " << flags_str << std::endl;
      std::cout << "    Description: " << port_desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}

po::options_description
make_options()
{
  po::options_description desc;

  desc.add_options()
    ("help,h", "output help message and quit")
    ("type,t", po::value_desc<vistk::process::types_t>()->metavar("TYPE"), "type to describe")
    ("list,l", "simply list types")
    ("detail,d", "output detailed information")
  ;

  return desc;
}

void
usage(po::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(EXIT_FAILURE);
}
