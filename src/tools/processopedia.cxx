/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/process.h>
#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/process_registry_exception.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/foreach.hpp>

#include <iostream>
#include <string>

#include <cstdlib>

static std::string const hidden_prefix = "_";

static boost::program_options::options_description processopedia_options();

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(tool_common_options())
    .add(processopedia_options());

  boost::program_options::variables_map const vm = tool_parse(argc, argv, desc);

  vistk::process_registry_t const reg = vistk::process_registry::self();

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

  bool const hidden = vm.count("hidden");

  BOOST_FOREACH (vistk::process::type_t const& proc_type, types)
  {
    try
    {
      if (!vm.count("detail"))
      {
        std::cout << proc_type << ": " << reg->description(proc_type) << std::endl;

        continue;
      }

      std::cout << "Process type: " << proc_type << std::endl;
      std::cout << "  Description: " << reg->description(proc_type) << std::endl;
    }
    catch (vistk::no_such_process_type_exception const& e)
    {
      std::cerr << "Error: " << e.what() << std::endl;

      continue;
    }

    vistk::process_t const proc = reg->create_process(proc_type, vistk::process::name_t());

    vistk::process::properties_t const properties = proc->properties();
    std::string const properties_str = boost::join(properties, ", ");

    std::cout << "  Properties: " << properties_str << std::endl;

    std::cout << "  Configuration:" << std::endl;

    vistk::config::keys_t const keys = proc->available_config();

    BOOST_FOREACH (vistk::config::key_t const& key, keys)
    {
      if (!hidden && boost::starts_with(key, hidden_prefix))
      {
        continue;
      }

      vistk::process::conf_info_t const info = proc->config_info(key);

      vistk::config::value_t const& def = info->def;
      vistk::config::description_t const& conf_desc = info->description;
      bool const& tunable = info->tunable;

      std::cout << "    Name       : " << key << std::endl;
      std::cout << "    Default    : " << def << std::endl;
      std::cout << "    Description: " << conf_desc << std::endl;
      std::cout << "    Tunable    : " << tunable << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Input ports:" << std::endl;

    vistk::process::ports_t const iports = proc->input_ports();

    BOOST_FOREACH (vistk::process::port_t const& port, iports)
    {
      if (!hidden && boost::starts_with(port, hidden_prefix))
      {
        continue;
      }

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
      if (!hidden && boost::starts_with(port, hidden_prefix))
      {
        continue;
      }

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

boost::program_options::options_description
processopedia_options()
{
  boost::program_options::options_description desc;

  desc.add_options()
    ("type,t", boost::program_options::value<vistk::process::types_t>()->value_name("TYPE"), "type to describe")
    ("list,l", "simply list types")
    ("hidden,H", "show hidden properties")
    ("detail,d", "output detailed information")
  ;

  return desc;
}
