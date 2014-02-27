/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include <sprokit/tools/tool_main.h>
#include <sprokit/tools/tool_usage.h>

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/process_registry_exception.h>

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
sprokit_tool_main(int argc, char const* argv[])
{
  sprokit::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(sprokit::tool_common_options())
    .add(processopedia_options());

  boost::program_options::variables_map const vm = sprokit::tool_parse(argc, argv, desc);

  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::types_t types;

  if (vm.count("type"))
  {
    types = vm["type"].as<sprokit::process::types_t>();
  }
  else
  {
    types = reg->types();
  }

  if (vm.count("list"))
  {
    BOOST_FOREACH (sprokit::process::type_t const& type, types)
    {
      std::cout << type << std::endl;
    }

    return EXIT_SUCCESS;
  }

  bool const hidden = (0 != vm.count("hidden"));

  BOOST_FOREACH (sprokit::process::type_t const& proc_type, types)
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
    catch (sprokit::no_such_process_type_exception const& e)
    {
      std::cerr << "Error: " << e.what() << std::endl;

      continue;
    }

    sprokit::process_t const proc = reg->create_process(proc_type, sprokit::process::name_t());

    sprokit::process::properties_t const properties = proc->properties();
    std::string const properties_str = boost::join(properties, ", ");

    std::cout << "  Properties: " << properties_str << std::endl;

    std::cout << "  Configuration:" << std::endl;

    sprokit::config::keys_t const keys = proc->available_config();

    BOOST_FOREACH (sprokit::config::key_t const& key, keys)
    {
      if (!hidden && boost::starts_with(key, hidden_prefix))
      {
        continue;
      }

      sprokit::process::conf_info_t const info = proc->config_info(key);

      sprokit::config::value_t const& def = info->def;
      sprokit::config::description_t const& conf_desc = info->description;
      bool const& tunable = info->tunable;
      char const* const tunable_str = tunable ? "yes" : "no";

      std::cout << "    Name       : " << key << std::endl;
      std::cout << "    Default    : " << def << std::endl;
      std::cout << "    Description: " << conf_desc << std::endl;
      std::cout << "    Tunable    : " << tunable_str << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Input ports:" << std::endl;

    sprokit::process::ports_t const iports = proc->input_ports();

    BOOST_FOREACH (sprokit::process::port_t const& port, iports)
    {
      if (!hidden && boost::starts_with(port, hidden_prefix))
      {
        continue;
      }

      sprokit::process::port_info_t const info = proc->input_port_info(port);

      sprokit::process::port_type_t const& type = info->type;
      sprokit::process::port_flags_t const& flags = info->flags;
      sprokit::process::port_description_t const& port_desc = info->description;

      std::string const flags_str = boost::join(flags, ", ");

      std::cout << "    Name       : " << port << std::endl;
      std::cout << "    Type       : " << type << std::endl;
      std::cout << "    Flags      : " << flags_str << std::endl;
      std::cout << "    Description: " << port_desc << std::endl;
      std::cout << std::endl;
    }

    std::cout << "  Output ports:" << std::endl;

    sprokit::process::ports_t const oports = proc->output_ports();

    BOOST_FOREACH (sprokit::process::port_t const& port, oports)
    {
      if (!hidden && boost::starts_with(port, hidden_prefix))
      {
        continue;
      }

      sprokit::process::port_info_t const info = proc->output_port_info(port);

      sprokit::process::port_type_t const& type = info->type;
      sprokit::process::port_flags_t const& flags = info->flags;
      sprokit::process::port_description_t const& port_desc = info->description;

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
    ("type,t", boost::program_options::value<sprokit::process::types_t>()->value_name("TYPE"), "type to describe")
    ("list,l", "simply list types")
    ("hidden,H", "show hidden properties")
    ("detail,d", "output detailed information")
  ;

  return desc;
}
