/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_registry.h>
#include <vistk/pipeline/pipeline.h>

#include <vistk/utilities/path.h>

#include <boost/program_options/value_semantic.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>

namespace po = boost::program_options;

static vistk::config::key_t const scheduler_block = vistk::config::key_t("_scheduler");

static po::options_description make_options();

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  po::options_description const desc = make_options();

  po::variables_map const vm = tool_parse(argc, argv, desc);

  if (vm.count("help"))
  {
    tool_usage(EXIT_SUCCESS, desc);
  }

  if (!vm.count("pipeline"))
  {
    std::cerr << "Error: pipeline not set" << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  vistk::pipeline_t pipe;

  vistk::config_t conf;

  {
    std::istream* pistr;
    std::ifstream fin;

    vistk::path_t const ipath = vm["pipeline"].as<vistk::path_t>();

    if (ipath.native() == vistk::path_t("-"))
    {
      pistr = &std::cin;
    }
    else
    {
      fin.open(ipath.native().c_str());

      if (!fin.good())
      {
        std::cerr << "Error: Unable to open input file" << std::endl;

        return EXIT_FAILURE;
      }

      pistr = &fin;
    }

    std::istream& istr = *pistr;

    /// \todo Include paths?

    pipeline_builder builder;

    builder.load_pipeline(istr);

    // Load supplemental configuration files.
    if (vm.count("config"))
    {
      vistk::paths_t const configs = vm["config"].as<vistk::paths_t>();

      std::for_each(configs.begin(), configs.end(), boost::bind(&pipeline_builder::load_supplement, &builder, _1));
    }

    // Insert lone setting variables from the command line.
    if (vm.count("setting"))
    {
      std::vector<std::string> const settings = vm["setting"].as<std::vector<std::string> >();

      std::for_each(settings.begin(), settings.end(), boost::bind(&pipeline_builder::add_setting, &builder, _1));
    }

    pipe = builder.pipeline();

    conf = builder.config();
  }

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  pipe->setup_pipeline();

  vistk::scheduler_registry::type_t scheduler_type = vistk::scheduler_registry::default_type;

  if (vm.count("scheduler"))
  {
    scheduler_type = vm["scheduler"].as<vistk::scheduler_registry::type_t>();
  }

  vistk::config_t const scheduler_config = conf->subblock(scheduler_block + vistk::config::block_sep + scheduler_type);

  vistk::scheduler_registry_t reg = vistk::scheduler_registry::self();

  vistk::scheduler_t scheduler = reg->create_scheduler(scheduler_type, pipe, scheduler_config);

  if (!scheduler)
  {
    std::cerr << "Error: Unable to create scheduler" << std::endl;

    return EXIT_FAILURE;
  }

  scheduler->start();
  scheduler->wait();

  return EXIT_SUCCESS;
}

po::options_description
make_options()
{
  po::options_description desc;

  desc.add_options()
    ("help,h", "output help message and quit")
    ("pipeline,p", po::value<vistk::path_t>()->value_name("FILE"), "pipeline")
    ("config,c", po::value<vistk::paths_t>()->value_name("FILE"), "supplemental configuration file")
    ("setting,s", po::value<std::vector<std::string> >()->value_name("VAR=VALUE"), "additional configuration")
    ("include,I", po::value<vistk::paths_t>()->value_name("DIR"), "configuration include path")
    ("scheduler,S", po::value<vistk::scheduler_registry::type_t>()->value_name("TYPE"), "scheduler type")
  ;

  return desc;
}
