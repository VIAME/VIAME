/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/pipeline.h>

#include <vistk/config.h>

#include <tools/helpers/typed_value_desc.h>

#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>

namespace po = boost::program_options;

static vistk::config::key_t const schedule_block = vistk::config::key_t("_schedule");

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

  if (!vm.count("pipeline"))
  {
    std::cerr << "Error: pipeline not set" << std::endl;
    usage(desc);
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

      std::for_each(configs.begin(), configs.end(), boost::bind(&pipeline_builder::load_supplement, builder, _1));
    }

    // Insert lone setting variables from the command line.
    if (vm.count("setting"))
    {
      std::vector<std::string> const settings = vm["setting"].as<std::vector<std::string> >();

      std::for_each(settings.begin(), settings.end(), boost::bind(&pipeline_builder::add_setting, builder, _1));
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

  vistk::schedule_registry::type_t schedule_type = vistk::schedule_registry::default_type;

  if (vm.count("schedule"))
  {
    schedule_type = vm["schedule"].as<vistk::schedule_registry::type_t>();
  }

  vistk::config_t const schedule_config = conf->subblock(schedule_block + vistk::config::block_sep + schedule_type);

  vistk::schedule_registry_t reg = vistk::schedule_registry::self();

  vistk::schedule_t schedule = reg->create_schedule(schedule_type, schedule_config, pipe);

  if (!schedule)
  {
    std::cerr << "Error: Unable to create schedule" << std::endl;

    return EXIT_FAILURE;
  }

  schedule->start();
  schedule->wait();

  return EXIT_SUCCESS;
}

po::options_description
make_options()
{
  po::options_description desc;

  desc.add_options()
    ("help,h", "output help message and quit")
    ("pipeline,p", po::value_desc<vistk::path_t>()->metavar("FILE"), "pipeline")
    ("config,c", po::value_desc<vistk::paths_t>()->metavar("FILE"), "supplemental configuration file")
    ("setting,s", po::value_desc<std::vector<std::string> >()->metavar("VAR=VALUE"), "additional configuration")
    ("include,I", po::value_desc<vistk::paths_t>()->metavar("DIR"), "configuration include path")
    ("schedule,S", po::value_desc<vistk::schedule_registry::type_t>()->metavar("TYPE"), "schedule type")
  ;

  return desc;
}

void
usage(po::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(EXIT_FAILURE);
}
