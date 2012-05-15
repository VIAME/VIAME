/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/literal_pipeline.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/schedule.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/pipeline.h>

#include <vistk/config.h>

#include <tools/helpers/typed_value_desc.h>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
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

static std::string base_pipeline();
static std::string layer_connection(std::string const& layer);

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

  if (!vm.count("layer"))
  {
    std::cerr << "Error: there must be at least one layer to read" << std::endl;
    usage(desc);
  }

  vistk::pipeline_t pipe;

  vistk::config_t conf;

  {
    std::stringstream sstr;

    sstr << base_pipeline();

    if (vm.count("layer"))
    {
      std::vector<std::string> const layers = vm["layer"].as<std::vector<std::string> >();

      BOOST_FOREACH (std::string const& layer, layers)
      {
        sstr << layer_connection(layer);
      }
    }

    if (vm.count("dump"))
    {
      std::ostream* postr;
      std::ofstream fout;

      vistk::path_t const opath = vm["dump"].as<vistk::path_t>();

      if (opath.native() == vistk::path_t("-"))
      {
        postr = &std::cout;
      }
      else
      {
        fout.open(opath.native().c_str());

        if (fout.bad())
        {
          std::cerr << "Error: Unable to open output file" << std::endl;

          return EXIT_FAILURE;
        }

        postr = &fout;
      }

      std::ostream& ostr = *postr;

      ostr << sstr.str();

      return EXIT_SUCCESS;
    }

    /// \todo Include paths?

    pipeline_builder builder;

    builder.load_pipeline(sstr);

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
    ("config,c", po::value_desc<vistk::paths_t>()->metavar("FILE"), "supplemental configuration file")
    ("setting,s", po::value_desc<std::vector<std::string> >()->metavar("VAR=VALUE"), "additional configuration")
    ("include,I", po::value_desc<vistk::paths_t>()->metavar("DIR"), "configuration include path")
    ("schedule,S", po::value_desc<vistk::schedule_registry::type_t>()->metavar("TYPE"), "schedule type")
    ("dump,d", po::value_desc<vistk::path_t>()->metavar("FILE"), "output the generated pipeline")
    ("layer,l", po::value_desc<std::vector<std::string> >()->metavar("LAYER"), "layer name")
  ;

  return desc;
}

void
usage(po::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(EXIT_FAILURE);
}

std::string
base_pipeline()
{
  return
    PROCESS("timestamp_reader", "timestamp")
    PROCESS("layered_image_reader", "truth_reader")
      CONFIG_FLAGS("pixfmt", "ro", "mask")
      CONFIG_FLAGS("pixtype", "ro", "byte")
    PROCESS("image_reader", "reader")
      CONFIG_FLAGS("verify", "ro", "true")
      CONFIG_FLAGS("pixfmt", "ro", "mask")
      CONFIG_FLAGS("pixtype", "ro", "byte")
    PROCESS("source", "source")
    PROCESS("combine_masks", "combine")
    PROCESS("mask_scoring", "scoring")
    PROCESS("score_aggregation", "aggregate")
    PROCESS("component_score_json_writer", "writer")

    CONNECT("reader", "image",
            "source", "src/computed_mask")
    CONNECT("timestamp", "timestamp",
            "source", "src/timestamp")

    CONNECT("source", "out/timestamp",
            "truth_reader", "timestamp")

    CONNECT("combine", "mask",
            "scoring", "truth_mask")
    CONNECT("source", "out/computed_mask",
            "scoring", "computed_mask")

    CONNECT("scoring", "result",
            "aggregate", "score")

    CONNECT("aggregate", "aggregate",
            "writer", "score")
  ;
}

std::string
layer_connection(std::string const& layer)
{
  return
    CONNECT("truth_reader", "image/" + layer +,
            "combine", "mask/" + layer +)
    PROCESS("mask_scoring", "scoring_" + layer +)
    PROCESS("score_aggregation", "aggregate_" + layer +)

    CONNECT("truth_reader", "image/" + layer +,
            "scoring_" + layer +, "truth_mask")
    CONNECT("source", "out/computed_mask",
            "scoring_" + layer +, "computed_mask")
    CONNECT("scoring_" + layer +, "result",
            "aggregate_" + layer +, "score")
    CONNECT("aggregate_" + layer +, "aggregate",
            "writer", "score/" + layer +)
  ;
}
