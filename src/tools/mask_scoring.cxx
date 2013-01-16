/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/literal_pipeline.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/scheduler.h>
#include <vistk/pipeline/scheduler_registry.h>
#include <vistk/pipeline/pipeline.h>

#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/foreach.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>

static vistk::config::key_t const scheduler_block = vistk::config::key_t("_scheduler");

static boost::program_options::options_description mask_scoring_options();

static std::string base_pipeline();
static std::string layer_connection(std::string const& layer);

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(tool_common_options())
    .add(pipeline_common_options())
    .add(pipeline_run_options())
    .add(mask_scoring_options());

  boost::program_options::variables_map const vm = tool_parse(argc, argv, desc);

  if (!vm.count("layer"))
  {
    std::cerr << "Error: there must be at least one layer to read" << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  vistk::pipeline_t pipe;

  vistk::config_t conf;

  {
    std::stringstream sstr;

    sstr << base_pipeline();

    std::vector<std::string> const layers = vm["layer"].as<std::vector<std::string> >();

    BOOST_FOREACH (std::string const& layer, layers)
    {
      sstr << layer_connection(layer);
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

    pipeline_builder builder;

    builder.load_pipeline(sstr);
    builder.load_from_options(vm);

    if (vm.count("name"))
    {
      std::string const name = vm["name"].as<std::string>();

      builder.add_setting("mask_scoring:name=" + name);
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

boost::program_options::options_description
mask_scoring_options()
{
  boost::program_options::options_description desc;

  desc.add_options()
    ("dump,d", boost::program_options::value<vistk::path_t>()->value_name("FILE"), "output the generated pipeline")
    ("name,n", boost::program_options::value<std::string>()->value_name("NAME"), "the name of the run")
    ("layer,l", boost::program_options::value<std::vector<std::string> >()->value_name("LAYER"), "layer name")
  ;

  return desc;
}

std::string
base_pipeline()
{
  return
    CONFIG_BLOCK("mask_scoring")
      CONFIG("input", "image_list.txt")
      CONFIG("truth_input", "truth_list.txt")
      CONFIG("output", "output.txt")
      CONFIG("name", "(unnamed)")
    PROCESS("layered_image_reader", "truth_reader")
      CONFIG_FULL("path", "ro", "CONF", "mask_scoring:truth_input")
      CONFIG_FLAGS("pixfmt", "ro", "mask")
      CONFIG_FLAGS("pixtype", "ro", "byte")
    PROCESS("image_reader", "reader")
      CONFIG_FULL("input", "ro", "CONF", "mask_scoring:input")
      CONFIG_FLAGS("verify", "ro", "true")
      CONFIG_FLAGS("pixfmt", "ro", "mask")
      CONFIG_FLAGS("pixtype", "ro", "byte")
    PROCESS("combine_masks", "combine")
    PROCESS("mask_scoring", "scoring")
    PROCESS("score_aggregation", "aggregate")
    PROCESS("component_score_json_writer", "writer")
      CONFIG_FULL("path", "ro", "CONF", "mask_scoring:output")
      CONFIG_FULL("name", "ro", "CONF", "mask_scoring:name")

    CONNECT("combine", "mask",
            "scoring", "truth_mask")
    CONNECT("reader", "image",
            "scoring", "computed_mask")

    CONNECT("scoring", "result",
            "aggregate", "score")

    CONNECT("aggregate", "aggregate",
            "writer", "score/ALL")
    CONNECT("aggregate", "statistics",
            "writer", "stats/ALL")
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
    CONNECT("reader", "image",
            "scoring_" + layer +, "computed_mask")
    CONNECT("scoring_" + layer +, "result",
            "aggregate_" + layer +, "score")
    CONNECT("aggregate_" + layer +, "aggregate",
            "writer", "score/" + layer +)
    CONNECT("aggregate_" + layer +, "statistics",
            "writer", "stats/" + layer +)
  ;
}
