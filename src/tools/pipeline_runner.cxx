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

#include <boost/program_options/variables_map.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>

namespace po = boost::program_options;

static vistk::config::key_t const scheduler_block = vistk::config::key_t("_scheduler");

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(tool_common_options())
    .add(pipeline_common_options())
    .add(pipeline_input_options())
    .add(pipeline_run_options());

  boost::program_options::variables_map const vm = tool_parse(argc, argv, desc);

  if (!vm.count("pipeline"))
  {
    std::cerr << "Error: pipeline not set" << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  pipeline_builder const builder(vm);

  vistk::pipeline_t const pipe = builder.pipeline();
  vistk::config_t const conf = builder.config();

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
