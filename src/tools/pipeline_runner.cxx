/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <sprokit/tools/pipeline_builder.h>
#include <sprokit/tools/tool_main.h>
#include <sprokit/tools/tool_usage.h>

#include <sprokit/pipeline_util/path.h>

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/modules.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_registry.h>
#include <sprokit/pipeline/pipeline.h>

#include <boost/program_options/variables_map.hpp>

#include <iostream>

#include <cstdlib>

static sprokit::config::key_t const scheduler_block = sprokit::config::key_t("_scheduler");

int
sprokit_tool_main(int argc, char const* argv[])
{
  sprokit::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(sprokit::tool_common_options())
    .add(sprokit::pipeline_common_options())
    .add(sprokit::pipeline_input_options())
    .add(sprokit::pipeline_run_options());

  boost::program_options::variables_map const vm = sprokit::tool_parse(argc, argv, desc);

  sprokit::pipeline_builder const builder(vm, desc);

  sprokit::pipeline_t const pipe = builder.pipeline();
  sprokit::config_t const conf = builder.config();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  pipe->setup_pipeline();

  sprokit::scheduler_registry::type_t scheduler_type = sprokit::scheduler_registry::default_type;

  if (vm.count("scheduler"))
  {
    scheduler_type = vm["scheduler"].as<sprokit::scheduler_registry::type_t>();
  }

  sprokit::config_t const scheduler_config = conf->subblock(scheduler_block + sprokit::config::block_sep + scheduler_type);

  sprokit::scheduler_registry_t reg = sprokit::scheduler_registry::self();

  sprokit::scheduler_t scheduler = reg->create_scheduler(scheduler_type, pipe, scheduler_config);

  if (!scheduler)
  {
    std::cerr << "Error: Unable to create scheduler" << std::endl;

    return EXIT_FAILURE;
  }

  scheduler->start();
  scheduler->wait();

  return EXIT_SUCCESS;
}
