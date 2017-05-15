/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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

#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/tools/tool_main.h>
#include <sprokit/tools/tool_usage.h>
#include <sprokit/tools/build_pipeline_from_options.h>

#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>
#include <sprokit/pipeline/pipeline.h>

#include <boost/program_options/variables_map.hpp>

#include <iostream>
#include <cstdlib>

static const auto scheduler_block = kwiver::vital::config_block_key_t("_scheduler");

int
sprokit_tool_main(int argc, char const* argv[])
{
  boost::program_options::options_description desc;
  desc
    .add(sprokit::tool_common_options())
    .add(sprokit::pipeline_common_options())
    .add(sprokit::pipeline_input_options())
    .add(sprokit::pipeline_run_options());

  boost::program_options::variables_map const vm = sprokit::tool_parse(argc, argv, desc, "");

    // Load all known modules
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();

  const sprokit::build_pipeline_from_options builder(vm, desc);

  sprokit::pipeline_t const pipe = builder.pipeline();
  kwiver::vital::config_block_sptr const conf = builder.config();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  pipe->setup_pipeline();

  auto scheduler_type = sprokit::scheduler_factory::default_type;

  if (vm.count("scheduler"))
  {
    scheduler_type = vm["scheduler"].as<sprokit::scheduler::type_t>();
  }
  else
  {
    scheduler_type = conf->get_value(
        scheduler_block + kwiver::vital::config_block::block_sep + "type",  // key string
        sprokit::scheduler_factory::default_type ); // default value
  }

  // Get scheduler sub block based on selected scheduler type
  kwiver::vital::config_block_sptr const scheduler_config = conf->subblock(scheduler_block +
                                              kwiver::vital::config_block::block_sep + scheduler_type);

  sprokit::scheduler_t scheduler = sprokit::create_scheduler(scheduler_type, pipe, scheduler_config);

  if (!scheduler)
  {
    std::cerr << "Error: Unable to create scheduler" << std::endl;

    return EXIT_FAILURE;
  }

  scheduler->start();
  scheduler->wait();

  return EXIT_SUCCESS;
}
