/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "helpers/pipeline_builder.h"
#include "helpers/tool_io.h"
#include "helpers/tool_main.h"
#include "helpers/tool_usage.h"

#include <vistk/pipeline_util/export_dot.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/modules.h>
#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <string>
#include <vector>

#include <cstddef>
#include <cstdlib>

static boost::program_options::options_description pipe_to_dot_options();

int
tool_main(int argc, char* argv[])
{
  vistk::load_known_modules();

  boost::program_options::options_description desc;
  desc
    .add(tool_common_options())
    .add(pipeline_common_options())
    .add(pipeline_input_options())
    .add(pipeline_output_options())
    .add(pipe_to_dot_options());

  boost::program_options::variables_map const vm = tool_parse(argc, argv, desc);

  pipeline_builder const builder(vm, desc);

  vistk::pipeline_t const pipe = builder.pipeline();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  vistk::path_t const opath = vm["output"].as<vistk::path_t>();

  ostream_t const ostr = open_ostream(opath);

  std::string const graph_name = vm["name"].as<std::string>();

  if (vm.count("setup"))
  {
    pipe->setup_pipeline();
  }

  vistk::export_dot(*ostr, pipe, graph_name);

  return EXIT_SUCCESS;
}

boost::program_options::options_description
pipe_to_dot_options()
{
  boost::program_options::options_description desc;

  desc.add_options()
    ("name,n", boost::program_options::value<std::string>()->value_name("NAME")->default_value("unnamed"), "the name of the graph")
    ("setup", "whether to setup the pipeline before exporting or not")
  ;

  return desc;
}
